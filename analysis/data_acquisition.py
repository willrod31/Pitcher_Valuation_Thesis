"""
Loads public Statcast pitch data (via the pybaseball library), caches it, and cleans it for every other file.

In simple terms: Statcast is MLB's tracking system; it records every pitch's speed, spin, movement, release point
and where it crossed the plate. This file downloads it once, saves it to disk, and removes bad readings.

How it works:
    1. Download a date range, one calendar month at a time (a single six-month league-wide pull runs out of
       memory on an 8GB machine), and cache each month in data/raw/statcast/.
    2. Keep pitchers only: position players who pitched a mop-up inning are dropped (MLB Stats API positions).
    3. Attach names (from the numeric pitcher id) and each pitch's team (the team in the field).
    4. Clean: IVB and HB (induced vertical break, i.e. rise, and horizontal break) are converted from feet to
       inches, and pitches with impossible values or no pitch type are removed.
Also: one-pitcher and one-team pulls, team rosters, and the "primary team" of a traded pitcher.

Inputs: Baseball Savant via pybaseball, MLB Stats API. Outputs: cached parquet files in data/raw/statcast/ and
data/player_positions.csv. Imported by the other files; not run on its own.
"""
import functools
import shutil

import numpy as np
import pandas as pd
import requests

import pybaseball as pb

from config import END_DATE, MIN_PITCHES_FOR_INCLUSION, START_DATE
from analysis.utils import CACHE_DIR, STATCAST_LEAGUE_DIR, STATCAST_PITCHER_DIR, STATCAST_TEAM_DIR

# Pitch-type codes dropped in step 4: PO pitchout, IN intentional ball, EP eephus, FA other, UN unknown, SC screwball
NON_PITCH_TYPES = {"PO", "IN", "EP", "FA", "UN", "SC"}

# Primary position per MLBAM id (MLB's player id), from the MLB Stats API, cached. Only pitchers ("1") and
# two-way players ("Y") are kept anywhere in the pipeline (step 2, pitchers_only)
POSITIONS_PATH = CACHE_DIR / "player_positions.csv"
PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"
PITCHER_POSITION_CODES = {"1", "Y"}
TWO_WAY_POSITION_CODE = "Y"


def fetch_statcast(start_date: str = START_DATE, end_date: str = END_DATE, force_refresh: bool = False) -> pd.DataFrame:
    """Every league pitch in a date range, from the cache on disk or downloaded and cached."""
    raw_path = STATCAST_LEAGUE_DIR / f"statcast_raw_{start_date}_{end_date}.parquet"
    if raw_path.exists() and not force_refresh:
        return pd.read_parquet(raw_path)

    pb.cache.enable()
    df = pb.statcast(start_dt=start_date, end_dt=end_date)
    df.to_parquet(raw_path, index=False)
    return df


def fetch_statcast_pitcher(pitcher_name: str, start_date: str = START_DATE, end_date: str = END_DATE, force_refresh: bool = False) -> pd.DataFrame:
    """One pitcher's pitches in a date range, cached; a few thousand rows instead of the league's ~700,000."""
    slug = pitcher_name.strip().lower().replace(" ", "_")
    raw_path = STATCAST_PITCHER_DIR / f"statcast_pitcher_{slug}_{start_date}_{end_date}.parquet"
    if raw_path.exists() and not force_refresh:
        return pd.read_parquet(raw_path)

    mlbam_id = lookup_mlbam_id(pitcher_name)

    pb.cache.enable()
    df = pb.statcast_pitcher(start_date, end_date, mlbam_id)
    df.to_parquet(raw_path, index=False)
    return df


def lookup_mlbam_id(pitcher_name: str) -> int:
    """The MLBAM id for a "First Last" name (the most recently active match wins)."""
    first, last = pitcher_name.strip().split(" ", 1)
    lookup = pb.playerid_lookup(last, first)
    if lookup.empty:
        raise ValueError(f"No player found for name '{pitcher_name}'")
    return int(lookup.sort_values("mlb_played_last", ascending=False)["key_mlbam"].iloc[0])


def load_pitcher_season(pitcher_name: str, start_date: str = START_DATE, end_date: str = END_DATE, force_refresh: bool = False) -> pd.DataFrame:
    """One pitcher's season, named and cleaned like load_season()."""
    raw = fetch_statcast_pitcher(pitcher_name, start_date, end_date, force_refresh=force_refresh)
    if raw.empty:
        raise ValueError(f"No Statcast pitches found for '{pitcher_name}' between {start_date} and {end_date}")
    df = attach_pitcher_names(raw)
    df = attach_pitcher_team(df)
    df = clean_pitch_physics(df)
    return df


def fetch_statcast_team(team: str, start_date: str = START_DATE, end_date: str = END_DATE, force_refresh: bool = False) -> pd.DataFrame:
    """Every pitch one team's pitchers threw in a date range, cached (about 20-25 thousand rows a season).

    pybaseball's `team` filter returns only the pitches where that team is pitching (checked), not every pitch
    of its games.
    """
    raw_path = STATCAST_TEAM_DIR / f"statcast_team_{team}_{start_date}_{end_date}.parquet"
    if raw_path.exists() and not force_refresh:
        return pd.read_parquet(raw_path)

    pb.cache.enable()
    df = pb.statcast(start_dt=start_date, end_dt=end_date, team=team)
    df.to_parquet(raw_path, index=False)
    return df


def load_team_season(team: str, start_date: str = START_DATE, end_date: str = END_DATE, force_refresh: bool = False) -> pd.DataFrame:
    """One team's pitches, named and cleaned like load_season(); a traded pitcher has only his time with `team`."""
    raw = fetch_statcast_team(team, start_date, end_date, force_refresh=force_refresh)
    if raw.empty:
        raise ValueError(f"No Statcast pitches found for team '{team}' between {start_date} and {end_date}")
    df = attach_pitcher_names(raw)
    df = attach_pitcher_team(df)
    df = clean_pitch_physics(df)
    return df


def seed_team_caches_from_league(teams, start_date: str = START_DATE, end_date: str = END_DATE) -> list[str]:
    """The teams whose team cache was written, cut out of the league monthly caches instead of downloaded.

    A team's cache is the league rows where that team is in the field (checked identical for PIT). Months are
    split one at a time into per-team part files, so memory holds one month at most.
    """
    todo = [t for t in teams if not (STATCAST_TEAM_DIR / f"statcast_team_{t}_{start_date}_{end_date}.parquet").exists()]
    if not todo:
        return []
    parts_dir = STATCAST_TEAM_DIR / "_parts"
    parts_dir.mkdir(exist_ok=True)
    try:
        for n, (chunk_start, chunk_end) in enumerate(_month_chunks(start_date, end_date)):
            raw = fetch_statcast(chunk_start, chunk_end)
            fielding = np.where(raw["inning_topbot"] == "Top", raw["home_team"], raw["away_team"])
            for team in todo:
                raw[fielding == team].to_parquet(parts_dir / f"{team}_{n}.parquet", index=False)
            del raw
        for team in todo:
            parts = sorted(parts_dir.glob(f"{team}_*.parquet"))
            df = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True)
            df.to_parquet(STATCAST_TEAM_DIR / f"statcast_team_{team}_{start_date}_{end_date}.parquet", index=False)
    finally:
        shutil.rmtree(parts_dir, ignore_errors=True)
    return todo


def team_roster_ids(team: str, start_date: str = START_DATE, end_date: str = END_DATE, force_refresh: bool = False) -> pd.DataFrame:
    """PitcherId and Pitcher for everyone who threw for `team` in this range (remembered for the rest of the run)."""
    if force_refresh:
        _team_roster_ids.cache_clear()
    return _team_roster_ids(team, start_date, end_date, force_refresh).copy()


@functools.lru_cache(maxsize=None)
def _team_roster_ids(team: str, start_date: str, end_date: str, force_refresh: bool) -> pd.DataFrame:
    """team_roster_ids without the copy (this is the cached part)."""
    cached = STATCAST_TEAM_DIR / f"statcast_team_{team}_{start_date}_{end_date}.parquet"
    if cached.exists() and not force_refresh:
        raw = pd.read_parquet(cached, columns=["pitcher"])   # only the ids are needed
    else:
        raw = fetch_statcast_team(team, start_date, end_date, force_refresh=force_refresh)
    named = attach_pitcher_names(pitchers_only(raw[["pitcher"]].drop_duplicates()))
    named = named.dropna(subset=["Pitcher"]).rename(columns={"pitcher": "PitcherId"})
    named["PitcherId"] = named["PitcherId"].astype(int)
    return named.sort_values("Pitcher").reset_index(drop=True)


def team_roster(team: str, start_date: str = START_DATE, end_date: str = END_DATE, force_refresh: bool = False) -> list[str]:
    """Names of every pitcher who threw for `team` at any point in this range."""
    return team_roster_ids(team, start_date, end_date, force_refresh=force_refresh)["Pitcher"].tolist()


def load_team_roster_full_seasons(team: str, start_date: str = START_DATE, end_date: str = END_DATE, force_refresh: bool = False) -> pd.DataFrame:
    """Every pitcher who threw for `team`, each with his FULL season (other teams included), labeled `team`."""
    roster = team_roster(team, start_date, end_date, force_refresh=force_refresh)
    frames = []
    for name in roster:
        player_df = load_pitcher_season(name, start_date, end_date, force_refresh=force_refresh)
        player_df = player_df.copy()
        player_df["PitcherTeam"] = team
        frames.append(player_df)
    return pd.concat(frames, ignore_index=True)


def attach_pitcher_names(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: the pitches with a Pitcher name column, looked up from the numeric pitcher id.

    The id is used rather than Statcast's player_name column, because which player that column names depends on
    how the query was run.
    """
    df = df.copy()
    ids = df["pitcher"].dropna().unique().astype(int).tolist()
    lookup = pb.playerid_reverse_lookup(ids, key_type="mlbam")
    lookup = lookup.assign(
        Pitcher=lookup["name_first"].str.title() + " " + lookup["name_last"].str.title()
    )
    name_map = dict(zip(lookup["key_mlbam"], lookup["Pitcher"]))
    df["Pitcher"] = df["pitcher"].map(name_map)
    return df


def attach_pitcher_team(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: the pitches with PitcherTeam = the team in the field on that pitch (so a traded pitcher is split correctly)."""
    df = df.copy()
    df["PitcherTeam"] = np.where(df["inning_topbot"] == "Top", df["home_team"], df["away_team"])
    return df


def clean_pitch_physics(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4: the pitches with ivb / hb in inches, keeping only realistic values and real pitch types."""
    df = df.copy()
    df["ivb"] = df["pfx_z"] * 12   # feet -> inches
    df["hb"] = df["pfx_x"] * 12
    # realistic ranges: speed (mph), spin (rpm), extension (feet), break (inches)

    df = df[df["release_speed"].between(60, 105)]
    df = df[df["release_spin_rate"].between(1000, 3600)]
    df = df[df["release_extension"].between(3, 9)]
    df = df[df["ivb"].between(-25, 30)]
    df = df[df["hb"].between(-25, 25)]
    df = df[df["pitch_type"].notna()]
    df = df[~df["pitch_type"].isin(NON_PITCH_TYPES)]
    return df


def load_season(start_date: str = START_DATE, end_date: str = END_DATE, force_refresh: bool = False) -> pd.DataFrame:
    """A date range in one download, named and cleaned (no pitch minimum: callers apply their own)."""
    raw = fetch_statcast(start_date, end_date, force_refresh=force_refresh)
    df = attach_pitcher_names(raw)
    df = attach_pitcher_team(df)
    df = clean_pitch_physics(df)
    return df


def _month_chunks(start_date: str, end_date: str):
    """(start, end) date strings for each calendar month in a date range."""
    chunk_start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    while chunk_start <= end:
        chunk_end = min(chunk_start + pd.offsets.MonthEnd(0), end)
        yield chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        chunk_start = chunk_end + pd.Timedelta(days=1)


def load_season_monthly(start_date: str = START_DATE, end_date: str = END_DATE, columns: list[str] | None = None, force_refresh: bool = False) -> pd.DataFrame:
    """Steps 1-4: a date range loaded one calendar month at a time, pitchers only, named and cleaned.

    One six-month league-wide download runs out of memory on an 8GB machine (peak above 7GB); one month at a time
    stays near 3.3GB, and each month is cached. Pass `columns` to drop the other Statcast columns right after each
    month; it must include pitcher, home_team, away_team, inning_topbot, pfx_z, pfx_x, release_speed,
    release_spin_rate, release_extension and pitch_type (needed by steps 3-4).
    """
    chunks = []
    for chunk_start, chunk_end in _month_chunks(start_date, end_date):
        raw = fetch_statcast(chunk_start, chunk_end, force_refresh=force_refresh)
        if columns is not None:
            raw = raw[columns]
        chunks.append(raw)

    raw = pitchers_only(pd.concat(chunks, ignore_index=True))
    df = attach_pitcher_names(raw)
    df = attach_pitcher_team(df)
    df = clean_pitch_physics(df)
    return df


def fetch_positions(mlbam_ids) -> pd.DataFrame:
    """PitcherId, Name, PositionCode and PositionAbbrev for every id (MLB Stats API, in batches, cached)."""
    cache = pd.read_csv(POSITIONS_PATH, dtype={"PositionCode": str}) if POSITIONS_PATH.exists() else \
        pd.DataFrame(columns=["PitcherId", "Name", "PositionCode", "PositionAbbrev"])
    ids = sorted({int(i) for i in mlbam_ids})
    missing = [i for i in ids if i not in set(cache["PitcherId"])]
    rows = []
    for n in range(0, len(missing), 100):
        batch = missing[n:n + 100]
        resp = requests.get(PEOPLE_URL, params={"personIds": ",".join(map(str, batch))}, timeout=30)
        resp.raise_for_status()
        for person in resp.json().get("people", []):
            position = person.get("primaryPosition", {})
            rows.append({"PitcherId": person["id"], "Name": person.get("fullName"),
                         "PositionCode": str(position.get("code")), "PositionAbbrev": position.get("abbreviation")})
    if rows:
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True).sort_values("PitcherId")
        cache.to_csv(POSITIONS_PATH, index=False)
    return cache[cache["PitcherId"].isin(ids)].reset_index(drop=True)


def non_pitcher_ids(mlbam_ids) -> set[int]:
    """Ids whose main position is neither pitcher nor two-way (ids the API doesn't know are kept)."""
    positions = fetch_positions(mlbam_ids)
    return set(positions.loc[~positions["PositionCode"].isin(PITCHER_POSITION_CODES), "PitcherId"].astype(int))


def pitchers_only(df: pd.DataFrame, id_col: str = "pitcher") -> pd.DataFrame:
    """Step 2: the rows without position players (main position not pitcher or two-way)."""
    return df[~df[id_col].isin(non_pitcher_ids(df[id_col].dropna().unique()))]


def filter_qualified(df: pd.DataFrame, min_pitches: int = MIN_PITCHES_FOR_INCLUSION) -> pd.DataFrame:
    """The rows of pitchers with at least `min_pitches` pitches (counted by id, so two same-named pitchers stay apart)."""
    counts = df.groupby("pitcher").size()
    qualified = counts[counts >= min_pitches].index
    return df[df["pitcher"].isin(qualified)]


def primary_team(df: pd.DataFrame) -> pd.Series:
    """The team each pitcher threw the most pitches for, indexed by id (a traded pitcher's one league-wide label)."""
    counts = df.groupby(["pitcher", "PitcherTeam"]).size().rename("n").reset_index()
    top = counts.sort_values("n", ascending=False).drop_duplicates("pitcher")
    return top.set_index("pitcher")["PitcherTeam"]


def filter_to_team(graded: pd.DataFrame, team: str) -> pd.DataFrame:
    """The graded rows of pitchers on `team`'s roster (traded pitchers included, graded on their full season), labeled `team`."""
    roster_ids = set(team_roster_ids(team)["PitcherId"])
    return graded[graded["PitcherId"].isin(roster_ids)].assign(PitcherTeam=team)
