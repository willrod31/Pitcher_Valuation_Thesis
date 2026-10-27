"""
Pitch-type names, short codes and colors for the dashboard.

How it works: the data stays keyed by Statcast's pitch codes (FF = four-seam fastball, ST = sweeper, ...); these
helpers turn a code into its full name or the industry-standard abbreviation only when something is drawn.
Each pitch type keeps one color everywhere (a colorblind-safe palette), listed fastballs first.

Imported by the dashboard pages; not run on its own.
"""

# Standard Statcast order (fastballs first, not alphabetical) and one fixed color each
PITCH_TYPE_ORDER = ["FF", "SI", "FC", "SL", "ST", "CU", "KC", "CH", "FS", "FO", "SV"]
PITCH_TYPE_COLORS = dict(zip(PITCH_TYPE_ORDER, [
    "#0072B2", "#56B4E9", "#009E73", "#D55E00", "#E69F00",
    "#CC79A7", "#F0E442", "#000000", "#999999", "#882255", "#44AA99",
]))
PITCH_TYPE_NAMES = {
    "FF": "Four-Seam Fastball", "SI": "Sinker", "FC": "Cutter",
    "SL": "Slider", "ST": "Sweeper", "CU": "Curveball", "KC": "Knuckle Curve",
    "CS": "Slow Curve", "SV": "Slurve", "CH": "Changeup", "FS": "Splitter",
    "FO": "Forkball", "SC": "Screwball", "KN": "Knuckleball", "EP": "Eephus",
}
# Statcast code -> industry-standard abbreviation shown on screen
PITCH_TYPE_ABBR = {
    "FF": "FB", "SI": "SI", "FC": "CT", "SL": "SL", "ST": "SW", "CU": "CU", "KC": "KC",
    "CS": "CS", "SV": "SV", "CH": "CH", "FS": "SPL", "FO": "FK", "SC": "SC", "KN": "KN", "EP": "EP",
}


def pitch_abbr(code: str) -> str:
    """Display abbreviation ("ST" -> "SW"); unknown codes pass through."""
    return PITCH_TYPE_ABBR.get(code, code)


def pitch_name(code: str) -> str:
    """Full pitch name ("ST" -> "Sweeper"); unknown codes pass through."""
    return PITCH_TYPE_NAMES.get(code, code)


def pitch_label(code: str) -> str:
    """ "SW (Sweeper)", for hovers and tables. """
    name = PITCH_TYPE_NAMES.get(code)
    return f"{pitch_abbr(code)} ({name})" if name else pitch_abbr(code)
