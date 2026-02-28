"""Shared constants for the Dynasty Model project."""

BBR_TEAM_MAPPING = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "Charlotte Bobcats": "CHA",
    "Charlotte Hornets": "CHO",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Jersey Nets": "NJN",
    "New Orleans Hornets": "NOH",
    "New Orleans Pelicans": "NOP",
    "New Orleans/Oklahoma City Hornets": "NOK",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Seattle SuperSonics": "SEA",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

# Canonical column rename map applied at data load time
COLUMN_RENAME_MAP = {
    # Per-game stat renames
    "FG": "FGM",
    "FT": "FTM",
    "3P": "3PM",
    "TRB": "REB",
    "Age": "AGE",
    "Tm": "TEAM",
    # Percentage renames
    "TS%": "TS_PCT",
    "USG%": "USG_PCT",
    "FG%": "FG_PCT",
    "3P%": "3P_PCT",
    "2P%": "2P_PCT",
    "FT%": "FT_PCT",
    "eFG%": "EFG_PCT",
    "ORB%": "ORB_PCT",
    "DRB%": "DRB_PCT",
    "TRB%": "REB_PCT",
    "AST%": "AST_PCT",
    "STL%": "STL_PCT",
    "BLK%": "BLK_PCT",
    "TOV%": "TOV_PCT",
    "WS/48": "WS_PER_48",
    # Per-100 stat renames
    "FG_per100": "FGM_per100",
    "FT_per100": "FTM_per100",
    "3P_per100": "3PM_per100",
    "TRB_per100": "REB_per100",
}

# Columns to drop at load time (ranking artifacts, awards)
COLUMNS_TO_DROP = ["Rk", "Rk_per100", "Awards", "Awards_per100"]
