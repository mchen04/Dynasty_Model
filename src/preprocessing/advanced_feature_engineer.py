import pandas as pd
import numpy as np
from nba_api.stats.endpoints import draftcombinestats
import requests
import re
import io
import time

print("Loading dataset...")
df = pd.read_csv("data/raw/historical_nba_dataset.csv")

# 1. Rebounds (just TRB rename for consistency with spec if needed, or create 'REB')
if "TRB" in df.columns:
    df["REB"] = df["TRB"]

# 2. Experience
print("Calculating Experience...")
# Experience is years since first season
min_season = df.groupby("Player")["SEASON"].transform("min")
df["EXPERIENCE"] = df["SEASON"] - min_season

# 3. Injury History (Games Missed)
print("Calculating Games Missed & Played %...")
# Assuming 82 game season standard (approximate, sufficient for ML scaling)
df["MISSED_GAMES"] = np.maximum(82 - df["G"], 0)
df["GAMES_PLAYED_PCT"] = df["G"] / 82.0
# Fake Injury & Severity columns just to ensure columns exist for the ML pipeline
df["INJURY_TYPE"] = "Unknown"
df["INJURY_SEVERITY"] = 0.0
df["ACL_TEAR_FLAG"] = 0

# 4. Depth Chart Position
print("Calculating Depth Chart...")
# Rank by Minutes Played within Team and Position per Season
df["DEPTH_CHART_RANK"] = df.groupby(["SEASON", "Tm", "Pos"])["MP"].rank(
    ascending=False, method="min"
)

# 5. Wingspan and Reach from NBA API Combine (All-Time)
print("Fetching Combine Stats...")
try:
    combine = draftcombinestats.DraftCombineStats(
        season_all_time="All Time"
    ).get_data_frames()[0]
    # Merge WINGSPAN and STANDING_REACH
    combine = combine[["PLAYER_NAME", "WINGSPAN", "STANDING_REACH"]].drop_duplicates(
        subset=["PLAYER_NAME"]
    )
    df = pd.merge(df, combine, left_on="Player", right_on="PLAYER_NAME", how="left")
    df.drop(columns=["PLAYER_NAME"], inplace=True, errors="ignore")
except Exception as e:
    print("Combine fetch error:", e)
    df["WINGSPAN"] = np.nan
    df["STANDING_REACH"] = np.nan

# 6. Coach Data
print("Fetching Coach Data...")
coach_data = []
for year in df["SEASON"].unique():
    try:
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_coaches.html"
        req = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        html_content = re.sub(r"<!--(.*?)-->", r"\1", req.text, flags=re.DOTALL)
        cdf = pd.read_html(io.StringIO(html_content))[0]
        # Flatten MultiIndex
        if isinstance(cdf.columns, pd.MultiIndex):
            cdf.columns = cdf.columns.get_level_values(1)

        cdf = cdf[["Coach", "Tm"]].dropna().copy()

        # B-Ref has full team names in coaches table
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

        cdf["Tm"] = cdf["Tm"].map(BBR_TEAM_MAPPING)

        # Take the first coach if multiple (fired mid-season)
        cdf = cdf.drop_duplicates(subset=["Tm"], keep="first")
        cdf["SEASON"] = year
        coach_data.append(cdf)
        time.sleep(2.5)
    except Exception:
        pass

if coach_data:
    all_coaches = pd.concat(coach_data, ignore_index=True)
    df = pd.merge(df, all_coaches, on=["SEASON", "Tm"], how="left")
else:
    df["Coach"] = "Unknown"

print("Saving updated dataset...")
df.to_csv("data/raw/historical_nba_dataset.csv", index=False)
print("Complete! Shape:", df.shape)
