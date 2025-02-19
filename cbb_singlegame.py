import pandas as pd
from march_madness_setup.game_setup import *


# input season year
season = 2025

# manual input for date
today = pd.to_datetime("2025-02-14").strftime("%Y-%m-%d")
# automatic date (today)
today = datetime.now().strftime("%Y-%m-%d")

# matchup
away_tm = "Vanderbilt"
home_tm = "Kentucky"
n = 1001

# neutral site game
neutral_gm = False

# single game results (win probability and average point spread)
sg_win = game_sim(season, away_tm, home_tm, today, neutral_gm, n)

# graph single game results
sim_graph(season, away_tm, home_tm, sg_win)
