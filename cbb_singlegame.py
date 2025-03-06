import pandas as pd
from single_game_setup.singlegame_setup import *


# input season year
season = 2025

# manual input for date
today = pd.to_datetime("2025-02-19").strftime("%Y-%m-%d")
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

# donut chart win probability for single game
## hm_tm_prim/aw_tm_prim == True: will show the team's primary color on chart
## == False: will show the team's secondary color
sim_donut_graph(season, away_tm, home_tm, sg_win, hm_tm_prim=True, aw_tm_prim=True)
