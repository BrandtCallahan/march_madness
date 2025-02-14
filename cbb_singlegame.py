import pandas as pd
from march_madness_setup.play_matchup import *


# input season year
season = 2025

# manual input for date
today = pd.to_datetime("2025-02-14").strftime("%Y-%m-%d")
# automatic date (today)
today = datetime.now().strftime("%Y-%m-%d")

tm_df = tm_rating(season, today)

# matchup
away_tm = 'Samford'
home_tm = 'Kansas'

# neutral site game
neutral_gm = True

# single game results (win probability and average point spread)
sg_win = single_matchup(away_tm, home_tm, tm_df, neutral_gm)

# graph single game results
graph_win_prob(away_tm, home_tm, sg_win['Point Diff'][0], neutral_gm, 3.5, 11)
