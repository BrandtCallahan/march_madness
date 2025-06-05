import pandas as pd
from single_game_setup.singlegame_setup import *
from single_game_setup.ml_singlegame_setup import *


# input season year
season = 2025

# manual input for date
today = pd.to_datetime("2025-01-18").strftime("%Y-%m-%d")
# automatic date (today)
today = datetime.now().strftime("%Y-%m-%d")

# matchup
away_tm = "Tennessee"
home_tm = "Vanderbilt"

# model type
model_type = 'Model'

if model_type == 'Simulation':
    # number of simulation iterations
    n = 301
    # neutral site game
    neutral_gm = False  # NCAA Tournament

    # single game results (win probability and average point spread)
    sg_win = game_sim(season, away_tm, home_tm, today, neutral_gm, n)

elif model_type == 'Model':
    sg_win = single_game_model(data_seasons=[2025], today=today, matchup=f'{away_tm} vs. {home_tm}')[3]

# donut chart for single game
sim_donut_graph(season, away_tm, home_tm, sg_win, hm_tm_prim=True, aw_tm_prim=True)
