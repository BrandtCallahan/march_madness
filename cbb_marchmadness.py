import pandas as pd
from march_madness_setup.march_madness import *
from march_madness_setup.interactive_march_madness_bracket import *


# input season year
season = 2025

# manual input for date
today = pd.to_datetime("2025-03-19").strftime("%Y-%m-%d")
# automatic date (today)
today = datetime.now().strftime("%Y-%m-%d")

# March Madness automatic simulation
bracketology_df = march_madness(season, today, 501)

# Interactive March Madness bracket
#   User input
modeltype = "Model"  # ('Simulation'/'Model')
bracketology_df = interactive_march_madness(season, today, modeltype)
