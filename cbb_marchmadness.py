import pandas as pd
from march_madness_setup.march_madness import *
from march_madness_setup.interactive_march_madness_bracket import *


# input season year
season = 2025
seasons = [2023, 2024, 2025]

# manual input for date
today = pd.to_datetime("2025-03-19").strftime("%Y-%m-%d")

# automatic date (today)
# today = datetime.now().strftime("%Y-%m-%d")

# March Madness automatic simulation
run_type = "Model"  # ('Model', 'AutoEq', 'Interactive')
if run_type == "Model":
    bracketology_df = march_madness_model(seasons, today, n=1)
elif run_type == 'Interactive':
    bracketology_df = interactive_march_madness(season, today, 'Simulation')
else:
    bracketology_df = march_madness(season, today, 501)

# Visualize the bracket
with pd.ExcelWriter(
    f"~/OneDrive - Tennessee Titans/Documents/Python/professional_portfolio/march_madness/march_madness_bracket.xlsx",
    engine="openpyxl",
    mode="a",
    if_sheet_exists="replace",
) as writer:
    bracketology_df.to_excel(writer, sheet_name="marchmadness_data", index=False)
