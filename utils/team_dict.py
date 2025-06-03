import bidict
import pandas as pd


def get_teamnm():
    team_df = pd.read_csv(
        f"~/Documents/Python/professional_portfolio/march_madness/csv_files/team_df.csv"
    )

    return team_df
