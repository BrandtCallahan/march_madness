import bidict
import pandas as pd


def get_teamnm():
    team_df = pd.read_csv(
        f"~/OneDrive - Tennessee Titans/Documents/Python/professional_portfolio/march_madness/csv_files/team_df.csv"
    )

    return team_df


tmname_dict = bidict.bidict(
    alabama="Alabama",
    arkansas="Arkansas",
    auburn="Auburn",
    florida="Florida",
    georgia="Georgia",
    kentucky="Kentucky",
    lsu="LSU",
    olemiss="Ole Miss",
    mississippist="Mississppi State",
    missouri="Missouri",
    oklahoma="Oklahoma",
    southcarolina="South Carolina",
    tennessee="Tennessee",
    texas="Texas",
    texasam="Texas A&M",
    vanderbilt="Vanderbilt",
    jacksonst="Jackson State",
)

reference_dict = bidict.bidict(
    alabama="alabama",
    arkansas="arkansas",
    auburn="auburn",
    florida="florida",
    georgia="georgia",
    kentucky="kentucky",
    lsu="louisiana-state",
    olemiss="mississippi",
    mississippist="mississppi-state",
    missouri="missouri",
    oklahoma="oklahoma",
    southcarolina="south-carolina",
    tennessee="tennessee",
    texas="texas",
    texasam="texas-am",
    vanderbilt="vanderbilt",
    jacksonst="jackson-state",
)

conference_dict = {
    "Alabama": "SEC",
    "Arkansas": "SEC",
    "Auburn": "SEC",
    "Florida": "SEC",
    "Georgia": "SEC",
    "Kentucky": "SEC",
    "LSU": "SEC",
    "Ole Miss": "SEC",
    "Mississippi State": "SEC",
    "Missouri": "SEC",
    "Oklahoma": "SEC",
    "South Carolina": "SEC",
    "Tennessee": "SEC",
    "Texas": "SEC",
    "Texas A&M": "SEC",
    "Vanderbilt": "SEC",
    "Jackson State": "SWAC",
}
