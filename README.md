# March Madness


### Set Up:
The .csv files are built in a folder heirarchy that will need to either be mimicked or changed when you run the code. The currect folder heirarchy:

f"~/Documents/Python/professional_portfolio/march_madness/..."

This is the base directory (starting point) for this entire repo. Once you set the base directory you will be able to run everything from the "march_madness" directory folder.

## Single Game Win Probability
If you are looking to run a matchup for a single game then you should use the "cbb_singlegame.py" file. Here you will be able to setup a single game matchup between two teams from any point in the season. You will need to input the season you are wanting to use as well as the base day that you want the game to "happen on." This base day will limit the game data when calculating a team's efficiency rating. 

The single game view uses two teams' efficiencies (offense and defense), effective field goal percentages', and their respective average possessions per game to simulate a full game and then generate a point differential for the game as well as a win probability. If you would like to see the win probability and point spread visualized you can use the "sim_graph" function to generate that visual.

Here is an example of the single game graphing results from the Vanderbilt @ Kentucky game on 2025-02-19.

![van_ken_bball2](https://github.com/user-attachments/assets/731bc7a9-17bf-4f4d-96e7-85b2c2987c3a)


## March Madness Bracket
If you want to simulate a march madness bracket for a year you will want to use the "cbb_marchmadness.py" file. Here you will be able to input the season and the base day to anchor your stats. Once you have those defined you will run the "march_madness" function. This function is set up as a Monte Carlo simulation in order to run through multiple bracket scenarios. The function will tabulate each simulation and output a dataframe that shows a teams percentage chance to advance to each round. 

Another option in the "cbb_marchmadness.py" file is to manually pick winners given their predicted winning percentage. This is done through the the "interactive_march_madness" function. When running this you will be able to choose the winner of each game and your bracket will fill out in accordance to those picks. As you move through the bracket you will be prompted to enter your choosen winner of each game. The screen will prompt you with the winning percentage and predicted point spread of the game, but you have the ultimate choice of who you want to win that game. 
