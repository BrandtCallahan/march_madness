from datetime import datetime, date

from unidecode import unidecode

import re

import pandas as pd
import requests
from bs4 import BeautifulSoup
from logzero import logger

from beautiful_soup_helper import get_soup_from_url


# functions for finding and pull tables taken from GitHub
def findTables(url):
    res = requests.get(url)
    # The next two lines get around the issue with comments breaking the parsing.
    comm = re.compile("<!--|-->")
    soup = BeautifulSoup(comm.sub("", res.text), features="html.parser")  # 'lxml'
    divs = soup.findAll("div", id="content")
    divs = divs[0].findAll("div", id=re.compile("^all"))
    ids = []
    for div in divs:
        searchme = str(div.findAll("table"))
        x = searchme[searchme.find("id=") + 3 : searchme.find(">")]
        x = x.replace('"', "")
        if len(x) > 0:
            ids.append(x)
    return ids


def pullTable(url, tableID):
    res = requests.get(url)
    # Work around comments
    comm = re.compile("<!--|-->")
    soup = BeautifulSoup(comm.sub("", res.text), features="html.parser")  # 'lxml'
    tables = soup.findAll("table", id=tableID)
    data_rows = tables[0].findAll("tr")
    data_header = tables[0].findAll("thead")
    data_header = data_header[0].findAll("tr")
    data_header = data_header[1].findAll("th")
    game_data = [
        [td.getText() for td in data_rows[i].findAll(["th", "td"])]
        for i in range(len(data_rows))
    ]
    data = pd.DataFrame(game_data)
    header = []
    for i in range(len(data.columns)):
        header.append(data_header[i].getText())
    data.columns = header
    data = data.loc[data[header[0]] != header[0]]
    data = data.reset_index(drop=True)
    data = data.iloc[1:]

    index_mapping = {
        2: "Location",
        5: "Tm Score",
        6: "Opp Score",
        24: "Opp FG",
        25: "Opp FGA",
        26: "Opp FG%",
        27: "Opp 3P",
        28: "Opp 3PA",
        29: "Opp 3P%",
        30: "Opp FT",
        31: "Opp FTA",
        32: "Opp FT%",
        33: "Opp ORB",
        34: "Opp TRB",
        35: "Opp AST",
        36: "Opp STL",
        37: "Opp BLK",
        38: "Opp TOV",
        39: "Opp PF",
    }
    data.columns = [index_mapping.get(i, col) for i, col in enumerate(data.columns)]

    return data
