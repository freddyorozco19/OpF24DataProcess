# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 04:27:41 2025
@author: Freddy J. Orozco R.
@Powered: WinStats.
"""

import streamlit as st
import datetime
import base64
import pandas as pd
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as mplt
import matplotlib.font_manager as font_manager
import mplsoccer
from mplsoccer import Pitch, VerticalPitch, FontManager
import sklearn
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
from scipy.ndimage import gaussian_filter
import seaborn as sns
from matplotlib import colors as mcolors
import requests
#from PIL import Image
from matplotlib.patches import Rectangle
import math
from PIL import Image

################################################################################################################################################################################################################################################################

im = Image.open("Resources/Isotipo-FF046.ico")
st.set_page_config(layout="wide", page_icon=im)
st.logo("Resources/Isotipo-FF046.png")
navigation_tree = {"Main": [
        #st.Page("main/OptaEventingData.py", title="Extract Eventing Data", icon=":material/download:"),
        #st.Page("main/OptaJoinEventingData.py", title="Join Eventing Data", icon=":material/cell_merge:"),
        #st.Page("main/OptaExploreEventingData.py", title="Explore Eventing Data", icon=":material/sports_and_outdoors:"),
        #st.Page("main/OptaExploreMatchData.py", title="Explore Match Data", icon=":material/search_insights:"),
        #st.Page("main/OptaExploreTeamData.py", title="Explore Team Data", icon=":material/reduce_capacity:"),
        #st.Page("main/OptaExploreLeagueData.py", title="Explore League Data", icon=":material/analytics:"),   
        #st.Page("main/OptaExtractProMatchData.py", title="Extract Pro Match Data", icon=":material/leaderboard:"),
        st.Page("main/OpF24ExtractData.py", title="F24 Extract Data", icon=":material/lists:"),
        st.Page("main/OpF24RegisterData.py", title="F24 Register Data", icon=":material/download:")]}
nav = st.navigation(navigation_tree, position="sidebar")
nav.run()
st.sidebar.write("2025 WS")
