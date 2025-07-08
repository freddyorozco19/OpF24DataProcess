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
from PIL import Image
from matplotlib.patches import Rectangle
import math
import altair as alt
import json
import re
import sklearn
from sklearn.preprocessing import StandardScaler

################################################################################################################################################################################################################################################################################################################################################

def reparar_y_extraer(cadena):
    # Reparar claves sin comillas: 1": → "1":
    cadena = re.sub(r'(?<=\{|\s)(\d+)":', r'"\1":', cadena)
    cadena = cadena.replace('null', 'null')
    # Buscar todos los bloques clave + valor
    patron = r'"(\d+)":\s*\{[^{}]*?"value":\s*(null|"[^"]*"|[\d.]+)'
    matches = re.findall(patron, cadena)
    resultado = []
    for clave, valor in matches:
        if valor.startswith('"') and valor.endswith('"'):
            valor = valor[1:-1]
        elif valor == 'null':
            valor = 'null'
        resultado.append(f'{clave}:{valor}')
    return ', '.join(resultado)

st.markdown("<style> div { text-align: center } </style>", unsafe_allow_html=True)
st.subheader('EXTRACT DATA')


with st.form(key='form4'):
    uploaded_file = st.file_uploader("Choose a csv file", type="csv")
    submit_button2 = st.form_submit_button(label='Aceptar')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=';')


df = df.iloc[6:].reset_index(drop=True)
df = df.drop(columns=['timestamp', 'date'])

#Replace Type_id values
df['type_id'] = df['type_id'].replace(1, 'Pass')
df['type_id'] = df['type_id'].replace(5, 'Out')
df['type_id'] = df['type_id'].replace(61, 'Ball touch')
df['type_id'] = df['type_id'].replace(6, 'Corner awarded')
df['type_id'] = df['type_id'].replace(12, 'Clearance')
df['type_id'] = df['type_id'].replace(8, 'Interception')
df['type_id'] = df['type_id'].replace(49, 'Ball recovery')
df['type_id'] = df['type_id'].replace(4, 'Foul')
df['type_id'] = df['type_id'].replace(13, 'Shot Off Target')
df['type_id'] = df['type_id'].replace(10, 'Save')
df['type_id'] = df['type_id'].replace(15, 'Shot On Target')
df['type_id'] = df['type_id'].replace(16, 'Goal')
df['type_id'] = df['type_id'].replace(43, 'Deleted Event')
df['type_id'] = df['type_id'].replace(44, 'Aerial')
df['type_id'] = df['type_id'].replace(2, 'Offside Pass')
df['type_id'] = df['type_id'].replace(55, 'Offside Prokoved')
df['type_id'] = df['type_id'].replace(7, 'Tackle')
df['type_id'] = df['type_id'].replace(50, 'Dispossessed')
df['type_id'] = df['type_id'].replace(11, 'Claim')
df['type_id'] = df['type_id'].replace(17, 'Card')
df['type_id'] = df['type_id'].replace(18, 'Player Off')
df['type_id'] = df['type_id'].replace(19, 'Player On')
df['type_id'] = df['type_id'].replace(27, 'Start Delay')
df['type_id'] = df['type_id'].replace(28, 'End Delay')
df['type_id'] = df['type_id'].replace(3, 'Take On')
df['type_id'] = df['type_id'].replace(30, 'End Period')
df['type_id'] = df['type_id'].replace(32, 'Start Period')
df['type_id'] = df['type_id'].replace(34, 'Team Lineup')
df['type_id'] = df['type_id'].replace(37, 'Collection End')
df['type_id'] = df['type_id'].replace(40, 'Formation Change')
df['type_id'] = df['type_id'].replace(41, 'Punch')
df['type_id'] = df['type_id'].replace(45, 'Challenge')
df['type_id'] = df['type_id'].replace(52, 'Pick-up')
df['type_id'] = df['type_id'].replace(58, 'Penalty faced')
df['type_id'] = df['type_id'].replace(59, 'Sweeper')
df['type_id'] = df['type_id'].replace(65, 'Contentious Referee')

#STR 
#df['player_id'] = df['player_id'].astype(int).astype(str)

df["Event"] = ""
df.loc[(df["type_id"] == "Pass") & (df["outcome"] == True), "Event"] = "Successful Passes"
df.loc[(df["type_id"] == "Pass") & (df["outcome"] == False), "Event"] = "Unsuccessful Passes"
df.loc[(df["type_id"] == "Clearance"), "Event"] = "Clearance"
df.loc[(df["type_id"] == "Interception"), "Event"] = "Interception"
df.loc[(df["type_id"] == "Ball touch"), "Event"] = "Touch"
df.loc[(df["type_id"] == "Ball recovery"), "Event"] = "Recovery"



df['qualifiers2'] = df['qualifiers'].apply(reparar_y_extraer)
df['Handball'] = df['qualifiers2'].str.contains(r'\b10:null\b')
df['Overrun'] = df['qualifiers2'].str.contains(r'\b211:null\b')
df['AerialFoul'] = df['qualifiers2'].str.contains(r'\b264:null\b')
df['Throw-in'] = df['qualifiers2'].str.contains(r'\b107:null\b')

df['DuelosOfensivos'] = df['qualifiers2'].str.contains(r'\b286:null\b')
df['DuelosDefensivos'] = df['qualifiers2'].str.contains(r'\b285:null\b')

df['Cross'] = df['qualifiers2'].str.contains(r'\b2:null\b')
df['CornerTaken'] = df['qualifiers2'].str.contains(r'\b6:null\b')

# Paso 1: Duplicar la columna
df['qualifiers3'] = df['qualifiers2']

# Diccionario de reemplazos
replacements = {
    '1': 'LongBall',  # Este es delicado
    '2': 'Cross',
    '3': 'HeadPass',
    '5': 'FreeKickTaken',
    '6': 'CornerTaken',
    '56': 'Zone',
    '74': 'HitCrossbar',
    '123': 'KeeperThrow',
    '124': 'GoalKick',
    '140': 'PassEndX',
    '141': 'PassEndY',
    '154': 'IntentionalAssist',
    '155': 'Chipped',
    '157': 'Launch',
    '189': 'PlayerNotVisible',
    '199': 'KickHands',
    '210': 'Assist',
    '212': 'Length',
    '213': 'Angle',
    '223': 'In-Swinger',
    '224': 'Out-Swinger',
    '233': 'OppEventID',
    '237': 'LowGoalKick',
    '279': 'KickOff'
}

# Crear un patrón que detecte exactamente los códigos seguidos de ":"
pattern = r'\b(' + '|'.join(re.escape(k) for k in replacements.keys()) + r')(?=:)'
# Reemplazar con una sola pasada usando una función lambda
def reemplazar_codigos(texto):
    return re.sub(pattern, lambda m: replacements[m.group(0)], texto)

# Aplicar al DataFrame
df['qualifiers3'] = df['qualifiers3'].apply(reemplazar_codigos)

df['player_id'] = df['player_id'].astype('Int64').astype(str)
df['LastEvent'] = df['type_id'].shift(1)
df['NextEvent'] = df['type_id'].shift(-1)

df['NextPlayer'] = np.where(
    (df['type_id'].shift(-1) == 83) |
    (df['NextEvent'] == 'Formation Change') |
    (df['NextEvent'] == 'Start Delay'),
    df['player_name'].shift(-2),
    df['player_name'].shift(-1)
)

df_backup = df

################################################################################################################################################################################################################################################################################################################################################
#Filtros
filteropt01, filteropt02, filteropt03, filteropt04, filteropt05, filteropt06 = st.columns(6)

with filteropt01:
    MatchdayList = df['matchday'].drop_duplicates().tolist()
    MatchdayList.insert(0, "All Matchdays")  
    MatchdaySel = st.selectbox('Matchday', MatchdayList)
    dfbk_filteropt_01 = df
    if MatchdaySel == "All Matchdays":
        df = dfbk_filteropt_01
    else:
        df = df[df['matchday'] == MatchdaySel].reset_index(drop=True)
        
with filteropt02:
    MatchIDList = df['matchId'].drop_duplicates().tolist()
    MatchIDList.insert(0, "All Matches")  
    MatchIDSel = st.selectbox('MatchID', MatchIDList)
    dfbk_filteropt_02 = df
    if MatchIDSel == "All Matches":
        df = dfbk_filteropt_02
    else:
        df = df[df['matchId'] == MatchIDSel].reset_index(drop=True)

with filteropt03:
    TeamList = df['team_id'].drop_duplicates().tolist()
    TeamList.insert(0, "All Teams")  
    TeamSel = st.selectbox('Team', TeamList)
    dfbk_filteropt_03 = df
    if TeamSel == "All Teams":
        df = dfbk_filteropt_03
    else:
        df = df[df['team_id'] == TeamSel].reset_index(drop=True)
        
with filteropt04:
    PlayerList = df['player_id'].drop_duplicates().tolist()
    PlayerList.insert(0, "All Players")  
    PlayerSel = st.selectbox('Player', PlayerList)
    dfbk_filteropt_04 = df
    if PlayerSel == "All Players":
        df = dfbk_filteropt_04
    else:
        df = df[df['player_id'] == PlayerSel].reset_index(drop=True)

with filteropt05:
    EventList = df['type_id'].drop_duplicates().tolist()
    EventList.insert(0, "All Events")  
    EventSel = st.selectbox('Event', EventList)
    dfbk_filteropt_05 = df
    if EventSel == "All Events":
        df = dfbk_filteropt_05
    else:
        df = df[df['type_id'] == EventSel].reset_index(drop=True)

with filteropt06:
    OutcomeList = [True, False]
    OutcomeList.insert(0, "All")  
    OutcomeSel = st.selectbox('Outcome', OutcomeList)
    dfbk_filteropt_06 = df
    if OutcomeSel == "All":
        df = dfbk_filteropt_06
    else:
        df = df[df['outcome'] == OutcomeSel].reset_index(drop=True)

filteropt11, filteropt12, filteropt13, filteropt14, filteropt15, filteropt16 = st.columns(6)

with filteropt01:
    NextPlayerList = df['NextPlayer'].drop_duplicates().tolist()
    NextPlayerList.insert(0, "All NextPlayer")  
    NextPlayerSel = st.selectbox('NextPlayer', NextPlayerList)
    dfbk_filteropt_11 = df
    if NextPlayerSel == "All NextPlayer":
        df = dfbk_filteropt_11
    else:
        df = df[df['NextPlayer'] == NextPlayerSel].reset_index(drop=True)
df = df[['matchday', 'matchId', 'team_id', 'player_id', 'player_name', 'Event', 'type_id', 'outcome', 'LastEvent', 'NextEvent', 'NextPlayer', 'min', 'sec', 'x', 'y', 'x2', 'y2', 'qualifiers', 'qualifiers2', 'qualifiers3', 'Cross', 'CornerTaken', 'Throw-in', 'Handball', 'Overrun', 'AerialFoul', 'DuelosOfensivos', 'DuelosDefensivos']]

################################################################################################################################################################################################################################################################################################################################################

st.dataframe(df)

st.divider()
