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

################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

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
st.subheader('EXTRACT---DATA')


with st.form(key='form4'):
    uploaded_file = st.file_uploader("Choose a csv file", type="csv")
    submit_button2 = st.form_submit_button(label='Aceptar')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=';')
else:
    df = pd.read_csv('Data/undefined_undefined_events.csv', sep=';')

st.write(df)

df = df.iloc[6:].reset_index(drop=True)
df = df.drop(columns=['timestamp', 'date'])

df["x"] = df["x"] * 105 / 100
df["x2"] = df["x2"] * 105 / 100
df["y"] = df["y"] * 68 / 100
df["y2"] = df["y2"] * 68 / 100

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
df['qualifiers2'] = df['qualifiers'].apply(reparar_y_extraer)
df['ChanceCreated'] = df['qualifiers2'].str.contains(r'\b210:\b')
df['Assist'] = df['qualifiers2'].str.contains(r'\b210:16\b')
#df['KeyPass'] = df['qualifiers2'].str.contains(r'\b210:13\b')
df['KeyPass'] = df['qualifiers2'].str.contains(r'\b210:(13|14|15)\b')
df['Handball'] = df['qualifiers2'].str.contains(r'\b10:null\b')
df['Overrun'] = df['qualifiers2'].str.contains(r'\b211:null\b')
df['AerialFoul'] = df['qualifiers2'].str.contains(r'\b264:null\b')
df['Throw-in'] = df['qualifiers2'].str.contains(r'\b107:null\b')



# INTO FINAL THIRD - PASSES
df['FinalThirdPass'] = (df['x'] <= 70) & (df['x2'] >= 70)
# INTO OPPONNENT FIELD - PASSES
df['OpponnentFieldPass'] = (df['x'] <= 52.5) & (df['x2'] >= 52.5)
# INTO PENALTY AREA - PASSES
condicion1 = ((df['x'] < 88.5) | (df['y'] < 13.84) | (df['x'] > 105) | (df['y'] > 54.16))
condicion2 = ((df['x2'] >= 88.5) & (df['y2'] >= 13.84) & (df['x2'] <= 105) & (df['y2'] <= 54.16))
df['PenaltyAreaPass'] = condicion1 & condicion2
# PROGRESSIVE - PASSES
df['Beginning'] = np.sqrt(np.square(105 - df['x']) + np.square(34 - df['y']))
df['Ending'] = np.sqrt(np.square(105 - df['x2']) + np.square(34 - df['y2']))
df['ProgressPass'] = df['Ending'] / df['Beginning'] <= 0.75
df = df.drop(['Beginning', 'Ending'], axis=1)
# LONG - PASSES
df['Distance'] = np.sqrt((df['x2'] - df['x'])**2 + (df['y2'] - df['y'])**2)
df['LongPass'] = df['Distance'] >= 32
df = df.drop(['Distance'], axis=1)

df['DuelosOfensivos'] = df['qualifiers2'].str.contains(r'\b286:null\b')
df['DuelosDefensivos'] = df['qualifiers2'].str.contains(r'\b285:null\b')

df['Cross'] = df['qualifiers2'].str.contains(r'\b2:null\b')
df['CornerTaken'] = df['qualifiers2'].str.contains(r'\b6:null\b')
df['Blocked'] = df['qualifiers2'].str.contains(r'\b82:null\b')

df['DefBlock'] = df['qualifiers2'].str.contains(r'\b94:null\b')



df["Event"] = ""
df.loc[(df["type_id"] == "Pass") & (df["outcome"] == True), "Event"] = "Successful Passes"
df.loc[(df["type_id"] == "Pass") & (df["outcome"] == False), "Event"] = "Unsuccessful Passes"
df.loc[(df["type_id"] == "Offside Pass"), "Event"] = "Offside Pass"

df.loc[(df["type_id"] == "Shot On Target") & (df["Blocked"] == True), "Event"] = "Shot Blocked"
df.loc[(df["type_id"] == "Shot On Target") & (df["Blocked"] == False), "Event"] = "Shot On Target"


df.loc[(df["type_id"] == "Clearance"), "Event"] = "Clearance"
df.loc[(df["type_id"] == "Interception"), "Event"] = "Interception"
df.loc[(df["type_id"] == "Ball touch"), "Event"] = "Touch"
df.loc[(df["type_id"] == "Ball recovery"), "Event"] = "Recovery"
df.loc[(df["type_id"] == "Save"), "Event"] = "Save"
df.loc[(df["type_id"] == "Foul") & (df["outcome"] == False), "Event"] = "Foul Lost"
df.loc[(df["type_id"] == "Foul") & (df["outcome"] == True), "Event"] = "Foul Won"
df.loc[(df["type_id"] == "Goal"), "Event"] = "Goal"
#df.loc[(df["type_id"] == "Shot On Target"), "Event"] = "Shot On Target"
df.loc[(df["type_id"] == "Shot Off Target"), "Event"] = "Shot Off Target"
df.loc[(df["type_id"] == "Tackle") & (df["outcome"] == True), "Event"] = "Tackle Won"
df.loc[(df["type_id"] == "Tackle") & (df["outcome"] == False), "Event"] = "Tackle Lost"
df.loc[(df["type_id"] == "Aerial") & (df["outcome"] == True), "Event"] = "Aerial Won"
df.loc[(df["type_id"] == "Aerial") & (df["outcome"] == False), "Event"] = "Aerial Lost"
df.loc[(df["type_id"] == "Deleted Event"), "Event"] = "Deleted Event"






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
    #'210': 'Assist',
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
df = df[['matchday', 'matchId', 'team_id', 'player_id', 'player_name', 'Event', 'type_id', 'outcome', 'LastEvent', 'NextEvent', 'NextPlayer', 'min', 'sec', 'x', 'y', 'x2', 'y2', 'qualifiers', 'qualifiers2', 'qualifiers3', 'Cross', 'CornerTaken', 'Throw-in', 'ChanceCreated', 'Assist', 'KeyPass', 'Handball', 'Overrun', 'AerialFoul', 'DuelosOfensivos', 'DuelosDefensivos', 'Blocked', 'DefBlock', 'FinalThirdPass']]

################################################################################################################################################################################################################################################################################################################################################

st.dataframe(df)
st.divider()

st.subheader('DUELS')

duelo_eventos = ['Challenge', 'Foul', 'Aerial', 'Take On', 'Tackle', 'Dispossessed']
def contar_eventos(grupo):
    # Filtrar duelos
    duelos = grupo[grupo['type_id'].isin(duelo_eventos)]
    #duelos = duelos[~((duelos['type_id'] == 'Foul') & (duelos['Handball'] == True))]
    duelos = duelos[~(((duelos['type_id'] == 'Foul') & (duelos['Handball'] == True)) |
    ((duelos['type_id'] == 'Take On') & (duelos['Overrun'] == True)))]
    # Contar duelos y aerials
    total_duelos = len(duelos)
    total_aerials = ((grupo['type_id'] == 'Aerial') | ((grupo['type_id'] == 'Foul') & (grupo['AerialFoul'] == True))).sum()
    duelos_aereos_w = grupo[((grupo['type_id'] == 'Aerial') | ((grupo['type_id'] == 'Foul') & (grupo['AerialFoul'] == True))) & (grupo['outcome'] == True)].shape[0]

    duelos_w = duelos[
        ((duelos['type_id'] == 'Foul') & (duelos['outcome'] == True)) |
        (duelos['type_id'] == 'Aerial') & (duelos['outcome'] == True) |
        (duelos['type_id'] == 'Take On') & (duelos['outcome'] == True) |
        (duelos['type_id'] == 'Tackle')
    ].shape[0]

    duelos_w_df = duelos[
        ((duelos['type_id'] == 'Foul') & (duelos['outcome'] == True)) |
        ((duelos['type_id'] == 'Aerial') & (duelos['outcome'] == True)) |
        ((duelos['type_id'] == 'Take On') & (duelos['outcome'] == True)) |
        (duelos['type_id'] == 'Tackle')]
    duelos_f = duelos[
        ((duelos['type_id'] == 'Foul') & (duelos['outcome'] == False)) |
        (duelos['type_id'] == 'Aerial') & (duelos['outcome'] == False) |
        (duelos['type_id'] == 'Take On') & (duelos['outcome'] == False) |
        (duelos['type_id'] == 'Dispossessed') | (duelos['type_id'] == 'Challenge')
    ].shape[0]
    duelos_f_df = duelos[
        ((duelos['type_id'] == 'Foul') & (duelos['outcome'] == False)) |
        (duelos['type_id'] == 'Aerial') & (duelos['outcome'] == False) |
        (duelos['type_id'] == 'Take On') & (duelos['outcome'] == False) |
        (duelos['type_id'] == 'Dispossessed') | (duelos['type_id'] == 'Challenge')]
    total_duelos_ofensivos = duelos[duelos['DuelosOfensivos'] == True].shape[0]
    total_duelos_defensivos = duelos[duelos['DuelosDefensivos'] == True].shape[0]
    duelos_of_w = duelos_w_df[duelos_w_df['DuelosOfensivos'] == True].shape[0]
    duelos_of_f = duelos_f_df[duelos_f_df['DuelosOfensivos'] == True].shape[0]
    duelos_def_w = duelos_w_df[duelos_w_df['DuelosDefensivos'] == True].shape[0]
    duelos_def_f = duelos_f_df[duelos_f_df['DuelosDefensivos'] == True].shape[0]
    duelo_ground = ['Challenge', 'Foul', 'Take On', 'Tackle', 'Dispossessed']
    # Filtrar duelos de tipo ground
    duelos_ground_df = duelos[(duelos['type_id'].isin(duelo_ground)) & ~((duelos['type_id'] == 'Foul') & (duelos['AerialFoul'] == True))]
    # Contar total de duelos ground
    total_duelos_ground = duelos_ground_df.shape[0]
    # DuelosGround W
    duelos_ground_w = duelos_ground_df[
        ((duelos_ground_df['type_id'] == 'Foul') & (duelos_ground_df['outcome'] == True)) |
        ((duelos_ground_df['type_id'] == 'Take On') & (duelos_ground_df['outcome'] == True)) |
        (duelos_ground_df['type_id'] == 'Tackle')
    ].shape[0]
    
    # DuelosGround F
    duelos_ground_f = duelos_ground_df[
        ((duelos_ground_df['type_id'] == 'Foul') & (duelos_ground_df['outcome'] == False)) |
        ((duelos_ground_df['type_id'] == 'Take On') & (duelos_ground_df['outcome'] == False)) |
        (duelos_ground_df['type_id'] == 'Dispossessed') | (duelos['type_id'] == 'Challenge')
    ].shape[0]

    
    return pd.Series({'Duelos Totales': total_duelos, 'Duelos W': duelos_w, 'DuelosAéreos': total_aerials, 'DuelosAéreos W': duelos_aereos_w, 'DuelosGround': total_duelos_ground, 'DuelosGroundW': duelos_ground_w, 'DuelosGroundF': duelos_ground_f, 'DuelosOfensivos': total_duelos_ofensivos, 'DuelosOfensivos W': duelos_of_w, 'DuelosOfensivos F': duelos_of_f, 'DuelosDefensivos': total_duelos_defensivos, 'DuelosDefensivos W': duelos_def_w, 'DuelosDefensivos F': duelos_def_f})

df = df_backup
df_resultado = df.groupby(['matchId', 'player_id', 'player_name', 'team_id']).apply(contar_eventos).reset_index()

#df_resultado['Duelos %'] = round(((df_resultado['Duelos W'] / df_resultado['Duelos Totales']).fillna(0) * 100), 1)
df_resultado.insert(6, 'Duelos %', round(((df_resultado['Duelos W'] / df_resultado['Duelos Totales']).fillna(0) * 100), 1))

st.dataframe(df_resultado)
st.divider()

################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

st.subheader('PASSES')
# --- BLOQUE 1: Cálculo de estadísticas de pase ---
df = df_backup.copy()

pases = df[df['type_id'] == 'Pass'].copy()
pases_validos = pases[(pases['Cross'] == False) & (pases['Throw-in'] == False)]

Conteo_Agrupado = pases_validos.groupby(['matchId', 'player_id', 'player_name', 'team_id']).size().reset_index(name='TotalOPPases')

pases_exitosos = pases_validos[pases_validos['outcome'] == True]

Conteo_Exitosos = pases_exitosos.groupby(['matchId', 'player_id', 'player_name', 'team_id']).size().reset_index(name='PasesExitosos')
Conteo_Cross = pases[pases['Cross'] == True].groupby(['matchId', 'player_id', 'player_name', 'team_id'])['Cross'].count().reset_index(name='TotalCrosses')
Conteo_Throwin = pases[pases['Throw-in'] == True].groupby(['matchId', 'player_id', 'player_name', 'team_id'])['Throw-in'].count().reset_index(name='Throw-in')
Conteo_Assist = pases[pases['Assist'] == True].groupby(['matchId', 'player_id', 'player_name', 'team_id'])['Assist'].count().reset_index(name='TotalAssist')
Conteo_KeyPass = pases[pases['KeyPass'] == True].groupby(['matchId', 'player_id', 'player_name', 'team_id'])['KeyPass'].count().reset_index(name='TotalKeyPass')

# Unir todos los conteos en un solo DataFrame
Resultado = Conteo_Agrupado.merge(Conteo_Exitosos, on=['matchId', 'player_id', 'player_name', 'team_id'], how='left')
Resultado = Resultado.merge(Conteo_Cross, on=['matchId', 'player_id', 'player_name', 'team_id'], how='left')
Resultado = Resultado.merge(Conteo_Throwin, on=['matchId', 'player_id', 'player_name', 'team_id'], how='left')
Resultado = Resultado.merge(Conteo_Assist, on=['matchId', 'player_id', 'player_name', 'team_id'], how='left')
Resultado = Resultado.merge(Conteo_KeyPass, on=['matchId', 'player_id', 'player_name', 'team_id'], how='left')

# Completar NaNs
Resultado[['TotalCrosses', 'Throw-in', 'PasesExitosos', 'TotalAssist', 'TotalKeyPass']] = Resultado[['TotalCrosses', 'Throw-in', 'PasesExitosos', 'TotalAssist', 'TotalKeyPass']].fillna(0).astype(int)

# --- BLOQUE 2: Cálculo de pases recibidos excluyendo Throw-in ---
df = df_backup.copy()
df['Throw-in'] = df['Throw-in'].fillna(False)

pases_exitosos = df[(df['Event'] == 'Successful Passes') & (df['Throw-in'] == False)].copy()
pases_recibidos = pases_exitosos.groupby(['matchId', 'NextPlayer']).size().reset_index(name='PasesRecibidos')

jugadores = df[['matchId', 'player_name', 'player_id', 'team_id']].drop_duplicates()

resultado_recibidos = jugadores.merge(pases_recibidos, left_on=['matchId', 'player_name'], right_on=['matchId', 'NextPlayer'], how='left')

resultado_recibidos['PasesRecibidos'] = resultado_recibidos['PasesRecibidos'].fillna(0).astype(int)
resultado_recibidos = resultado_recibidos[['matchId', 'player_id', 'player_name', 'team_id', 'PasesRecibidos']]

# --- BLOQUE 3: Unión final de todo ---
resultado_total = Resultado.merge(resultado_recibidos, on=['matchId', 'player_id', 'player_name', 'team_id'], how='left')

resultado_total['PasesRecibidos'] = resultado_total['PasesRecibidos'].fillna(0).astype(int)

st.write(resultado_total)
st.divider()

df = df_backup
#st.dataframe(df)

################################################################################################################################################################################################################################################################################################################################################

st.subheader('SHOTS')
shots_eventos = ['Shot On Target', 'Shot Off Target', 'Goal']
shots_df = df[df['type_id'].isin(shots_eventos)]
Shots_Conteo = shots_df.groupby(['matchId', 'matchday', 'player_id', 'player_name', 'team_id']).size().reset_index(name='TotalShots')

Goal_df = shots_df[(shots_df['Event'] == 'Goal')].reset_index(drop=True)
Goal_Conteo = Goal_df.groupby(['matchId', 'matchday', 'player_id', 'player_name', 'team_id']).size().reset_index(name='Goals')

SoT_df = shots_df[(shots_df['Event'] == 'Shot On Target')].reset_index(drop=True)
SoT_Conteo = SoT_df.groupby(['matchId', 'matchday', 'player_id', 'player_name', 'team_id']).size().reset_index(name='Total SoT')

SoffT_df = shots_df[(shots_df['Event'] == 'Shot Off Target')].reset_index(drop=True)
SoffT_Conteo = SoffT_df.groupby(['matchId', 'matchday', 'player_id', 'player_name', 'team_id']).size().reset_index(name='Total SoffT')

SB_df = shots_df[(shots_df['Event'] == 'Shot Blocked')].reset_index(drop=True)
SB_Conteo = SB_df.groupby(['matchId', 'matchday', 'player_id', 'player_name', 'team_id']).size().reset_index(name='Total SoB')


Resultado_Shots = Shots_Conteo.merge(SoT_Conteo, on=['matchId', 'matchday', 'player_id', 'player_name', 'team_id'], how='left')
Resultado_Shots = Resultado_Shots.merge(Goal_Conteo, on=['matchId', 'matchday', 'player_id', 'player_name', 'team_id'], how='left')
Resultado_Shots = Resultado_Shots.merge(SoffT_Conteo, on=['matchId', 'matchday', 'player_id', 'player_name', 'team_id'], how='left')
Resultado_Shots = Resultado_Shots.merge(SB_Conteo, on=['matchId', 'matchday', 'player_id', 'player_name', 'team_id'], how='left')

Resultado_Shots['Total SoT'] = Resultado_Shots['Total SoT'].fillna(0).astype(int)
Resultado_Shots['Goals'] = Resultado_Shots['Goals'].fillna(0).astype(int)
Resultado_Shots['Total SoffT'] = Resultado_Shots['Total SoffT'].fillna(0).astype(int)
Resultado_Shots['Total SoB'] = Resultado_Shots['Total SoB'].fillna(0).astype(int)

st.write(Resultado_Shots)

st.divider()
################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

st.subheader("DEFENSIVE")
df = df_backup.copy()

def_eventos = ['Clearance', 'Interception', 'Tackle', 'Ball recovery', 'Save']
defWS_df = df[df['type_id'].isin(def_eventos)]
def_Conteo = defWS_df.groupby(['matchId', 'matchday', 'player_id', 'player_name', 'team_id']).size().reset_index(name='TotalDefActions')

Clearance_df = defWS_df[(defWS_df['Event'] == 'Clearance')].reset_index(drop=True)
Clearance_Conteo = Clearance_df.groupby(['matchId', 'matchday', 'player_id', 'player_name', 'team_id']).size().reset_index(name='TotalClearances')

Interception_df = defWS_df[(defWS_df['Event'] == 'Interception')].reset_index(drop=True)
Interception_Conteo = Interception_df.groupby(['matchId', 'matchday', 'player_id', 'player_name', 'team_id']).size().reset_index(name='TotalInterceptions')

Recoveries_df = defWS_df[(defWS_df['type_id'] == 'Ball recovery')].reset_index(drop=True)
Recoveries_Conteo = Recoveries_df.groupby(['matchId', 'matchday', 'player_id', 'player_name', 'team_id']).size().reset_index(name='TotalRecoveries')

DefBlocks_df = defWS_df[(defWS_df['type_id'] == 'Save') & (defWS_df['DefBlock'] == True)].reset_index(drop=True)
#DefBlocks_df = defWS_df[(defWS_df['DefBlock'] == True)].reset_index(drop=True)
DefBlocks_Conteo = DefBlocks_df.groupby(['matchId', 'matchday', 'player_id', 'player_name', 'team_id']).size().reset_index(name='TotalDefBlocks')

Tackles_df = defWS_df[(defWS_df['type_id'] == 'Tackle')].reset_index(drop=True)
Tackles_Conteo = Tackles_df.groupby(['matchId', 'matchday', 'player_id', 'player_name', 'team_id']).size().reset_index(name='TotalTackles')

TackleWon_df = defWS_df[(defWS_df['Event'] == 'Tackle Won')].reset_index(drop=True)
TackleWon_Conteo = TackleWon_df.groupby(['matchId', 'matchday', 'player_id', 'player_name', 'team_id']).size().reset_index(name='TacklesWon')

TackleLost_df = defWS_df[(defWS_df['Event'] == 'Tackle Lost')].reset_index(drop=True)
TackleLost_Conteo = TackleLost_df.groupby(['matchId', 'matchday', 'player_id', 'player_name', 'team_id']).size().reset_index(name='TacklesLost')

Resultado_Def = def_Conteo.merge(Clearance_Conteo, on=['matchId', 'matchday', 'player_id', 'player_name', 'team_id'], how='left')
Resultado_Def = Resultado_Def.merge(Interception_Conteo, on=['matchId', 'matchday', 'player_id', 'player_name', 'team_id'], how='left')
Resultado_Def = Resultado_Def.merge(Recoveries_Conteo, on=['matchId', 'matchday', 'player_id', 'player_name', 'team_id'], how='left')
Resultado_Def = Resultado_Def.merge(DefBlocks_Conteo, on=['matchId', 'matchday', 'player_id', 'player_name', 'team_id'], how='left')
Resultado_Def = Resultado_Def.merge(Tackles_Conteo, on=['matchId', 'matchday', 'player_id', 'player_name', 'team_id'], how='left')
Resultado_Def = Resultado_Def.merge(TackleWon_Conteo, on=['matchId', 'matchday', 'player_id', 'player_name', 'team_id'], how='left')
Resultado_Def = Resultado_Def.merge(TackleLost_Conteo, on=['matchId', 'matchday', 'player_id', 'player_name', 'team_id'], how='left')


Resultado_Def['TotalDefActions'] = Resultado_Def['TotalDefActions'].fillna(0).astype(int)
Resultado_Def['TotalClearances'] = Resultado_Def['TotalClearances'].fillna(0).astype(int)
Resultado_Def['TotalInterceptions'] = Resultado_Def['TotalInterceptions'].fillna(0).astype(int)
Resultado_Def['TotalRecoveries'] = Resultado_Def['TotalRecoveries'].fillna(0).astype(int)
Resultado_Def['TotalDefBlocks'] = Resultado_Def['TotalDefBlocks'].fillna(0).astype(int)
Resultado_Def['TotalTackles'] = Resultado_Def['TotalTackles'].fillna(0).astype(int)
Resultado_Def['TacklesWon'] = Resultado_Def['TacklesWon'].fillna(0).astype(int)
Resultado_Def['TacklesLost'] = Resultado_Def['TacklesLost'].fillna(0).astype(int)

st.write(Resultado_Def)

st.divider()

################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

st.subheader("DRIBBLES")

df = df_backup.copy()

Dribbles_eventos = ['Take On']
Dribbles_df = df[df['type_id'].isin(Dribbles_eventos)]
Dribbles_Conteo = Dribbles_df.groupby(['matchId', 'matchday', 'player_id', 'player_name', 'team_id']).size().reset_index(name='TotalDribbles')

DribblesW_df = Dribbles_df[(Dribbles_df['outcome'] == True)].reset_index(drop=True)
DribblesW_Conteo = DribblesW_df.groupby(['matchId', 'matchday', 'player_id', 'player_name', 'team_id']).size().reset_index(name='DribblesW')

DribblesF_df = Dribbles_df[(Dribbles_df['outcome'] == False)].reset_index(drop=True)
DribblesF_Conteo = DribblesF_df.groupby(['matchId', 'matchday', 'player_id', 'player_name', 'team_id']).size().reset_index(name='DribblesF')

Resultado_Dribbles = Dribbles_Conteo.merge(DribblesW_Conteo, on=['matchId', 'matchday', 'player_id', 'player_name', 'team_id'], how='left')
Resultado_Dribbles = Resultado_Dribbles.merge(DribblesF_Conteo, on=['matchId', 'matchday', 'player_id', 'player_name', 'team_id'], how='left')

Resultado_Dribbles['TotalDribbles'] = Resultado_Dribbles['TotalDribbles'].fillna(0).astype(int)
Resultado_Dribbles['DribblesW'] = Resultado_Dribbles['DribblesW'].fillna(0).astype(int)
Resultado_Dribbles['DribblesF'] = Resultado_Dribbles['DribblesF'].fillna(0).astype(int)

st.write(Resultado_Dribbles)

st.divider()

################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

st.subheader("FOULS")

df = df_backup.copy()

Fouls_eventos = ['Foul Won', 'Foul Lost']
Fouls_df = df[df['Event'].isin(Fouls_eventos)]
Fouls_Conteo = Fouls_df.groupby(['matchId', 'matchday', 'player_id', 'player_name', 'team_id']).size().reset_index(name='TotalFouls')

FoulsW_df = Fouls_df[(Fouls_df['Event'] == 'Foul Won')].reset_index(drop=True)
FoulsW_Conteo = FoulsW_df.groupby(['matchId', 'matchday', 'player_id', 'player_name', 'team_id']).size().reset_index(name='FoulsW')

FoulsF_df = Fouls_df[(Fouls_df['Event'] == 'Foul Lost')].reset_index(drop=True)
FoulsF_Conteo = FoulsF_df.groupby(['matchId', 'matchday', 'player_id', 'player_name', 'team_id']).size().reset_index(name='FoulsF')

Resultado_Fouls = Fouls_Conteo.merge(FoulsW_Conteo, on=['matchId', 'matchday', 'player_id', 'player_name', 'team_id'], how='left')
Resultado_Fouls = Resultado_Fouls.merge(FoulsF_Conteo, on=['matchId', 'matchday', 'player_id', 'player_name', 'team_id'], how='left')

Resultado_Fouls['FoulsW'] = Resultado_Fouls['FoulsW'].fillna(0).astype(int)
Resultado_Fouls['FoulsF'] = Resultado_Fouls['FoulsF'].fillna(0).astype(int)

st.write(Resultado_Fouls)

st.divider()

################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

st.subheader("VIZ")

#df = df.drop(["Out"], axis=1)
df = df[~df['type_id'].isin(['Out'])]


fig, ax = mplt.subplots(figsize=(8, 8), dpi = 800)
ax.axis("off")
fig.patch.set_visible(False)

pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1.0, goal_type='box', pitch_length=105, pitch_width=68)
#Adding directon arrow
ax29 = fig.add_axes([0.368,0.22,0.3,0.05])
ax29.axis("off")
ax29.set_xlim(0,10)
ax29.set_ylim(0,10)
ax29.annotate('', xy=(2, 6), xytext=(8, 6), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
#ax29.annotate(s='', xy=(2, 5), xytext=(8, 5), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
ax29.text(5, 2, 'Dirección campo de juego',  c=(1,1,1,0.5), fontsize=10, ha='center')
pitch.draw(ax=ax)

ax.scatter(df['x'], df['y'], color = "#FF0046", edgecolors='w', s=30, zorder=2, alpha=0.2)

st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")

st.write(df)
st.divider()
