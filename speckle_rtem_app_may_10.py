# from asyncio.constants import SENDFILE_FALLBACK_READBUFFER_SIZE
# from enum import unique
from onboard.client import RtemClient
from onboard.client.models import PointSelector
from datetime import datetime, timezone, timedelta
import pytz
from onboard.client.models import TimeseriesQuery, PointData
from onboard.client.dataframes import points_df_from_streaming_timeseries
from specklepy.api.credentials import get_default_account
from specklepy.api.client import SpeckleClient
from specklepy.api.credentials import get_account_from_token
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn import preprocessing as prep
# from logging.handlers import TimedRotatingFileHandler
# from email import message_from_string
import numpy as np
import pandas as pd
import base64
import requests
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels as sm
import plotly.express as px
import seaborn as sns
cmap = sns.color_palette("Spectral")
# %matplotlib inline
plt.rcParams.update({'font.size': 18})
plt.style.use('ggplot')

# initialise the RTEM client
client = RtemClient(
    api_key='ob-p-PftYB3t-_72rkNxUHJar-6QRSskt7YnG9jBTQrs7n8dXlbQn7U23yKcmYjYUViGVuGc')
query = PointSelector()

# initialise the Speckle client
client_speckle = SpeckleClient(host="https://speckle.xyz/streams/c463c6f6bd")

# ----------------------------
# PAGE CONFIG
st.set_page_config(
    page_title='RTEM New York Buildings',
    page_icon="🔌 "
)
# ----------------------------

# ----------------------------
# CONTAINERS
header = st.container()
input = st.container()
viewer = st.container()
rtem = st.container()
heatmap = st.container()
plots = st.container()
sensors = st.container()
report = st.container()
maps = st.container()
merged = st.container()
# ----------------------------

# ----------------------------
# HEADER
with header:
    st.title('New York Buildings 🏢 🔌  ')

    st.markdown('''
    This app shows 200+ New York building real-time sensor data.
    * **Python libraries:** base64, pandas, numpy, streamlit, seaborn, matplotlib.pyplot, specklepy.api.client, specklepy.api.credientials, onboard.client
    * **Data source:** [NY-RTEM](https://ny-rtem.com)
''')
# ----------------------------

# ----------------------------
# INPUTS
with input:
    # st.subheader("Inputs")

    # --------
    # Columns for inputs
    serverCol, tokenCol = st.columns([1, 2])
    # user input boxes
    speckleServer = "speckle.xyz"
    speckleToken = "54cc9934fda25c598900ab2a6a9228c907c2724d87"
    # --------

    # --------
    client_speckle = SpeckleClient(host=speckleServer)
    # get account from tocken
    account = get_account_from_token(speckleToken, speckleServer)
    # authenticate
    client_speckle.authenticate_with_account(account)
    # --------

    # --------
    # streams list
    streams = client_speckle.stream.list()
    # get stream names
    streamNames = [s.name for s in streams]
    # dropdown for stream selection
    sName = st.selectbox(label='select your stream', options=streamNames)
    # selected stream ✅
    stream = client_speckle.stream.search(sName)[0]
    # stream branches 🌴
    branches = client_speckle.branch.list(stream.id)
    # stream commits 🏹
    commits = client_speckle.commit.list(stream.id, limit=100)
    # --------

# ----------------------------

# ----------------------------
# FUNCTIONS


def listToMarkdown(list, column):
    list = ["- " + i + "\n" for i in list]
    list = "".join(list)
    return column.markdown(list)


def commit2viewer(stream, commit):
    embed_src = "https://speckle.xyz/embed?stream"+stream.id+"&commit="+commit.id
    return st.components.v1.iframe(src=embed_src, height=400)
# ----------------------------


# ----------------------------
# SPECKLE VIEWER 👓
with viewer:
    st.subheader("Carleton University 👇")

    # st.components.v1.iframe(
    #     src="https://speckle.xyz/embed?stream=c463c6f6bd&commit=4204c3b944", height=400)

    # st.write(commits)
    commit2viewer(stream, commits[4])
# ----------------------------

# ----------------------------
# RTEM
with rtem:
    building_col = st.columns(3)

    # API request
    key = {'key': 'ob-p-PftYB3t-_72rkNxUHJar-6QRSskt7YnG9jBTQrs7n8dXlbQn7U23yKcmYjYUViGVuGc'}
    response = requests.post(
        url='https://api.onboarddata.io/login/api-key', data=key)
    response = response.json()

    # response['access_token']
    headers = {'Authorization': 'Bearer ' + response['access_token']}
    bdgs = requests.get(
        url='https://api.onboarddata.io/buildings', headers=headers).json()
    bd = pd.json_normalize(bdgs)

    # drop cols that are not meaningful
    drop_cols = ['org_id', 'info.geoState', 'timezone', 'status', 'image_src', 'bms_manufacturer',
                 'bms_product_name', 'bms_version', 'address', 'info.geoCountry', 'info.weatherRef']
    bd.drop(columns=drop_cols, inplace=True)

    # identify appropriate datatype for each cols
    time_cols = ['info.sunstart', 'info.satstart',
                 'info.m2fstart', 'info.sunend', 'info.satend', 'info.m2fend']
    int_cols = ['info.yearBuilt', 'info.floors', 'name']

    # examine to check time format
    for col in time_cols:
        print(f'{col}: \n', bd[col].value_counts())

    # Unify time format
    for col in time_cols:
        bd[col].replace('900', '09:00', inplace=True)
        bd[col].replace('1700', '17:00', inplace=True)
    #     bd[col] = bd[col].apply(pd.Timestamp)
        bd[col] = pd.to_datetime(bd[col])
        bd[col] = bd[col].dt.time
        print()
        print(f'{col}: \n')
        print('-'*20)
        print(bd[col].value_counts())

    # we have 215 missing values for the number of floors (info.floors)
    # first, convert non-numeric values to numeric
    bd['info.floors'] = pd.to_numeric(bd['info.floors'], errors='coerce')

    # then, fill in missing values with 0
    bd['info.floors'] = bd['info.floors'].fillna(0)

    # check for dtype. The number of floors should show as integer.
    bd['info.floors'] = bd['info.floors'].astype('int64')
    bd['info.floors'].value_counts()

    # covnert to numeric
    # fill missing values with 0

    bd['info.yearBuilt'] = pd.to_numeric(bd['info.yearBuilt'], errors='coerce')
    bd['info.yearBuilt'] = bd['info.yearBuilt'].fillna(0)

    # update datatype to integer
    bd['info.yearBuilt'] = bd['info.yearBuilt'].astype('int64')
    bd['name'] = bd['name'].astype('int64')
    bd.dtypes.sort_values()

    bd['sq_ft'].fillna(0, inplace=True)

    bd['info.customerType'].fillna('na', inplace=True)
    bd['info.customerType'].replace('', 'na', inplace=True)

    # update city name from 'purchase' to 'new york'
    # bd['info.geoCity'] = np.where(
    #     bd['info.org'] == 'Simone Property Dev', 'new york', bd['info.geoCity'])
    # bd[bd['info.org'] == 'Simone Property Dev']

    # update column datatypes
    cat_cols = bd.select_dtypes('object').columns.to_list()
    num_cols = bd.select_dtypes(['int64', 'float64']).columns.to_list()
    int_cols = bd.select_dtypes('int64').columns.to_list()
    float_cols = bd.select_dtypes('float64').columns.to_list()

    # update info.org missing values with 'na'
    # bd['info.org'] = bd['info.org'].str.strip()
    # bd['info.org'].fillna('na', inplace=True)
    # bd['info.org'].replace('', 'na', inplace=True)

    # fill in missing values, replace duplicate city names
    # change to lower case to prevent double counting due to capitalization
    bd['info.geoCity'].replace('NYC', 'New York', inplace=True)
    bd['info.geoCity'].fillna('na', inplace=True)
    bd['info.geoCity'] = bd['info.geoCity'].apply(str.lower)
    bd['info.geoCity'].value_counts()

    cols = bd.columns.to_list()
    cols = cols[:2] + cols[2:3] + cols[-1:] + cols[-2:-1] + cols[3:-2]
    cols = cols[:4] + cols[-4:-3] + cols[4:-4] + cols[-3:]
    cols = cols[:3] + cols[5:6] + cols[3:5] + cols[6:]
    cols = cols[:5] + cols[8:10] + cols[5:8] + cols[10:]
    cols = cols[:-6] + cols[-3:] + cols[-6:-3]
    bd = bd[cols]

    bd_fig = px.scatter(bd, x='id', y='sq_ft')
    bd_fig.update_layout(
        margin=dict(l=2, r=2, t=2, b=2)
    )
    # bd

    # --------------
    # SIDEBAR

    # sidebar - Year selection
    st.sidebar.header('User Input Features')
    selected_year = st.sidebar.selectbox('Year', list(range(2019, 2021)))

    # sidebar - Building Type selection
    sorted_unique_bdtype = sorted(bd['info.customerType'].unique())
    selected_unique_bdtype = st.sidebar.multiselect(
        'Building Type', sorted_unique_bdtype, sorted_unique_bdtype)

    # sidebar - City selection
    unique_city = sorted(bd['info.geoCity'].unique())
    selected_city = st.sidebar.multiselect('City', unique_city, unique_city)

    # filtering data
    df_selected_bd = bd[(bd['info.customerType'].isin(
        selected_unique_bdtype)) & (bd['info.geoCity'].isin(selected_city))]

    st.subheader('Summary of Selected buildings(s) 🙌')
    # + str(
    st.write('Data Dimension: ' + str(df_selected_bd.shape[0]) + ' rows')
    # df_selected_bd.shape[1]) + ' columns')
    st.dataframe(
        df_selected_bd[['id', 'name', 'sq_ft', 'equip_count', 'point_count', 'info.geoCity', 'info.customerType']].sort_values('point_count', ascending=False))

    # --------------
    # EXPORT TO .CSV
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="selected_buildings.csv">Download CSV File</a>'
        return href
    st.markdown(filedownload(df_selected_bd), unsafe_allow_html=True)
    # --------------

# ----------------------------
# HEATMAP
with heatmap:
    if st.button('Intercorrelation Heatmap (Buildings) 🔥'):
        st.subheader('Intercorrelation Matrix Heatmap')
        df_selected_bd.to_csv('output.csv', index=False)
        df = pd.read_csv('output.csv')

        corr = df.corr()
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(7, 5))
            ax = sns.heatmap(corr, mask=mask, vmax=1,
                             square=True, annot=True, fmt='.1f')
        st.pyplot(f)
# ----------------------------


# ------------------------------------------
# PLOTS - buildings
with plots:
    # PLOTS - ALL NY BUILDINGS
    st.subheader('All New York Buliidngs 🎉')
    st.markdown("""
    `K-12 Schools` have smaller floor area than Commercial Office buildings. However, it has the most amount of equipments in average and quite large amount of points as well.
    `Office` buildings have the larges floor area in average. However, it has relatively small amount of equipments.
    Public service buildings (`K-12 School, Not for Profit, Hospitality`) tend to have more equipements & points compared to commercial / residential buildngs). This intuitively makes sense as it may be reuiqred by policiesd or local authorities.
    Buildings with large amount of points are possibly `smart buildings` with many IoT devices. Let's keep this in mind and investigate further later.
    """)

    # plot floor area vs building type
    bd = bd.sort_values('sq_ft', ascending=False)
    fig_floor_area = plt.figure(figsize=(10, 4))
    data = bd[bd['info.customerType'].notna() == True].sort_values('sq_ft')
    sns.scatterplot(data=data, x='info.customerType', y='sq_ft')
    plt.xticks(rotation=90)
    plt.title('Total Floor Area vs Building Type')
    # plt.xlabel('Building Type')
    plt.ylabel('Total Floor Area [SQ_FT]')
    plt.grid(True)
    plt.show()
    st.pyplot(fig_floor_area)

    # plot equipment counts
    fig_equipment_counts = plt.figure(figsize=(10, 4))
    sns.boxplot(data=bd, x='info.customerType', y='equip_count')
    plt.xticks(rotation=90)
    plt.title('Equip_counts vs Building Type')
    plt.xlabel('Building Type')
    plt.ylabel('Counts')
    plt.grid(True)
    plt.show()
    st.pyplot(fig_equipment_counts)

    # plot point counts
    fig_point_counts = plt.figure(figsize=(10, 4))
    sns.boxplot(data=bd, x='info.customerType', y='point_count')
    plt.xticks(rotation=90)
    plt.title('Point_counts vs Building Type')
    plt.xlabel('Building Type')
    plt.ylabel('Counts')
    plt.grid(True)
    plt.show()
    st.pyplot(fig_point_counts)
    # --------------

# ------------------------------------------


# ------------------------------------------
# ENCODING
discretize_names = ['info.customerType', 'info.geoCity']
enc = prep.OrdinalEncoder(dtype=np.int16)
discretized_features = bd[discretize_names]
# print(discretized_features)

# fit a discretizer on our data.
# This will assign a different distinct value
# for each distinct class in each column
# (0,..., n-1 for the n classes in each column)
# now use this discretizer to transform our dataset
enc.fit(discretized_features)
discretized_features = enc.transform(discretized_features)
bd_encoded = bd.copy()
bd_encoded[discretize_names] = discretized_features
# bd_encoded

# correlation between features
fig_correlation = plt.figure(figsize=(10, 4))
corr = bd_encoded.corr(method='spearman')
sns.heatmap(corr, annot=True)
plt.title('Correlations between Features (Ordinal Encoded)')
plt.show()
# st.pyplot(fig_correlation)
# ------------------------------------------

# ------------------------------------------
# POINTS
st.subheader('All New York Sensors 🚀')

# --------------
# import points data
points = pd.json_normalize(client.get_all_points())
missing = pd.DataFrame(points.isnull().sum() * 100 / len(points)).sort_values(
    0, ascending=False).rename(columns={0: 'percent_missing'})
drop_cols = missing[missing['percent_missing'] > 90].index.tolist()
points = points.drop(columns=drop_cols, axis=1)
keep_cols = ['id', 'equip_id', 'building_id', 'first_updated', 'last_updated', 'type',
             'point_type_id', 'description', 'value', 'units', 'tagged_units', 'measurement_id']
points = points[keep_cols]
points = points.rename(columns={'id': 'point_id', 'measurement_id': 'tagged_measurement_id',
                                'type': 'point_type', 'description': 'point_description'})
# --------------

# --------------
# import equipment data
equipments = pd.json_normalize(client.get_all_equipment())
missing = pd.DataFrame(equipments.isnull().sum() * 100 / len(equipments)
                       ).sort_values(0, ascending=False).rename(columns={0: 'percent_missing'})
drop_cols = missing[missing['percent_missing'] > 90].index.tolist()
equipments = equipments.drop(columns=drop_cols, axis=1)

keep_cols = ['id', 'equip_id', 'building_id', 'suffix', 'equip_type_name',
             'equip_type_id', 'equip_type_abbr', 'equip_type_tag', 'equip_dis', 'flow_order', ]
equipments = equipments[keep_cols]
equipments = equipments.rename(columns={'equip_id': 'equip_id_name', 'id': 'equip_id',
                                        'tags': 'equip_tags', 'suffix': 'equip_suffix', 'flow_order': 'equip_flow_order'})
bd = bd.rename(columns={'id': 'building_id', 'name': 'building_name'})
# --------------

# --------------
# merge dataframe (building + equipement +sensor)
merged = pd.merge(points, equipments, how='left', left_on=[
    'equip_id', 'building_id'], right_on=['equip_id', 'building_id'])
# merge with buildings
merged = pd.merge(merged, bd, how='left',
                  left_on='building_id', right_on='building_id')
merged
# --------------

# --------------
# EXPORT TO .CSV - Sensors
st.markdown(filedownload(merged), unsafe_allow_html=True)
# --------------

# ----------------------------
# BUTTONS

# --------------
# college/university buildings
# if st.button('College/University Available Sensors 🏛'):
st.subheader('College/University Building Sensors 🏫')
df = merged[merged['info.customerType'] == 'College/University']
x = 'building_id'
y = 'point_type'

# plot
fig_univ = plt.figure(figsize=(10, 20))
sns.histplot(data=df, x=x, y=y, legend=True)  # , hue='equip_type_tag'
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=4)

plt.show()
st.pyplot(fig_univ)

# --------------
# all sensors, equipements, buildings
# if st.button('All building Available Sensors 🏛'):
st.subheader('All Building Sensors 🏫')
df = merged
x = 'building_id'
y = 'point_type'

# plot
fig_merged = plt.figure(figsize=(10, 40))
sns.histplot(data=df, x=x, y=y, legend=True)  # , hue='equip_type_tag'
plt.show()
st.pyplot(fig_merged)
# ----------------------------

# sidebar - Sensor selection
selected_unique_sensors = sorted(merged['point_type'].unique())
selected_sensors = st.sidebar.multiselect(
    'Sensor', selected_unique_sensors, selected_unique_sensors)

# filtering data
df_selected_sensors = merged[merged['point_type'].isin(
    selected_unique_sensors)]

st.subheader('Summary of Selected buildings(s) 🙌')
st.write('Data Dimension: ' + str(df_selected_bd.shape[0]) + ' rows')
# df_selected_bd.shape[1]) + ' columns')
st.dataframe(df_selected_sensors)
# df_selected_bd[['id', 'name', 'sq_ft', 'equip_count', 'point_count', 'info.geoCity', 'info.customerType']].sort_values('point_count', ascending=False))


# ----------------------------
# MAPS
#
# US map Shape file(https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html)
# ----------------------------

# ----------------------------
# WEB SCRAPING
# @ st.cache(suppress_st_warning=True)
# def load_data(year):
#     url = 'https://www.pro-football-reference.com/years/' + \
#         str(selected_year) + '/rushing.htm'
#     # https://www.pro-football-reference.com/years/2021/rushing.htm
#     html = pd.read_html(url, header=1)
#     df = html[0]
#     df = df.drop(df[df['Age'] == 'Age'].index)
#     raw = df.drop(columns=['Att'])
#     playerstats = raw.drop(['Rk'], axis=1)
#     raw = str(raw.Pos.fillna(0))
#     return playerstats

# playerstats = load_data(selected_year)
# playerstats
# ----------------------------
