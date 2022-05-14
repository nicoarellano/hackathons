# --------------------------
# IMPORTING LIBRARIES
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
# import streamlit
from http import client
import streamlit as st
# import specklepy
from specklepy.api.client import SpeckleClient
from specklepy.api.credentials import get_account_from_token
# import pandas
import pandas as pd
# import plotly express
import plotly.express as px

# IMPORTING OTHER LIBRARIES

# --------------------------
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (20, 9)
# --------------------------
# PAGE CONFIG
st.set_page_config(
    page_title="NYC - Baby Daikon Radish",
    page_icon="🗽"
)
# --------------------------

# --------------------------
# CONTAINERS
header = st.container()
viewerMap = st.container()
radars = st.container()
input = st.container()
viewer = st.container()
report = st.container()
graphs = st.container()
# --------------------------

# --------------------------
# HEADER
# Page Header
with header:
    st.title('New York State - Best Neighbourhood Finder 🗽')
    st.write("@ Baby Daikon Radish")
    # 🥕🥒
# about app

with header.expander("About this app🔽", expanded=False):
    st.markdown("""
    What was the inspiration behind the Hack?
    - Bringing together data science and architecture to allow a richer interpretation of select data sets
    - Connecting data points to BIM and GIS using the browser
    - Investigating graphical environments to make building data more accessible and test it against non building related open data sources\n
    What does the hack do ? How was it built?
    - Data is organized from a number of sources including environmental, social/political, and architectural
    - Browser hosts the model that visualizes the data points
    - Data from buildings and surrounding are parsed and visualized through text and graphs
    - The hack was built from open data sources collected from New York databases, parsed and graphed in Kaggle, then mapped in QGIS and viewed in the browser using leaflet.js. The buildings were integrated using openstreetmaps to blender, then shared through Speckle.\n
    What technology was involved in particular?
    - Speckle
    - Leaflet
    - Python
    - Kaggle
    - Javascript
    - Streamlit
    Accomplishments that you're proud of?
    - Working as a team to decipher complex data sets and organize them through architecture and geography using open source software
    - Bringing together data science and architecture
    - Uncover meaningful insights from open data sources
        - US Census Bureau https://www.census.gov
        - OPEN NY  https://data.ny.gov
        - Tree Equity Score https://www.treeequityscore.org
    """)
# --------------------------

# --------------------------
# SIDE BAR
# Dropdown for radar chart
st.sidebar.header('Social & Environmental Radar Chart 🕸️')
selected_county = st.sidebar.selectbox(
    'County', list(['New York County', 'Bronx County', 'Orange County', 'Niagara County']))

# Checkbox for commit selection
st.sidebar.header('NYC 3D Viewer 🍰')
showBldgs = st.sidebar.checkbox('Show Buildings 🏢', value='Show Buildings 🏢')
showRoads = st.sidebar.checkbox('Show Roads 🛣️', value='Show Roads 🛣️')
showWater = st.sidebar.checkbox('Show Water 💧', value='Show Water 💧')
# --------------------------


# --------------------------
# RADARS
with radars:

    # ------------
    # 0.1 Geo Data
    ny_counties = gpd.read_file('./newyork-counties.geojson')
    ny_counties['IBRC_Geo_ID'] = ny_counties['geoid'].str[-5:]
    ny_counties['state_geo_ID'] = ny_counties['geoid'].str[-5:-3]
    ny_counties['county_geo_ID'] = ny_counties['geoid'].str[-3:]
    ny_counties['IBRC_Geo_ID'] = ny_counties['IBRC_Geo_ID'].astype('int')
    # st.write(ny_counties)

    # ------------
    # 0.2 Social Conext Data
    # import csv
    social = pd.read_csv('./Social Context.csv')
    social = social[social.Description.str.contains('NY')]

    # drop cols
    social[['IBRC_Geo_ID', 'Year']].groupby('IBRC_Geo_ID').max().describe()
    social_2019 = social[social.Year == 2019]
    social_2019 = social_2019.reset_index().drop(columns={'index'})
    social_2019 = social_2019.iloc[:, ([0, 3, 4, 5, 6, 7])].reset_index()

    # social code dictionary for mapping
    social_code_df = social_2019[['Social_Context_Code', 'Social_Context_Code_Description']].groupby('Social_Context_Code_Description').mean(
    ).astype(int).unstack().to_frame().reset_index().drop(columns='level_0').rename(columns={0: 'Social_Context_Code'})
    social_code_dict = social_code_df.set_index(
        'Social_Context_Code').to_dict()

    social_2019_pivot = pd.pivot_table(social_2019, values='Social_Context_Domain_Data', index=[
                                       'IBRC_Geo_ID', 'Description', 'Social_Context_Code_Description'], aggfunc='median')
    social_2019_unstack = social_2019_pivot.unstack()
    social_2019_unstack = social_2019_unstack.droplevel(level=0, axis=1)

    # ------------
    # 0.3  Age Data
    age = pd.read_csv('./Population by Age and Sex - US, States, Counties.csv')
    age = age[age.Description.str.contains(', NY')]
    age[['IBRC_Geo_ID', 'Year']].groupby('IBRC_Geo_ID').max().describe()
    age_2019 = age[age.Year == 2019]
    age_2019 = age_2019.reset_index().drop(columns=[
        'index', 'Statefips', 'Countyfips', 'Year', 'Male Population', 'Female Population'])

    age_2019['percent_Population 0-4'] = age_2019['Population 0-4'] / \
        age_2019['Total Population']
    age_2019['percent_Population 5-17'] = age_2019['Population 5-17'] / \
        age_2019['Total Population']
    age_2019['percent_Population 18-24'] = age_2019['Population 18-24'] / \
        age_2019['Total Population']
    age_2019['percent_Population 25-44'] = age_2019['Population 25-44'] / \
        age_2019['Total Population']
    age_2019['percent_Population 45-64'] = age_2019['Population 45-64'] / \
        age_2019['Total Population']
    age_2019['percent_Population 65+'] = age_2019['Population 65+'] / \
        age_2019['Total Population']

    # ------------
    # 04 Tree Social Equity Data
    ny_tree_equity = gpd.read_file('./ny.shp')
    ny_tree_equity['IBRC_Geo_ID'] = ny_tree_equity['geoid'].astype(str).str[:5]

    # Convert column types
    int_cols = ['geoid', 'IBRC_Geo_ID']

    str_cols = ['state', 'county', 'geometry', 'ua_name',
                'incorpname', 'congressio', 'biome', 'source']

    float_cols = ['total_pop', 'pctpov', 'pctpoc',
                  'unemplrate', 'medhhinc', 'dep_ratio', 'child_perc', 'seniorperc',
                  'treecanopy', 'area', 'avg_temp', 'bgpopdense', 'popadjust', 'tc_gap', 'tc_goal',
                  'phys_hlth', 'ment_hlth', 'asthma', 'core_m', 'core_w', 'core_norm',
                  'healthnorm', 'priority', 'tes', 'tesctyscor']

    ny_tree_equity[int_cols] = ny_tree_equity[int_cols].astype(int)
    ny_tree_equity[str_cols] = ny_tree_equity[str_cols].astype(str)
    ny_tree_equity[float_cols] = ny_tree_equity[float_cols].astype(float)
    ny_tree_equity['geoid_county'] = ny_tree_equity['geoid'].astype(
        str).str[:5]

    tmp = ny_tree_equity[['geoid', 'geoid_county']
                         ].groupby('geoid_county').median()

    tmp1 = tmp.geoid.astype(int).to_frame()
    geoid_geoid_county_dict = tmp1.copy()
    geoid_geoid_county_dict = geoid_geoid_county_dict.reset_index()

    ny_tree_equity_county_num = ny_tree_equity.groupby('county').mean()
    column_median = ny_tree_equity_county_num.median()

    # fill missing values with median
    ny_tree_equity_county_num = ny_tree_equity_county_num.fillna(column_median)
    # we will merge only selected cols from TREE data to QGIS
    ny_tree_equity_county_num.columns.to_list()

    selected_cols = ['pctpov', 'pctpoc', 'unemplrate', 'medhhinc',
                     'dep_ratio', 'treecanopy',
                     'avg_temp', 'phys_hlth',
                     'ment_hlth', 'asthma', 'core_norm', 'healthnorm',
                     'tes', 'tesctyscor', ]

    ny_tree_equity_county_all = ny_tree_equity_county_num[selected_cols].rename(columns={'pctpov': 'percent_poverty',
                                                                                         'pctpoc': 'percent_people_of_color',
                                                                                         'unemplrate': 'unemployment_rate',
                                                                                         'medhhinc': 'median_household_income',
                                                                                         'dep_ratio': 'dependency_ratio',
                                                                                         'treecanopy': 'tree_canopy',
                                                                                         'avg_temp': 'hot_summer_avg_temp',
                                                                                         'phys_hlth': 'physical_health_challenges',
                                                                                         'ment_hlth': 'mental_health_challenges',
                                                                                         'asthma': 'athma_challenge',
                                                                                         'core_norm': 'heart challenges',
                                                                                         'healthnorm': 'health_problem',
                                                                                         'tes': 'tree_equity_score',
                                                                                         'tesctyscor': 'tree_equity_score_municipal'})

    # add 'geoid' column here (geoid:geoid_county)
    ny_tree_equity_county_all['geoid_county'] = ny_tree_equity_county_num['geoid'].astype(
        str).str[:5]
    ny_tree_equity_county_all.drop(
        columns=['tree_canopy', 'tree_equity_score_municipal'], inplace=True)
    # ny_tree_equity_county_all

    # ------------
    # 0.5 Mergy NY data
    ny_geo = pd.merge(ny_counties, social_2019_unstack,
                      on='IBRC_Geo_ID', how='left')
    ny_geo = pd.merge(ny_geo, age_2019, on='IBRC_Geo_ID', how='left')
    ny_geo['geoid_county'] = ny_geo['geoid'].astype(str).str[-5:]

    # this is important TREES vs QGIS
    ny_geo = pd.merge(ny_geo, ny_tree_equity_county_all,
                      on='geoid_county', how='left')

    # export to csv
    # ny_geo.to_csv('./ny-counties-1_cols_added_v1.csv')

    # Counties with TREE data
    # st.write('Counties with TREE data:')
    # st.write(ny_geo[ny_geo['percent_poverty'].notna()].shape[0])
    # st.dataframe(ny_geo[ny_geo['percent_poverty'].notna()]['name'])

    # Counties without TREE data
    print('Counties with missing TREE data: ')
    print(ny_geo[ny_geo['percent_poverty'].isna()].shape[0])
    # ny_geo[ny_geo['percent_poverty'].isna()]['name']

    # missing values in final ny_geo dataframe
    # pd.DataFrame(ny_geo.isnull().sum()).sort_values(0,ascending=False).style.background_gradient(axis=0)

    # fill in missing values with median
    median_vals = ny_geo.median()
    # print('median values of each column: \n\n', median_vals)
    ny_geo_final = ny_geo.fillna(median_vals)
    # ny_geo_final['athma_challenge'].value_counts()

    # Correlation Heatmap
    st.write('1.0 : Positive correlation between the two x,y categories')
    st.write('0.0: Zero correlation between the two x,y categories')
    st.write('-1.0: Negative correlation between the two x,y categoris')
    corr = ny_geo_final.corr().drop(
        columns=['IBRC_Geo_ID'], index=['IBRC_Geo_ID'])

    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(45, 30))
        ax = sns.heatmap(corr, mask=mask, square=False,
                         annot=True, fmt='.1f', annot_kws={"fontsize": 15})
    sns.set(font_scale=3)
    st.pyplot(f)

    # ------------
    # Export merged file
    # Export to CSV
    ny_geo_final.to_csv('./ny_geo_final.csv')
    # Export to JSON
    # ny_geo_final.to_json()

    # ------------
    # Create Radar Chart
    num_cols = ny_geo_final.select_dtypes(include=np.number).columns.tolist()
    # ny_geo_final[num_cols]

    # Scale to between 0 to 1
    scaler = MinMaxScaler()
    scaled_ny = scaler.fit_transform(ny_geo_final[num_cols])
    scaled_ny = pd.DataFrame(scaled_ny, columns=num_cols)

    # ------------
    # ------------
    st.subheader('Radar Chart')
    st.write(
        "- dependency_ratio: the dependency ratio(childrens + seniors / 18-64 adults")
    st.write(
        "- physical_health: the self reported physical health challenges of the people in the block group(a percentage")
    st.write(
        "- mental_health:  the self reported mental health challenges of people in the block group(a percentage")
    st.write(
        "- asthma challenges: the self reported asthma challenges of people in the block group(a percentage")
    st.write(
        "- heart challenges: the self reported coronary heart challenges of people in the block group(a percentage")
    st.write("- health challenges: the normalized health index of the block group")
    st.write("- tree_equity_score: the tree equity score of the block group")
    # Output of Dropdown
    if selected_county == 'New York County':
        # st.write('Show New York map')
        # New York County (index=29)
        df = scaled_ny[['heart challenges', 'athma_challenge', 'Entrepreneurship', 'tree_equity_score', 'Hopefulness', 'mental_health_challenges',
                        'Gender Equality', 'hot_summer_avg_temp', 'Belief In Science', 'Neuroticism', 'Income Per Capita', 'Income Mobility']]
        cols = scaled_ny[['heart challenges', 'athma_challenge', 'Entrepreneurship', 'tree_equity_score', 'Hopefulness', 'mental_health_challenges',
                          'Gender Equality', 'hot_summer_avg_temp', 'Belief In Science', 'Neuroticism', 'Income Per Capita', 'Income Mobility']].columns

        fig = px.line_polar(
            r=df.loc[29].values,
            theta=cols,
            line_close=True,
            range_r=[0, 1.0],
            title='New York County vs Other Counties in New York State:')
        fig.update_traces(fill='toself')
        st.plotly_chart(fig, use_container_width=True)

    if selected_county == 'Bronx County':
        # st.write('Show Bronx County map')
        # Bronx County (index=42)
        df = scaled_ny[['heart challenges', 'athma_challenge', 'Entrepreneurship', 'tree_equity_score', 'Hopefulness', 'mental_health_challenges',
                        'Gender Equality', 'hot_summer_avg_temp', 'Belief In Science', 'Neuroticism', 'Income Per Capita', 'Income Mobility']]
        cols = scaled_ny[['heart challenges', 'athma_challenge', 'Entrepreneurship', 'tree_equity_score', 'Hopefulness', 'mental_health_challenges',
                          'Gender Equality', 'hot_summer_avg_temp', 'Belief In Science', 'Neuroticism', 'Income Per Capita', 'Income Mobility']].columns

        fig = px.line_polar(
            r=df.loc[42].values,
            theta=cols,
            line_close=True,
            range_r=[0, 1.0],
            title='Bronx County vs Other Counties in New York State:')
        fig.update_traces(fill='toself')
        st.plotly_chart(fig, use_container_width=True)

    if selected_county == 'Orange County':
        # st.write('Show Orange County map')
        # Orange County
        df = scaled_ny[['heart challenges', 'athma_challenge', 'Entrepreneurship', 'tree_equity_score', 'Hopefulness', 'mental_health_challenges',
                        'Gender Equality', 'hot_summer_avg_temp', 'Belief In Science', 'Neuroticism', 'Income Per Capita', 'Income Mobility']]
        cols = scaled_ny[['heart challenges', 'athma_challenge', 'Entrepreneurship', 'tree_equity_score', 'Hopefulness', 'mental_health_challenges',
                          'Gender Equality', 'hot_summer_avg_temp', 'Belief In Science', 'Neuroticism', 'Income Per Capita', 'Income Mobility']].columns

        fig = px.line_polar(
            r=df.loc[54].values,
            theta=cols,
            line_close=True,
            range_r=[0, 1.0],
            title='Orange County vs Other Counties in New York State:')
        fig.update_traces(fill='toself')
        st.plotly_chart(fig, use_container_width=True)

    if selected_county == 'Niagara County':
        # st.write('Show Niagara County map')
        # Niagara County
        df = scaled_ny[['heart challenges', 'athma_challenge', 'Entrepreneurship', 'tree_equity_score', 'Hopefulness', 'mental_health_challenges',
                        'Gender Equality', 'hot_summer_avg_temp', 'Belief In Science', 'Neuroticism', 'Income Per Capita', 'Income Mobility']]
        cols = scaled_ny[['heart challenges', 'athma_challenge', 'Entrepreneurship', 'tree_equity_score', 'Hopefulness', 'mental_health_challenges',
                          'Gender Equality', 'hot_summer_avg_temp', 'Belief In Science', 'Neuroticism', 'Income Per Capita', 'Income Mobility']].columns

        fig = px.line_polar(
            r=df.loc[60].values,
            theta=cols,
            line_close=True,
            range_r=[0, 1.0],
            title='Niagara County vs Other Counties in New York State:')
        fig.update_traces(fill='toself')
        st.plotly_chart(fig, use_container_width=True)
    # ------------
    # ------------
# --------------------------


# --------------------------
# INPUTS
with input:
    # ------------
    # Columns for inputs
    # speckleToken = tokenCol.text_input("Speckle Token", "54cc9934fda25c598900ab2a6a9228c907c2724d87") #JL
    # ------------

    # ------------
    # CLIENT
    client = SpeckleClient(host="speckle.xyz")
    # get account from token
    account = get_account_from_token(
        "54cc9934fda25c598900ab2a6a9228c907c2724d87", "speckle.xyz")
    # Authenticate
    client.authenticate_with_account(account)
    # ------------

    # ------------
    # Streams list 👇
    # Selected Stream ✅
    stream = client.stream.search("Hackathons")[0]
    streamId = "c463c6f6bd"
    # Stream Branches 🌴
    branches = client.branch.list(streamId)
    # Stream Commits 🏹
    commits = client.commit.list(streamId, limit=100)

    # Get Stream names
    commitNames = [c.branchName for c in commits]

    # Apped layers to Speckle 3D view
    cIds = []
    cId = ""
    if showBldgs:
        # st.write('Buildings 🏢!')
        cId = "d4bc744cc2"
        cIds.append(cId)
    if showRoads:
        # st.write('Roads 🛣️!')
        cId = "fc2c326bee"
        cIds.append(cId)
    if showWater:
        # st.write('Water 💧!')
        cId = "d2ea061d60"
        cIds.append(cId)
# --------------------------


# --------------------------
# DEFINITIONS
# Python list to markdown list


def listToMarkdown(list, column):
    list = ["- " + i + " \n" for i in list]
    list = "".join(list)
    return column.markdown(list)

# creates an iframe from commit


def commit2viewer(cId):
    if cId == "":
        embed_src = "https://speckle.xyz/embed?stream=c463c6f6bd&commit=6acb9b670f&overlay="
        # embed_src = "https://speckle.xyz/embed?stream=c463c6f6bd&commit=6acb9b670f"
    elif len(cIds) == 1:
        embed_src = "https://speckle.xyz/embed?stream=c463c6f6bd&commit=6acb9b670f&overlay=" + \
            cIds[0]
    elif len(cIds) == 2:
        embed_src = "https://speckle.xyz/embed?stream=c463c6f6bd&commit=6acb9b670f&overlay=" + \
            cIds[0] + "," + cIds[1]
    elif len(cIds) == 3:
        embed_src = "https://speckle.xyz/embed?stream=c463c6f6bd&commit=6acb9b670f&overlay=" + \
            cIds[0] + "," + cIds[1] + "," + cIds[2]
    return st.components.v1.iframe(src=embed_src, height=400)
# --------------------------


# --------------------------
# VIEWER 👀
with viewer:
    st.subheader("Explore New York City in 3D Viewer 👀")
    commit2viewer(cId)
# --------------------------

# VIEWER MAP 👀
with viewerMap:
    st.subheader("Tree Equity Score Map 🗺️")
    st.write("Red: Trees are accessble to divers demographics (i.e. people of color, income level, education level, etc.)")
    st.write("Yellow: Trees access is not easily accessible by divers demographics(i.e. low income, low education, people of colour populations.")

    st.image("./ny-map.jpg", caption="NY MAP", width=700)
    # url = 'http://babydaikonradish.rf.gd/?i=1'
    # r = requests.get(url)
    # st.components.v1.iframe(src='', height=400)
    # r.text

    # st.write(my_map)
# --------------------------


# --------------------------
# REPORT
with report:
    st.subheader("3D Viewer Statistics 🔢")
    # ------------
    # Columns for cards
    branchCol, childrenCol, connectorCol, contributorCol = st.columns(4)
    # ------------

    # ------------
    # Branch Card 💳
    branchCol.metric(label="Num of branches (commits)",
                     value=stream.branches.totalCount)
    # List of branches
    listToMarkdown([(b.name + " (" + str(b.commits.totalCount) + " commits)")
                   for b in branches], branchCol)
    # ------------

    # ------------
    # Children Card 💳
    totalOfChildren = sum(c.totalChildrenCount for c in commits)
    childrenCol.metric(label="Number of elements", value=totalOfChildren)
    numOfChildren = [(str(c.totalChildrenCount) + " (" + c.branchName + ")")
                     for c in commits[0:-17]]
    listToMarkdown(numOfChildren, childrenCol)
    # ------------

    # ------------
    # Connector Card 💳
    # connector list
    connectorList = [c.sourceApplication for c in commits]
    # onnector names
    connectorNames = list(dict.fromkeys(connectorList))
    # number of connectors
    connectorCol.metric(label="Number of connectors",
                        value=len(dict.fromkeys(connectorList)))
    # list of connectors
    listToMarkdown(connectorNames, connectorCol)
    # ------------

    # ------------
    # Contributor Card 💳
    contributorCol.metric(label="Number of contributors",
                          value=len(stream.collaborators))
    # contributor names
    contributorNames = list(dict.fromkeys(
        [col.name for col in stream.collaborators]))
    # list of contributors
    listToMarkdown(contributorNames, contributorCol)
    # ------------

# --------------------------

# --------------------------
# GRAPHS
with graphs:
    st.subheader("3D Viewer Graphs 📊")
    # columns for charts
    branch_graph_col, connector_graph_col, collaborator_graph_col = st.columns([
        2, 1, 1])

    # ------------
    # BRANCH GRAPHS 📊
    # branch count
    branch_counts = pd.DataFrame(
        [[b.name, b.commits.totalCount] for b in branches])
    # rename columns
    branch_counts.columns = ["branchName", "totalCommits"]
    # create graph
    branch_count_graph = px.bar(branch_counts, x=branch_counts.branchName,
                                y=branch_counts.totalCommits, color=branch_counts.branchName)
    branch_count_graph.update_layout(
        showlegend=False,
        height=220,
        margin=dict(l=1, r=1, t=1, b=1)
    )
    branch_graph_col.write("Number of commits per building:")
    branch_graph_col.plotly_chart(branch_count_graph, use_container_width=True)

    # ------------

    # ------------
    # CONNECTOR CHART 🍩
    commits = pd.DataFrame.from_dict([c.dict() for c in commits])
    # get apps from dataframe
    apps = commits["sourceApplication"]
    # reset index apps
    apps = apps.value_counts().reset_index()
    # rename columns
    apps.columns = ["app", "count"]
    # donut chart
    appsFig = px.pie(apps, names=apps["app"], values=apps["count"], hole=0.5)
    appsFig.update_layout(
        showlegend=False,
        height=200,
        margin=dict(l=2, r=2, t=2, b=2)
    )
    connector_graph_col.write("Origin of commits:")
    connector_graph_col.plotly_chart(appsFig, use_container_width=True)
    # ------------

    # ------------
    # COLLABORATOR CHART 🍩
    authors = commits["authorName"].value_counts().reset_index()
    authors.columns = ["author", "count"]
    # donut chart
    authorFig = px.pie(
        authors, names=authors["author"], values=authors["count"], hole=0.5)
    authorFig.update_layout(
        showlegend=False,
        height=200,
        margin=dict(l=2, r=2, t=2, b=2)
    )
    collaborator_graph_col.write("Author of commit:")
    collaborator_graph_col.plotly_chart(authorFig, use_container_width=True)
    # ------------

    # ------------
    # COMMIT ACTIVITY TIMELINE ⌚
    st.subheader("3D Viewer Commit Activity Timeline ⌚")
    cdate = pd.to_datetime(commits["createdAt"]).dt.date.value_counts(
    ).reset_index().sort_values("index")
    # date range to fill null dates
    null_days = pd.date_range(
        start=cdate["index"].min(), end=cdate["index"].max())
    # add null days to table
    cdate = cdate.set_index("index").reindex(null_days, fill_value=0)
    # reset index
    cdate = cdate.reset_index()
    # rename
    cdate.columns = ["date", "count"]
    # redate indexed dates
    cdate["date"] = pd.to_datetime(cdate["date"]).dt.date

    dateFig = px.line(cdate, x=cdate["date"], y=cdate["count"], markers=True)
    dateFig.update_layout(
        margin=dict(l=2, r=2, t=2, b=2)
    )
    st.plotly_chart(dateFig, use_container_width=True)
    # ------------

# --------------------------
