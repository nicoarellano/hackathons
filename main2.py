#--------------------------
#IMPORTING LIBRARIES
#import streamlit
from http import client
import streamlit as st
#import pandas
# import pandas as pd
#import plotly express
# import plotly.express as px
#--------------------------

#--------------------------
#PAGE CONFIG
st.set_page_config(
    page_title= "NYC - Baby Daikon Radish",
    page_icon="🗽"
)
#--------------------------

#--------------------------
#CONTAINERS
header = st.container()
input = st.container()
viewer = st.container()
viewerMap = st.container()
report = st.container()
graphs = st.container()
#--------------------------

#--------------------------
#HEADER
#Page Header
with header:
    st.title("NYC - Baby Daikon Radish 🗽🥕")
#about app

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
    Accomplishments that you're proud of?
    - Working as a team to decipher complex data sets and organize them through architecture and geography using open source software
    - Bringing together data science and architecture
    """)
#--------------------------

#--------------------------
#INPUTS
with input:
    st.subheader("Add Layers 🍰")

    #------------
    #Streams list 👇
    #Selected Stream ✅
    streamId = "c463c6f6bd"
    
    showBldgs = st.checkbox('Show Buildings 🏢')
    showRoads = st.checkbox('Show Roads 🛣️')
    showWater = st.checkbox('Show Water 💧')
    # showTrees = st.checkbox('Show Trees 🌳')
    cIds = []
    cId = ""
    if showBldgs:
        st.write('Buildings 🏢!')
        cId = "d4bc744cc2"
        cIds.append(cId)
    if showRoads:
        st.write('Roads 🛣️!')
        cId = "fc2c326bee"
        cIds.append(cId)
    if showWater:
        st.write('Water 💧!')
        cId = "d2ea061d60"
        cIds.append(cId)

#--------------------------
#DEFINITIONS
#Python list to markdown list
def listToMarkdown(list, column):
    list = ["- " + i + " \n" for i in list]
    list = "".join(list)
    return column.markdown(list)

#creates an iframe from commit
def commit2viewer(cId):
    if cId == "":
        embed_src = "https://speckle.xyz/embed?stream=c463c6f6bd&commit=6acb9b670f"
    elif len(cIds) == 1:
        embed_src = "https://speckle.xyz/embed?stream=c463c6f6bd&commit=6acb9b670f&overlay=" + cIds[0]
    elif len(cIds) == 2:
        embed_src = "https://speckle.xyz/embed?stream=c463c6f6bd&commit=6acb9b670f&overlay=" + cIds[0] + "," + cIds[1]
    elif len(cIds) == 3:
        embed_src = "https://speckle.xyz/embed?stream=c463c6f6bd&commit=6acb9b670f&overlay=" + cIds[0] + "," + cIds[1] + "," + cIds[2]
    return st.components.v1.iframe(src=embed_src, height=400)
#--------------------------

#--------------------------
#VIEWER 👀
with viewer:
    st.subheader("NYC - Speckle viewer 🗽🟦👀")
    commit2viewer(cId)
#--------------------------

#VIEWER MAP 👀
with viewerMap:
    st.subheader("Map 🗺️")
    st.image("https://raw.githubusercontent.com/nicoarellano/hackathons/main/assets/ny-map.jpg?token=GHSAT0AAAAAABT7YK2CH3VQD6M3PPX4EIOEYT7K4IA", caption="NY MAP", width=700)
#--------------------------

#GRAPHS 📊
with viewerMap:
    st.subheader("Graphs 📊")
    leftCol, rightCol = st.columns(2)

    leftCol.write("Bronx County")
    leftCol.image("https://raw.githubusercontent.com/nicoarellano/hackathons/main/assets/bronx_county_map.jpg?token=GHSAT0AAAAAABT7YK2DSIZSHXSAJ5PVEN6CYT7K3RQ", caption="Bronx MAP", width=300)
    leftCol.write("New York County")
    leftCol. image("https://raw.githubusercontent.com/nicoarellano/hackathons/main/assets/new_york_county_map.jpg?token=GHSAT0AAAAAABT7YK2D7WWAB5AMPBUPM23IYT7KXHA", caption="NY MAP", width=300)
    rightCol.write("Niagara County")
    rightCol.image("https://raw.githubusercontent.com/nicoarellano/hackathons/main/assets/niagara_county_map.jpg?token=GHSAT0AAAAAABT7YK2DKDFXLHEGAD5WEJSGYT7KX3A", caption="Niagara MAP", width=300)
    rightCol.write("Orange County")
    rightCol.image("https://raw.githubusercontent.com/nicoarellano/hackathons/main/assets/orange_county_map.jpg?token=GHSAT0AAAAAABT7YK2C7XPM4BKMJSEI6VDKYT7KYIQ", caption="Orange MAP", width=300)

#--------------------------

