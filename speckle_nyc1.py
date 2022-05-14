#--------------------------
#IMPORTING LIBRARIES
import streamlit
from http import client
import streamlit as st
#pip install specklepy
import specklepy
from specklepy.api.client import SpeckleClient
from specklepy.api.credentials import get_account_from_token
#import pandas
import pandas as pd
#import plotly express
import plotly.express as px
#--------------------------

#--------------------------


#PAGE CONFIG
st.set_page_config(
    page_title= "NYC - Baby Radish",
    page_icon="🗽"
)
#--------------------------

#--------------------------
#CONTAINERS
header = st.container()
input = st.container()
viewer = st.container()
report = st.container()
graphs = st.container()
#--------------------------

#--------------------------
#HEADER
#Page Header
with header:
    st.title("NYC - Baby Radish 🗽🥕")
#about app

with header.expander("About this app🔽", expanded=True):
    st.markdown("""
    Compare APIs, 3d models, and 3d maps of newyourk blah blah...
    """)
#--------------------------

#--------------------------
#INPUTS
with input:
    st.subheader("Add Layers")

    #------------
    #Columns for inputs
    # speckleToken = tokenCol.text_input("Speckle Token", "54cc9934fda25c598900ab2a6a9228c907c2724d87") #JL
    #------------

    #------------
    #CLIENT
    client = SpeckleClient(host="speckle.xyz")
    #get account from token
    account = get_account_from_token("54cc9934fda25c598900ab2a6a9228c907c2724d87", "speckle.xyz")
    #Authenticate
    client.authenticate_with_account(account)
    #------------

    #------------
    #Streams list 👇
    #Selected Stream ✅
    stream = client.stream.search("Hackathons")[0]
    streamId = "c463c6f6bd"
    #Stream Branches 🌴
    branches = client.branch.list(streamId)
    #Stream Commits 🏹
    commits = client.commit.list(streamId, limit=100)

    #Get Stream names
    commitNames = [c.branchName for c in commits]
    #Dropdown for commit selection
    # cName = st.selectbox(label="Add layers", options=commitNames)
    showBldgs = st.checkbox('Show Buildings 🏢')
    showRoads = st.checkbox('Show Roads 🛣️')
    showWater = st.checkbox('Show Water 💧')
    showTrees = st.checkbox('Show Trees 🌳')

    cIDs = []

    cId = ""
    if showBldgs:
        st.write('Buildings 🏢!')
        cId = "d4bc744cc2"
        
    elif showRoads:
        st.write('Roads 🛣️!')
        cId = "fc2c326bee"
    elif showWater:
        st.write('Water 💧!')
        cId = "d2ea061d60"
    elif showTrees:
        st.write('Trees 🌳')
    cIDs.append(cId)

    #Selected commit ✅
    # for c in commits:
    #     if c.branchName == cName:
    #         cId = c.id
    #         break
    # commit = commits(cName)[0]
    # st.write(cName)
    # st.write(cId)
    # st.write(commits)

    # st.write(commits)
    #------------
    
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
    else:
        embed_src = "https://speckle.xyz/embed?stream=c463c6f6bd&commit=6acb9b670f&overlay=" + cId
    return st.components.v1.iframe(src=embed_src, height=400)
    st  
#--------------------------


#--------------------------
#VIEWER 👀
with viewer:
    st.subheader("NYC - 3d viewer 🗽")
    commit2viewer(cId)
#--------------------------

#--------------------------
#REPORT
with report:
    st.subheader("Statistics")
    #------------
    #Columns for cards
    branchCol, childrenCol, connectorCol, contributorCol = st.columns(4)
    #------------

    #------------
    #Branch Card 💳
    branchCol.metric(label="Num of branches (commits)", value = stream.branches.totalCount)
    #List of branches
    listToMarkdown([(b.name + " (" + str(b.commits.totalCount) + " commits)") for b in branches], branchCol)
    #------------

    #------------
    #Children Card 💳
    totalOfChildren = sum(c.totalChildrenCount for c in commits)
    childrenCol.metric(label="Number of elements", value = totalOfChildren)
    numOfChildren = [(str(c.totalChildrenCount) + " (" + c.branchName + ")") for c in commits]
    listToMarkdown(numOfChildren, childrenCol)
    #------------

    #------------
    #Connector Card 💳
    #connector list
    connectorList = [c.sourceApplication for c in commits]
    #onnector names
    connectorNames = list(dict.fromkeys(connectorList))
    #number of connectors
    connectorCol.metric(label="Number of connectors", value = len(dict.fromkeys(connectorList)))
    #list of connectors
    listToMarkdown(connectorNames, connectorCol)
    #------------

    #------------
    #Contributor Card 💳
    contributorCol.metric(label="Number of contributors", value = len(stream.collaborators))
    #contributor names
    contributorNames = list(dict.fromkeys([col.name for col in stream.collaborators]))
    #list of contributors
    listToMarkdown(contributorNames, contributorCol)
    #------------

#--------------------------

#--------------------------
#GRAPHS
with graphs:
    st.subheader("Graphs")
    #columns for charts
    branch_graph_col, connector_graph_col, collaborator_graph_col = st.columns([2,1,1])

    #------------
    #BRANCH GRAPHS 📊
    #branch count
    branch_counts = pd.DataFrame([[b.name, b.commits.totalCount] for b in branches])
    #rename columns
    branch_counts.columns = ["branchName", "totalCommits"]
    #create graph
    branch_count_graph = px.bar(branch_counts, x=branch_counts.branchName, y=branch_counts.totalCommits, color=branch_counts.branchName)
    branch_count_graph.update_layout(
        showlegend = False,
        height = 220,
        margin = dict(l=1, r=1, t=1, b=1)
    )
    branch_graph_col.plotly_chart(branch_count_graph, use_container_width=True)


    #------------

    #------------
    #CONNECTOR CHART 🍩
    commits = pd.DataFrame.from_dict([c.dict() for c in commits])
    #get apps from dataframe
    apps = commits["sourceApplication"]
    #reset index apps
    apps = apps.value_counts().reset_index()
    #rename columns
    apps.columns = ["app", "count"]
    #donut chart
    appsFig = px.pie(apps, names=apps["app"], values=apps["count"], hole=0.5)
    appsFig.update_layout(
        showlegend = False,
        height = 200,
        margin=dict(l=2,r=2,t=2,b=2)
    )
    connector_graph_col.plotly_chart(appsFig, use_container_width=True)
    #------------

    #------------
    #COLLABORATOR CHART 🍩
    authors = commits["authorName"].value_counts().reset_index()
    authors.columns = ["author", "count"]
    #donut chart
    authorFig = px.pie(authors, names=authors["author"], values=authors["count"], hole=0.5)
    authorFig.update_layout(
        showlegend = False,
        height = 200,
        margin=dict(l=2,r=2,t=2,b=2)
    )
    collaborator_graph_col.plotly_chart(authorFig, use_container_width=True)
    #------------

    #------------
    #COMMIT ACTIVITY TIMELINE ⌚
    st.subheader("Commit Activity Timeline ⌚")
    cdate = pd.to_datetime(commits["createdAt"]).dt.date.value_counts().reset_index().sort_values("index")
    #date range to fill null dates
    null_days = pd.date_range(start=cdate["index"].min(), end=cdate["index"].max())
    #add null days to table
    cdate = cdate.set_index("index").reindex(null_days, fill_value=0)
    #reset index
    cdate = cdate.reset_index()
    #rename
    cdate.columns = ["date", "count"]
    #redate indexed dates
    cdate["date"] = pd.to_datetime(cdate["date"]).dt.date

    dateFig = px.line(cdate,x = cdate["date"], y = cdate["count"], markers=True)
    dateFig.update_layout(
        margin=dict(l=2,r=2,t=2,b=2)
    )
    st.plotly_chart(dateFig, use_container_width=True)
    st.write(commits)
    #------------

#--------------------------
