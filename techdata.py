import pandas as pd
import plotly.express as px
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sqlite3
import streamlit as st

CON = sqlite3.connect('m4_survey_data.sqlite')

QUERY = """
SELECT * from DatabaseWorkedWith
"""

dbs = pd.read_sql_query(QUERY,CON)
top_ten = dbs.groupby('DatabaseWorkedWith')['Respondent'].count().reset_index().sort_values('Respondent',ascending=False).head(10)
st.title("Tech Trends")
st.write("Top 10 databses Used By Respondents")
st.plotly_chart(px.bar(top_ten,x='DatabaseWorkedWith',y='Respondent',color='DatabaseWorkedWith'))

demog = pd.read_csv('m5_survey_data_demographics.csv')
tech = pd.read_csv('m5_survey_data_technologies_normalised.csv')

tech_demog = pd.merge(tech,demog,on='Respondent',how='inner')

model = tech_demog['Respondnet','Age','EdLevel','ConvertedComp']