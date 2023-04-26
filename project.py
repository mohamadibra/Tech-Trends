import pandas as pd
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objs as go
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sqlite3
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

demog = pd.read_csv('m5_survey_data_demographics.csv')
tech = pd.read_csv('m5_survey_data_technologies_normalised.csv')

tech_demog = pd.merge(tech,demog,on='Respondent',how='inner')

df = tech_demog[['Respondent','Age','EdLevel','ConvertedComp']]

df['Age'].fillna(df['Age'].median(), inplace=True)

df['EdLevel'].fillna(df['EdLevel'].mode()[0], inplace=True)

df.dropna(subset='ConvertedComp',inplace=True)

Q1 = df['ConvertedComp'].quantile(0.25)
Q3 = df['ConvertedComp'].quantile(0.75)
IQR = Q3 = Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['ConvertedComp'] < lower_bound) | (df['ConvertedComp'] > upper_bound)]

df_clean = df[~((df['ConvertedComp'] < lower_bound) | (df['ConvertedComp'] > upper_bound))]

dt = df_clean[['EdLevel','ConvertedComp','Age']]

# create dummy variables for education level
ohe = OneHotEncoder(sparse=False)
ct = make_column_transformer((ohe, ['EdLevel']), remainder='passthrough')
X = ct.fit_transform(dt.drop('ConvertedComp', axis=1))

# define the target variable and fit a linear regression model
y = dt['ConvertedComp']
model = LinearRegression().fit(X, y)

# define the education level options for the dropdown list
ed_level_options = dt['EdLevel'].unique()

# create a sidebar for user inputs
st.sidebar.title('Enter your information')
ed_level = st.sidebar.selectbox('Education level:', ed_level_options)
age = st.sidebar.slider('Age:', 16, 65, 30)

# transform the user inputs and predict the compensation
new_data = pd.DataFrame({'EdLevel': [ed_level], 'Age': [age]})
new_X = ct.transform(new_data)
predicted_compensation = model.predict(new_X)

# display the predicted compensation to the user
st.title('Your predicted compensation')
st.write('Education level:', ed_level)
st.write('Age:', age)
st.write('Predicted compensation:', predicted_compensation[0])

st.title("Tech Trends")

top_ten_lg_ww = tech_demog.groupby('LanguageWorkedWith')['Respondent'].count().reset_index().sort_values('Respondent',ascending=False).head(10)
top_ten_lg_d = tech_demog.groupby('LanguageDesireNextYear')['Respondent'].count().reset_index().sort_values('Respondent',ascending=False).head(10)

fig = sp.make_subplots(rows=1,cols=2,horizontal_spacing=0.4)

fig.add_trace(go.Bar(y=top_ten_lg_ww['LanguageWorkedWith'],x=top_ten_lg_ww['Respondent'],name='Worked Witg',orientation='h',marker=dict(
        color='purple',
        reversescale=True  # reverses the order of the bars
    )),row=1,col=1)
fig.update_xaxes(title_text='LanguageWorkedWith',row=1,col=1)
fig.update_yaxes(title_text='N Respondents',row=1,col=1)

fig.add_trace(go.Bar(y=top_ten_lg_d['LanguageDesireNextYear'],x=top_ten_lg_d['Respondent'],name='Desired Next Year',orientation='h',marker=dict(
        color='purple',
        reversescale=True  # reverses the order of the bars
    )),row=1,col=2)
fig.update_xaxes(title_text='LanguageDesireNextYear',row=1,col=2)
fig.update_yaxes(title_text='N Respondents',row=1,col=2)

fig.update_layout(height=400,width=1200, title='Top 10 Programming Languages worked with and Desired Next Year By Respondents',
)

st.plotly_chart(fig, use_container_width=True)

top_ten_db_ww = tech_demog.groupby('DatabaseWorkedWith')['Respondent'].count().reset_index().sort_values('Respondent',ascending=False).head(10)
top_ten_db_d = tech_demog.groupby('DatabaseDesireNextYear')['Respondent'].count().reset_index().sort_values('Respondent',ascending=False).head(10)

fig = sp.make_subplots(rows=1,cols=2,horizontal_spacing=0.4)

fig.add_trace(go.Bar(y=top_ten_db_ww['DatabaseWorkedWith'],x=top_ten_db_ww['Respondent'],name='Worked Witg',orientation='h',marker=dict(
        color='green',
        reversescale=True  # reverses the order of the bars
    )),row=1,col=1)
fig.update_xaxes(title_text='DatabaseWorkedWith',row=1,col=1)
fig.update_yaxes(title_text='N Respondents',row=1,col=1)

fig.add_trace(go.Bar(y=top_ten_db_d['DatabaseDesireNextYear'],x=top_ten_db_d['Respondent'],name='Desired Next Year',orientation='h',marker=dict(
        color='green',
        reversescale=True  # reverses the order of the bars
    )),row=1,col=2)
fig.update_xaxes(title_text='DatabaseDesireNextYear',row=1,col=2)
fig.update_yaxes(title_text='N Respondents',row=1,col=2)

fig.update_layout(height=400,width=1200, title='Top 10 databses worked with and Desired Next Year By Respondents',
)

st.plotly_chart(fig, use_container_width=True)

