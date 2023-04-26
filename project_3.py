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
from streamlit.components.v1 import html
import streamlit.components.v1 as components
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer



demog = pd.read_csv('m5_survey_data_demographics.csv')
tech = pd.read_csv('m5_survey_data_technologies_normalised.csv')

tech_demog = pd.merge(tech,demog,on='Respondent',how='inner')

df = tech_demog[['Respondent','Age','EdLevel','ConvertedComp','Country']]

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

# Define the compensation prediction tab
def prediction_model():
    
    st.title('Compensation Prediction Model')
    st.write('Based on data collected from respondents in the Tech field, a Linear model was trained to predict the compansation depending on `age` and `education level`')
    
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

    # get user inputs
    ed_level = st.selectbox('Education level:', ed_level_options)
    age = st.slider('Age:', 16, 65, 30)

    # transform the user inputs and predict the compensation
    new_data = pd.DataFrame({'EdLevel': [ed_level], 'Age': [age]})
    new_X = ct.transform(new_data)
    predicted_compensation = model.predict(new_X)

    # display the predicted compensation to the user
    st.title('Your predicted compensation')
    st.write('Education level:', ed_level)
    st.write('Age: ', f'<span style="color:lightgreen">{age}</span>', unsafe_allow_html=True)
    # st.write('Predicted compensation: $', predicted_compensation[0])
    st.write('Predicted compensation: ', f'<span style="color:lightgreen">$ {predicted_compensation[0]}</span>', unsafe_allow_html=True)

# Define the technical data tab
def technical_data():
    st.title('Technical Data')

    top_ten_lg_ww = tech_demog.groupby('LanguageWorkedWith')['Respondent'].count().reset_index().sort_values('Respondent',ascending=True).tail(10)
    top_ten_lg_d = tech_demog.groupby('LanguageDesireNextYear')['Respondent'].count().reset_index().sort_values('Respondent',ascending=True).tail(10)

    fig = sp.make_subplots(rows=1,cols=2,horizontal_spacing=0.4,column_widths=[0.5, 0.5])

    fig.add_trace(go.Bar(y=top_ten_lg_ww['LanguageWorkedWith'],x=top_ten_lg_ww['Respondent'],name='Worked With',orientation='h',marker=dict(
            color='lightgreen',
            reversescale=True  # reverses the order of the bars
        )),row=1,col=1)
    fig.update_xaxes(title_text='LanguageWorkedWith',row=1,col=1)
    fig.update_yaxes(title_text='N Respondents',row=1,col=1)

    fig.add_trace(go.Bar(y=top_ten_lg_d['LanguageDesireNextYear'],x=top_ten_lg_d['Respondent'],name='Desired Next Year',orientation='h',marker=dict(
            color='darkgreen',
            reversescale=True  # reverses the order of the bars
        )),row=1,col=2)
    fig.update_xaxes(title_text='LanguageDesireNextYear',row=1,col=2)
    fig.update_yaxes(title_text='N Respondents',row=1,col=2)

    fig.update_layout(height=400,width=1200, title='Top 10 Programming Languages worked with and Desired Next Year By Respondents',showlegend=True,
        legend=dict(
            x=0.5,
            y=1.05,
            orientation="h",
            yanchor="bottom",
            xanchor="center",
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    top_ten_db_ww = tech_demog.groupby('DatabaseWorkedWith')['Respondent'].count().reset_index().sort_values('Respondent',ascending=True).tail(10)
    top_ten_db_d = tech_demog.groupby('DatabaseDesireNextYear')['Respondent'].count().reset_index().sort_values('Respondent',ascending=True).tail(10)

    fig = sp.make_subplots(rows=1,cols=2,horizontal_spacing=0.4,column_widths=[0.5, 0.5])

    fig.add_trace(go.Bar(y=top_ten_db_ww['DatabaseWorkedWith'],x=top_ten_db_ww['Respondent'],name='Worked With',orientation='h',marker=dict(
            color='lightblue',
            reversescale=True  # reverses the order of the bars
        )),row=1,col=1)
    fig.update_xaxes(title_text='DatabaseWorkedWith',row=1,col=1)
    fig.update_yaxes(title_text='N Respondents',row=1,col=1)

    fig.add_trace(go.Bar(y=top_ten_db_d['DatabaseDesireNextYear'],x=top_ten_db_d['Respondent'],name='Desired Next Year',orientation='h',marker=dict(
            color='darkblue',
            reversescale=True  # reverses the order of the bars
        )),row=1,col=2)
    fig.update_xaxes(title_text='DatabaseDesireNextYear',row=1,col=2)
    fig.update_yaxes(title_text='N Respondents',row=1,col=2)

    fig.update_layout(height=400,width=1200, title='Top 10 databases worked with and Desired Next Year By Respondents',showlegend=True,
        legend=dict(
            x=0.5,
            y=1.05,
            orientation="h",
            yanchor="bottom",
            xanchor="center",
        )
    )

    st.plotly_chart(fig, use_container_width=True)


    top_ten_pm_ww = tech_demog.groupby('PlatformWorkedWith')['Respondent'].count().reset_index().sort_values('Respondent',ascending=True).tail(10)
    top_ten_pm_d = tech_demog.groupby('PlatformDesireNextYear')['Respondent'].count().reset_index().sort_values('Respondent',ascending=True).tail(10)

    fig = sp.make_subplots(rows=1,cols=2,horizontal_spacing=0.4,column_widths=[0.5, 0.5])

    fig.add_trace(go.Bar(y=top_ten_pm_ww['PlatformWorkedWith'],x=top_ten_pm_ww['Respondent'],name='Worked With',orientation='h',marker=dict(
            color='lightcyan',
            reversescale=True  # reverses the order of the bars
        )),row=1,col=1)
    fig.update_xaxes(title_text='PlatformWorkedWith',row=1,col=1)
    fig.update_yaxes(title_text='N Respondents',row=1,col=1)

    fig.add_trace(go.Bar(y=top_ten_pm_d['PlatformDesireNextYear'],x=top_ten_pm_d['Respondent'],name='Desired Next Year',orientation='h',marker=dict(
            color='darkcyan',
            reversescale=True  # reverses the order of the bars
        )),row=1,col=2)
    fig.update_xaxes(title_text='PlatformDesireNextYear',row=1,col=2)
    fig.update_yaxes(title_text='N Respondents',row=1,col=2)

    fig.update_layout(height=400,width=1200, title='Top 10 Platforms worked with and Desired Next Year By Respondents',showlegend=True,
        legend=dict(
            x=0.5,
            y=1.05,
            orientation="h",
            yanchor="bottom",
            xanchor="center",
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    top_ten_wf_ww = tech_demog.groupby('WebFrameWorkedWith')['Respondent'].count().reset_index().sort_values('Respondent',ascending=True).tail(10)
    top_ten_wf_d = tech_demog.groupby('WebFrameDesireNextYear')['Respondent'].count().reset_index().sort_values('Respondent',ascending=True).tail(10)

    fig = sp.make_subplots(rows=1,cols=2,horizontal_spacing=0.4,column_widths=[0.5, 0.5])

    fig.add_trace(go.Bar(y=top_ten_wf_ww['WebFrameWorkedWith'],x=top_ten_wf_ww['Respondent'],name='Worked With',orientation='h',marker=dict(
            color='lightpink',
            reversescale=True  # reverses the order of the bars
        )),row=1,col=1)
    fig.update_xaxes(title_text='WebFrameWorkedWith',row=1,col=1)
    fig.update_yaxes(title_text='N Respondents',row=1,col=1)

    fig.add_trace(go.Bar(y=top_ten_wf_d['WebFrameDesireNextYear'],x=top_ten_wf_d['Respondent'],name='Desired Next Year',orientation='h',marker=dict(
            color='darkmagenta',
            reversescale=True  # reverses the order of the bars
        )),row=1,col=2)
    fig.update_xaxes(title_text='WebFrameDesireNextYear',row=1,col=2)
    fig.update_yaxes(title_text='N Respondents',row=1,col=2)

    fig.update_layout(height=400,width=1200, title='Top 10 Web Frameworks worked with and Desired Next Year By Respondents',showlegend=True,
        legend=dict(
            x=0.5,
            y=1.05,
            orientation="h",
            yanchor="bottom",
            xanchor="center",
        )
    )

    st.plotly_chart(fig, use_container_width=True)

# Define the demographic data tab
def demographic_data():
    st.title('Demographic Data')
    # Add your demographic data visualizations and analysis here
    st.write('<t1 style="font-weight:bold"> Average Age, Number of Respondents For each Eduaction Level </t1>',unsafe_allow_html=True)
    EdLevel_age = tech_demog.groupby(['EdLevel']).agg({'Respondent':pd.Series.count,'Age':np.mean}).reset_index()
    st.write(EdLevel_age)

    country_age = tech_demog.groupby(['Country']).agg({'Respondent':pd.Series.count,'Age':np.mean}).reset_index().sort_values('Respondent',ascending=False).head(20)
    # st.write(country_age)

    # create a subplot with two axes
    fig = sp.make_subplots(specs=[[{"secondary_y": True}]])

    # add a bar trace to the subplot
    fig.add_trace(go.Bar(x=country_age['Country'], y=country_age['Respondent'], name='N Respondents',marker=dict(color='lightblue')))

    # add a line trace to the subplot's secondary axis
    fig.add_trace(go.Scatter(x=country_age['Country'], y=country_age['Age'], name='Mean Age', mode='lines', yaxis='y2',marker=dict(color='blue')))

    # update the layout to show the secondary axis
    fig.update_layout(yaxis=dict(title='N Respondnets'),yaxis2=dict(title='Mean Age'),width=900,height=600,title='Top 20 Countries By Respondents and Thier mean Age')

    st.plotly_chart(fig)

# Define your Streamlit app
def main():
    st.set_page_config(page_title='Tech Trends', page_icon=':chart_with_upwards_trend:')

    # st.title('My Streamlit App')

    # Create the navigation bar
    nav_options = ["Technical Data", "Demographic Data", "Compensation Prediction Model"]
    page = st.sidebar.container()
    with page:
        st.sidebar.header("Tech Fields")
        selection = st.sidebar.radio("", nav_options)

    # Show the appropriate page content
    if selection == "Technical Data":
        technical_data()
    elif selection == "Demographic Data":
        demographic_data()
    elif selection == "Compensation Prediction Model":
        prediction_model()


# Run your Streamlit app
if __name__ == '__main__':
    main()
