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

# import screeninfo

# Set display mode to inline
# st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide",page_title='Tech Trends',page_icon=':chart_with_upwards_trend:')
# st.set_option('deprecation.showfileUploaderEncoding', False)
# st.set_option('deprecation.showPyplotGlobalUse', False)
# st.set_option('deprecation.showPyplotGlobalUse', False)

# Get the screen width
# screen_width = screeninfo.get_monitors()[0].width

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

# st.title('Tech Trends: Analyzing Top Programming Languages, Databases, Platforms, and Web Frameworks Used and Desired for Learning')

# Define the different sections of your app
sections = ["Executive Summary", "Introduction", "Methodology", "Results", "Discussion", "Conclusion", "Appendix","Compensation Prediction Model"]

# Create a sidebar with a radio button to select the section
selected_section = st.sidebar.radio("", sections)

st.write("<style>ul { margin-left: 20px; } li.no_bullet { list-style-type: none;}</style>", unsafe_allow_html=True)

# Show the appropriate content based on the selected section
if selected_section == "Executive Summary":
    st.title("Executive Summary")
    st.write("To spot trends and forecast the near future in the IT sector, both businesses and individuals must answer the most crucial questions:")
    # Define a list of items
    my_list = ["Top 5 Languages used and desired to learn.",
               "Top 5 Databases used and desired to learn.",
               "Top 5 Web Frameworks used and desired to learn.",
               "Platforms used and desired to learn.",
               "The age, gender location, and education level."
               ]

    # Write the list in HTML format
    html_list = "<ul>" + "\n".join([f"<li>{item}</li>" for item in my_list]) + "</ul>"
    st.markdown(html_list, unsafe_allow_html=True)

    st.write("A stackoverflow survey served well the analysis goals where hundreds of respondents answered these question.")
    st.write("A summary of what we have obtained from the analysis:")
    my_list1 = ["Languages for web developments purposes showed huge dominance where JavaScript, HTML/CSS, SQL where in the Top 5 languages worked with and desired to learn in future.",
               "Databases analysis indicated a possible shift in the future. MySQL, PostgreSQL, and Microsoft SQL Server were in the top 3 worked with. PostgreSQL, MongoDB, and Redis were in the top 3 to learn next year.",
               "Platforms used and desired to learn bring Linux on top, where businesses constantly prioritize security and dependability. Docker, Windows, and AWS are the top three desired skills.",
               "Web Framework analysis revealed that JavaScript Web Frameworks dominated, with JQuery, Angular, and React being the top three worked with. React emerged as the most desired Web Framework next year ranking 33% higher than the second most desired one, Vue.js.",
               "Demographic analysis revealed that men outnumbered women by 93.5% to 6.5%, with the majority of respondents having a BA degree level 50%, 25% MA, and 25% for the remaining education levels.",
               "Most of the respondents where distributed in USA, India, UK, Germany and Canada."
               ]
    # Write the list in HTML format
    html_list1 = "<ul>" + "\n".join([f"<li>{item}</li>" for item in my_list1]) + "</ul>"
    st.markdown(html_list1, unsafe_allow_html=True)

elif selected_section == "Introduction":
    st.title("Introduction")    
    st.write("Analytical study was created to track recent technological progress and identify any potential shifts that may occur in the future. The rate of progress in this sector is rapid, so a significant shift may occur in only 1 year. That is why it is critical to inquire about respondents current technology worked on and preferred to study in the coming year. Each firm's goal is to detect the latest trend in order to stay up-to-date. This analysis is year by year requirement for every IT company.")
    st.write("Graduates and newcomers focus on learning and mastering languages, databases, and web frameworks, while ignoring platforms. That is why it is crucial to ask respondents about the platforms they use so that they can assist beginners.")
    st.write("It is necessary to establish each respondent's demographic background and how they are spread across nations, ages, gender, and education levels. It is also crucial to understand the man:woman ratio in this enormous industry, which is mirrored in the overall gender dominance in IT businesses.")
    st.write("This report analyzes the current state and future trends in the technology landscape, focusing on:")
    my_list2 = ["Languages",
               "Databases",
               "Platforms",
               "Web frameworks",
               "Demographic distribution (age-gender-location-education level)"
               ]
    html_list2 = "<ul>" + "\n".join([f"<li>{item}</li>" for item in my_list2]) + "</ul>"
    st.markdown(html_list2, unsafe_allow_html=True)

elif selected_section == "Methodology":
    st.title("Methodology")
    st.write("It is required for the study to ask IT experts about their everyday technologies utilized and wanted in the future, as well as to have demographic information. For this purpose, the 2019 Stackovreflow survey was the best option.")
    my_list3 = ["Data was used from a stackoverflow developer survey done in 2019.",
                "It was a 20 minutes survey",
                "filled out by 90000 developers.",
                "The Analysis was done on 80% of the total 90000 results."]
    html_list3=""
    for i in my_list3:
        html_list3 += "- " + i + "\n"
    st.markdown(html_list3)
    st.write("The analysis was divided into three major sections:")
    my_list4 = ["Current Technology Usage",
                "Future Technology Trend",
                "Demographics"]
    html_list4=""
    for i in my_list4:
        html_list4 += "- " + i + "\n"
    st.markdown(html_list4)

    st.write("Each technology section was grouped by `Current Year` and `Next Year`:")
    my_list5 = ["Language used this year - Language desire to learn next year",
                "Database used this year - Database desire to learn next year",
                "Plarform used this year - Platform desire to learn next year"
                "Web Framework used this year - Web Framework desire to learn next year"]
    html_list5=""
    for i in my_list5:
        html_list5 += "- " + i + "\n"
    st.markdown(html_list5)
    st.write("Demographics analysis focused on `age`, `gender`, `Education Level`, and `County`")

elif selected_section == "Results":
    st.title("Results")
    st.subheader("Current Technology Used / Future Technology Trend:")
    st.write("Programming Language Trends, Database Trends, Platform Trends, and Web Frameworks Trends are shown below, with each category displaying current year and desire to learn next year, as well as all findings and implications.")

    st.header("Programming Languages")

    top_ten_lg_ww = tech_demog.groupby('LanguageWorkedWith')['Respondent'].count().reset_index().sort_values('Respondent',ascending=True).tail(5)
    top_ten_lg_d = tech_demog.groupby('LanguageDesireNextYear')['Respondent'].count().reset_index().sort_values('Respondent',ascending=True).tail(5)

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

    fig.update_layout(height=400, width=1200, dragmode=False, selectdirection=None, title='Top 5 Programming Languages worked with and Desired Next Year By Respondents',showlegend=True,
        legend=dict(
            x=0.5,
            y=1.05,
            orientation="h",
            yanchor="bottom",
            xanchor="center",
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # create two columns
    left_col, right_col = st.columns(2)

    # add content to left column
    with left_col:
        st.header("Findings")
        # <li class='no_bullet'>&nbsp;</li>
        text = "<ul><li>JavaScript remains on top this and next year.</li><li>JavaScript, HTML/CSS/SQL is in the top 3 worked with.</li><li>Bash/Shell/PowerShell is in the Top 5 and higher than Python.</li><li>Python is trending up with 5239 respondents willing to learn next year.</li><li>Top 5 ranges</li><ul><li>current year: 8687 - 4542</li><li>Next year: 6630 – 4088</li></ul></ul>"
        st.markdown(text,unsafe_allow_html=True)

    # add separator
    # st.sidebar.markdown("---")

    # add content to right column
    with right_col:
        st.header("Implications")
        text = "<ul><li>Gives more popularity for JavaScript web frameworks.</li><li>Web development is highly dominant among developers.</li><li>Automating tasks is essential for developers.</li><li>There might be a shift in the dominance of machine learning and AI in the coming years.</li><li>Reflects the enormous ongoing need for JavaScript, Python, HTML/CSS, and SQL.</li></ul>"
        st.markdown(text,unsafe_allow_html=True)

    st.divider()  # 👈 Draws a horizontal rule

    st.header("DataBases")

    top_ten_db_ww = tech_demog.groupby('DatabaseWorkedWith')['Respondent'].count().reset_index().sort_values('Respondent',ascending=True).tail(5)
    top_ten_db_d = tech_demog.groupby('DatabaseDesireNextYear')['Respondent'].count().reset_index().sort_values('Respondent',ascending=True).tail(5)

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

    fig.update_layout(height=400,width=1200,dragmode=False, selectdirection=None, title='Top 5 databases worked with and Desired Next Year By Respondents',showlegend=True,
        legend=dict(
            x=0.5,
            y=1.05,
            orientation="h",
            yanchor="bottom",
            xanchor="center",
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # create two columns
    left_col, right_col = st.columns(2)

    # add content to left column
    with left_col:
        st.header("Findings")
        text = "<ul><li>MySQL is the most popular database. 25% higher than the second and third place databases.</li><li>PostgreSQL popularity booming with 4328 respondents willing to learn next year.</li><li>Redis and elastic search databases are growing in popularity, with 3331 and 2856 users willing to learn next year, respectively.</li><li>Respondents willing to learn MongoDB is 3649 falling in the second place.</li></ul>"
        st.markdown(text,unsafe_allow_html=True)

    # add separator
    # st.sidebar.markdown("---")

    # add content to right column
    with right_col:
        st.header("Implications")
        text = "<ul><li>Relational database management system (RDBMS) is now the most used for operating databases.</li><li>A possible shift in database systems dominance from (RDBMS) to Object-relational databases (ORDBMS).</li><li>Suggests a rise is the popularity of NoSQL databases and new storage technologies to follow up with.</li><li>shows the relevance of adopting NoSQL for particular internet applications in the next years.</li></ul>"
        st.markdown(text,unsafe_allow_html=True)

    st.divider()  # 👈 Draws a horizontal rule

    st.header("Platforms")

    top_ten_pm_ww = tech_demog.groupby('PlatformWorkedWith')['Respondent'].count().reset_index().sort_values('Respondent',ascending=True).tail(5)
    top_ten_pm_d = tech_demog.groupby('PlatformDesireNextYear')['Respondent'].count().reset_index().sort_values('Respondent',ascending=True).tail(5)

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

    fig.update_layout(dragmode=False, selectdirection=None,height=400,width=1200, title='Top 5 Platforms worked with and Desired Next Year By Respondents',showlegend=True,
        legend=dict(
            x=0.5,
            y=1.05,
            orientation="h",
            yanchor="bottom",
            xanchor="center",
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # create two columns
    left_col, right_col = st.columns(2)

    # add content to left column
    with left_col:
        st.header("Findings")
        text = "<ul><li>Linux on top, follows windows is on top with 5811, 5563 respectively.</li><li>Linux is the top platform desired to learn next year while docker userser are rapidly increasing.</li><li>Linux, Docker, Windows, and AWS continue to be the most wanted platforms for this year and next.</li</ul>"
        st.markdown(text,unsafe_allow_html=True)

    # add separator
    # st.sidebar.markdown("---")

    # add content to right column
    with right_col:
        st.header("Implications")
        text = "<ul><li>Linux and windows are currently the most used platforms by respondents.</li><li>Linux is still trending higher for next year with, Docker is expected to boom in the coming years.</li><li>Every IT professional should be able to grasp one or two of these platforms since they are in high demand today and in the future.</li></ul>"
        st.markdown(text,unsafe_allow_html=True)

    st.divider()  # 👈 Draws a horizontal rule

    st.header("Web Framework")

    top_ten_wf_ww = tech_demog.groupby('WebFrameWorkedWith')['Respondent'].count().reset_index().sort_values('Respondent',ascending=True).tail(5)
    top_ten_wf_d = tech_demog.groupby('WebFrameDesireNextYear')['Respondent'].count().reset_index().sort_values('Respondent',ascending=True).tail(5)

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

    fig.update_layout(dragmode=False, selectdirection=None,height=400,width=1200, title='Top 5 Web Frameworks worked with and Desired Next Year By Respondents',showlegend=True,
        legend=dict(
            x=0.5,
            y=1.05,
            orientation="h",
            yanchor="bottom",
            xanchor="center",
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # create two columns
    left_col, right_col = st.columns(2)

    # add content to left column
    with left_col:
        st.header("Findings")
        text = "<ul><li>JQuery on top, then Angular.js, then React.js on top 4629, 3327, 3302 respectively.</li><li>React is on top of web frameworks desired next year with 4714 and 33% higher than Vue.js 3143 which on 2nd place.</li><li>JQuery, Angular, React, and ASP.NET are remain among the top five web frameworks this year and next year.</li</ul>"
        st.markdown(text,unsafe_allow_html=True)

    # add separator
    # st.sidebar.markdown("---")

    # add content to right column
    with right_col:
        st.header("Implications")
        text = "<ul><li>JavaScript related web frameworks is currently dominating the web frameworks.</li><li>Powered by JavaScript, the number of React users is projected to grow in the next years.</li><li class='no_bullet'>&nbsp;</li><li>Web frameworks powered by JavaScript and .NET is hugely required for every web developer.</li></ul>"
        st.markdown(text,unsafe_allow_html=True)

    st.divider()  # 👈 Draws a horizontal rule

    st.header("Demographics")



    # st.subheader("Gender Distribution")
    m_f_tech_demog = tech_demog.query("Gender == 'Man' or Gender == 'Woman'")
    m_f_tech_demog['Gender'].value_counts()
    # Count number of men and women
    counts = m_f_tech_demog['Gender'].value_counts()

    # Calculate percentage of men and women
    percentages = counts / counts.sum()

    # Create pie chart
    fig = px.pie(
        values=percentages,
        names=percentages.index,
        title='Gender Distribution'
    )

    fig.update_layout(dragmode=False, selectdirection=None,legend=dict(
        x=0,
        y=1,
        traceorder='normal',
        font=dict(
            family='sans-serif',
            size=12,
            color='white'
        ),
        # bgcolor='LightSteelBlue',
        # bordercolor='Black',
        # borderwidth=2),
        ),
        height=400
        )
    
    st.plotly_chart(fig, use_container_width=True)

    # add separator
    # st.sidebar.markdown("---")

   

    # st.subheader("Country Distribution")
    # Group by country and count respondents
    tech_demog_loc = tech_demog.groupby('Country')['Respondent'].count().reset_index().sort_values('Respondent',ascending=False)

    # Create choropleth map
    data = go.Choropleth(
        locations=tech_demog_loc['Country'],
        z=tech_demog_loc['Respondent'],
        locationmode='country names',
        colorscale='Blues',
        marker_line_color='#010101',
        marker_line_width=0.5,
    )

    layout = go.Layout(
        title='Country Distribution',
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='natural earth',
            ),
        height=500,
        margin=dict(l=0, r=0, t=70, b=0),
    )

    fig = go.Figure(data=data, layout=layout)

    # Display map
    st.plotly_chart(fig)       
    
    # group by Gender and EdLevel and count the number of respondents
    df_count = m_f_tech_demog.groupby(['Gender', 'EdLevel'])['Respondent'].count().reset_index()
    # create stacked bar chart
    fig = px.bar(df_count, y='EdLevel', x='Respondent', color='Gender', barmode='group', orientation='h',
                title='Number of Respondents by Education Level and Gender')
    fig.update_layout(dragmode=False, selectdirection=None,xaxis_title='Number of Respondents', yaxis_title='Education Level',height=450)

    # Display the chart
    st.plotly_chart(fig)

    tech_demog_age = m_f_tech_demog.groupby('Age')['Respondent'].count().reset_index().sort_values('Age',ascending=False)
    fig = go.Figure(go.Scatter(
    x=tech_demog_age['Age'],
    y=tech_demog_age['Respondent'],
    mode='lines+markers',
    marker=dict(size=6),
    line=dict(width=2, color='royalblue'),
    hovertemplate='Age: %{x}<br>Respondent: %{y}',
    ))

    # Customize the layout
    fig.update_layout(dragmode=False, selectdirection=None,
        title='Number of Respondents by Age',
        xaxis_title='Age',
        yaxis_title='Number of Respondents',
        height = 350,
        width=500
    )
    st.plotly_chart(fig)

    st.subheader("Findings – Implications")
    st.write(" This shows the country's technical development, as IT Specialists shape the technology status of their country. Highest age is 99, lowest age is 16, the age with the most respondents (4942) is 28. 31 is the average age of all respondents.Most respondents have a bachelor's degree, followed by a master's degree, and then college/university studies without a degree with 37600, 17012, 8983 respondents respectively. It is worth noting that over half of the respondents hold a bachelor's degree.")
    text = "<ul><li>Men IT Specialists is 93.5% where Women is 6.5%, about 14:1 ratio.</li><li>USA, India, UK, Germany and Canada are the Top 5 countries with IT specialists of range 20818 – 2775. This shows the country's technical development, as IT Specialists shape the technology status of their country.</li><li>Highest age is 99, lowest age is 16, the age with the most respondents (4942) is 28. 31 is the average age of all respondents.</li><li>Most respondents have a bachelor's degree, followed by a master's degree, and then college/university studies without a degree with 5342, 2484, 1280 respondents respectively. It is worth noting that over half of the respondents hold a bachelor's degree.</li></ul>"
    st.markdown(text,unsafe_allow_html=True)
elif selected_section == "Discussion":
    st.title("Discussion")
    st.subheader("The analyzed data was able to answer all questions and revealed valuable hidden information that is needed for all IT specialists and Tech Companies.")
    st.subheader("Unexpected complete dominance for Men was revealed. Also Bash/Shell/PowerShell was unexpectedly well adopted and heavily relied on by IT Specialists.")
    st.subheader("It was Astonitiong how everything powered by javascript is currently dominating and expected to increase in future.")
    st.subheader("It is crystal clear that SQL, javascript, python, Bash, and linux is significantly important tech to carry for any IT Specialist.")
    st.subheader('Many new visions can be extracted from this study for now and the near future.')
elif selected_section == "Conclusion":
    st.title("Conclusion")
    st.subheader("IT Specialist Needs to Master at least Two of each:")
    text = "<ul><li>Language: JavaScript, HTML/CSS, Python, SQL.</li><li>Databases: MySQL, PostgreSQL, MongoDB, Redis.</li><li>Platforms: Linux, Docker, AWS, Windows.</li><li>Web Frameworks(if web developer): React.js, Angular.js, Vue.js, ASP.NET.</li></ul>"
    st.markdown(text,unsafe_allow_html=True)
    st.subheader('Extracted Knowledge:')
    text2 = "<ul><li>Python, Docker, PostgreSQL, MongoDB, and React is growing fast.</li><li>Everything Powered by javascript is currently booming and still growing.</li><li>Emerging technologies spot the lights on cloud computing, machine learning, and AI.</li><li>Data Science is expected to boom in future.</li><li>Cloud computing technologies are widely employed.</li></ul>"
    st.markdown(text2,unsafe_allow_html=True)
elif selected_section == "Appendix":
    st.title("Appendix")
    rel_bash = tech_demog[tech_demog['LanguageWorkedWith'] == 'Bash/Shell/PowerShell']
    rel_bash = rel_bash.groupby('PlatformWorkedWith').count().sort_values('Respondent',ascending=False).reset_index()
    rel_bash = rel_bash[['PlatformWorkedWith','Respondent']]
    st.subheader("Answering the question: Why are Bash/Shell/PowerShell among the top languages used? Here's a diagram of each platform to which this language contributes.")
    fig = px.bar(rel_bash,x='Respondent',y='PlatformWorkedWith',orientation='h')
    fig.update_layout(dragmode=False, selectdirection=None,xaxis_title="Number of Respondents",yaxis_title="Platform using Bash/Shell/Powershell",title='Bash/Shell/PowerShell Langauge Users for each Platform',width=400,margin=dict(l=0, r=0, t=170, b=0))
    st.plotly_chart(fig)

    st.divider()

    st.subheader("Python is the #1 option for data scientists, followed by SQL and R. The preceding charts demonstrate the significance of Bash/Shell/PowerShell and SQL is always required. We may conclude that a Data Scientist must be fluent in at least three languages. SQL, Python/R, Bash/Shell/PowerShell.")
    data_scientist = tech_demog.query("DevType == 'Data scientist or machine learning specialist'")

    left_col, right_col = st.columns(2)

    with left_col:
        DS1 = data_scientist.groupby('LanguageWorkedWith')['Respondent'].count().reset_index().sort_values('Respondent',ascending=False)
        fig = px.bar(DS1,x='Respondent',y='LanguageWorkedWith',orientation='h')
        fig.update_layout(dragmode=False, selectdirection=None,xaxis_title="Number of Respondents",yaxis_title="Language Worked With",title='Data Scientists languages Worked With',width=400,margin=dict(l=0, r=0, t=170, b=0))
        st.plotly_chart(fig)
    
    with right_col:
        DS2 = data_scientist.groupby('LanguageDesireNextYear')['Respondent'].count().reset_index().sort_values('Respondent',ascending=False)
        fig = px.bar(DS2,x='Respondent',y='LanguageDesireNextYear',orientation='h')
        fig.update_layout(dragmode=False, selectdirection=None,xaxis_title="Number of Respondents",yaxis_title="Language Desire Next Year",title='Data Scientists languages Desired Next Year',width=400,margin=dict(l=0, r=0, t=170, b=0))
        st.plotly_chart(fig)
    
    st.divider()

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
    fig.update_layout(dragmode=False, selectdirection=None,yaxis=dict(title='N Respondnets'),yaxis2=dict(title='Mean Age'),height=500,width=400,title='Top 20 Countries By Respondents and Thier average Age')

    st.plotly_chart(fig)

elif selected_section == "Compensation Prediction Model":
    st.title("Compensation Model")
    st.write("compensation refers to monetary payment given to an individual in exchange for their services.In the workplace, compensation is what is earned by employees. It includes salary or wages in addition to commission and any incentives or perks that come with the given employee's position.")
    st.write("Based on the compensation data collected from respondents, a compensation prediction model dependent on 'age' and 'Education level' was developed.")
    
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
    st.write('Predicted compensation: ', f'<span style="color:lightgreen">$ {int(predicted_compensation[0])}</span>', unsafe_allow_html=True)