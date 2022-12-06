import streamlit as st
import pandas as pd
import seaborn as sns
#import altair as alt
#import matplotlib.pyplot as plt
import nltk
import plotly.graph_objects as go

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
st.set_page_config(layout="wide")

st.title("Data Science Jobs Trend Analysis ")
data = pd.read_csv('C:/Users/Neeraja/Desktop/IUB-DS/SEMESTER 3/DATA VIS/salaries_data_viz.csv')
st.header("Data Overview")
st.write(data.head())
# And within an expander
Stats=st.header("Statistics About Data")
my_expander = st.expander("Statistics About Data", expanded=True)
with my_expander:
       c1,c2=st.columns((2,1))
       with c1:
          st.write(data.describe().transpose())
       
       with c2:

            null_values=(data.isnull().sum().sum())
            st.markdown("<h1 style='text-align: center; color: black;'>Null Values </h1>", unsafe_allow_html=True)
            st.markdown("<h1 style='text-align: center; color: black;'>0 </h1>", unsafe_allow_html=True)
            
            

Stats=st.header("Visualizations")
my_expander = st.expander("Visualizations", expanded=True)
with my_expander:   
    residence = data['company_location'].value_counts()
    top10_employee_location = residence[:10]
    fig_loc = px.bar(y=top10_employee_location.values, 
             x=top10_employee_location.index, 
             color = top10_employee_location.index,
             
             text_auto=True,
             title= 'Top 10 Locations of Companies',
             )
    fig_loc.update_layout(
        xaxis_title="Location",
        yaxis_title="Count",
        font = dict(size=17,family="Franklin Gothic"))
    c1,c2=st.columns((2,1))
    with c1:
          st.plotly_chart(fig_loc,use_container_width=True)
       
    with c2:
            #st.subheader("Data - US Location")
            st.markdown("<h3 style='text-align: center; color: black;'>Data - US Location </h3>", unsafe_allow_html=True)
            data=data[data['company_location'].str.contains("US")]
            st.write(data.head().transpose())
           
st.markdown("<h4 style='text-align: center; color: black;'>We can say that majority of the companies reside in US </h4>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: left; color: black;'>Categorical Variables Vs Salary Distribution</h4>", unsafe_allow_html=True)
plt.figure(figsize = (10,10))
plt.subplot(2,3,1)
sns.stripplot(x='employment_type', y='salary_in_usd', data=data)
plt.subplot(2,3,2)
sns.stripplot(x='company_size', y='salary_in_usd', data=data)
plt.subplot(2,3,3)
sns.stripplot(x='experience_level', y='salary_in_usd', data=data)
plt.subplot(2,3,4)
sns.boxplot(x="employment_type",y="salary_in_usd",data=data)
plt.subplot(2,3,5)
sns.boxplot(x="company_size",y="salary_in_usd",data=data)
plt.subplot(2,3,6)
sns.boxplot(x="experience_level",y="salary_in_usd",data=data)
#st.pyplot(plt)


emp=data["employment_type"]
com=data["company_size"]
exp=data["experience_level"]
sal=data["salary_in_usd"]


# Create figure
fig = go.Figure()

# Add scatter
fig.add_trace(go.Box(x=emp,y=sal,boxpoints='all'))

# Add drowdowns


 

fig.update_layout(
    updatemenus=[
         dict(buttons=list([

                        dict(
                args=["type", "Box"],
                label="Box Plot",
                method="restyle"
            ),
            dict(
                args=["type", "violin"],
                label="Violin Plot",
                method="restyle"
            )]),
            direction="down",
        ),
        go.layout.Updatemenu(
            buttons=list([
                dict(
                    args=["x", [emp]],
                    label="Exployment Type",
                    method="restyle"
                ),
                dict(
                    args=["x", [com]],
                    label="Company Size",
                    method="restyle"
                ),
                dict(
                    args=["x", [exp]],
                    label="Experience",
                    method="restyle"
                ),
            ]),
            direction="down",
            pad={"l": 2, "b": 10},
            showactive=True,
            
            xanchor="right",
           
            yanchor="bottom"
        ),
      
    ]
)



st.plotly_chart(fig,use_container_width=True)

c1,c2=st.columns((2,2))
with c1 :
    remote_type = ['Fully Remote','Partially Remote','No Remote Work']

    fig = go.Figure()
    fig=px.bar(x = ['Fully Remote','Partially Remote','No Remote Work'], 
       y = data['remote_ratio'].value_counts().values,
       color = remote_type,
       #color_discrete_sequence=px.colors.sequential.dense,
       text_auto=True,
       title = 'Remote Working Ratio Distribution',
       #template='plotly_dark'
            )
    
    fig.update_layout(


        xaxis_title="Remote Type",
        yaxis_title="Count",
        font = dict(size=17,family="Franklin Gothic"))    
# showing the plot
    st.plotly_chart(fig,use_container_width=True)
with c2:
    remote_type = ['Fully Remote','Partially Remote','No Remote Work']

    fig = go.Figure()
    fig=px.histogram(x = data["work_year"],nbins=6 ,color=data["remote_ratio"],
      
      
       #color = remote_type,
       #color_discrete_sequence=px.colors.sequential.dense,
       text_auto=True,
      title = 'Year Vs Remote Working ratio',
       #template='plotly_dark'
       labels={"remote_ratio":"0-Fully Remote 50- Partially Remote 100- fully remote"}
            )
    
    fig.update_layout(


        xaxis_title="Remote Type",
        yaxis_title="count",
        font = dict(size=17,family="Franklin Gothic"))    
# showing the plot
    st.plotly_chart(fig,use_container_width=True)

st.markdown("<h4 style='text-align: left; color: black;'>Most of the Employees are working from home . From 2020 to 2022 Employees working from fully remote has increased drastically</h4>", unsafe_allow_html=True)    
c1,c2=st.columns((2,2))
with c1 :
    data1=data.groupby(['job_title']).size() .sort_values(ascending=False) .reset_index(name='count')  
    data2=data1.head(10)
    x=data2["job_title"]
    y=data2["count"]

    fig = go.Figure()
    fig=px.bar(x = x, 
       y =y,
       color = x,
       #color_discrete_sequence=px.colors.sequential.dense,
       #text_auto=True,
       title = 'Top 10 Job Designations',
       #template='plotly_dark'
            )
    
    fig.update_layout(

        
        xaxis_title="",
        
        yaxis_title="Count",
        xaxis_tickangle=-20, 
        font = dict(size=15,family="Franklin Gothic"))    
# showing the plot
     
# showing the plot
    st.plotly_chart(fig,use_container_width=True,height=2200)
with c2:
    data1=data.groupby(['job_title']).size() .sort_values(ascending=False) .reset_index(name='count')  
    data2=data1
    fig = px.pie(data2[:10], values='count' ,names='job_title', title='Top Job Titles In Data Science',hole=0.3)
    fig.update_layout(


        xaxis_title="",
        yaxis_title="%",
        font = dict(size=17,family="Franklin Gothic"))   
    st.plotly_chart(fig,use_container_width=True)

st.markdown("<h4 style='text-align: left; color: black;'> Data Engineer is the most demand job designation followed by Data Scientist . Both of these designations hold more than 50% of the Designations each comprising of 31.7 % and 28.6% </h4>", unsafe_allow_html=True)    

data1=data.groupby(['salary_in_usd','job_title']).size().reset_index( )
data2=(data1[-15:])
fig1 = go.Figure()
fig1=px.bar(x=data2['job_title'],y=data2['salary_in_usd'],color=data2['salary_in_usd'],
      
       #color = remote_type,
       #color_discrete_sequence=px.colors.sequential.dense,
text_auto=True,
title = 'Top 15 Highest paid Job Roles',
      # title = 'Remote Working Ratio Distribution',
       #template='plotly_dark'
            )
    
fig1.update_layout(xaxis_title="",
height=700,
yaxis_title="Salary",
xaxis_tickangle=15, 
font = dict(size=16,family="Franklin Gothic"))    
# showing the plot
st.plotly_chart(fig1,use_container_width=True)
st.markdown("<h4 style='text-align: left; color: black;'>Research Scientist ranks the highest paid job next followed by Data Scientist . In the Top 15 highest paid salaries , Data Engineer  is the highest occured role  </h4>", unsafe_allow_html=True)    
c1,c2,c3=st.columns((2,2,2))
with c1:
    company_size = st.selectbox('Company Size ',options= data['company_size'].unique(),key=5)
    st.write(" S(small) - Less than 50 employees , M(medium) - 50 to 250 employees , L(large) - More than 250 employees")
    
with c2:
    experience = st.selectbox('Experience ',options= data['experience_level'].unique(),key=6)
    st.write("EN - Entry level / Junior, MI - Mid level / Intermediate, SE - Senior level / Expert, EX - Executive level / Director")
with c3:
    data2=data[(data['company_size'] == company_size )& (data['experience_level'] == experience)][['job_title','salary_in_usd','employment_type']]    
    employmentType = st.selectbox('Employment Type ',options= data2['employment_type'].unique(),key=7)
    st.write("PT - Part time, FT - Full time, CT - Contract, FL - Freelance")

data1=data[(data['company_size'] == company_size )& (data['experience_level'] == experience)&(data['employment_type'] == employmentType)][['job_title','salary_in_usd']]
data1=data1.groupby(['salary_in_usd','job_title']).size().reset_index()

data2=(data1[-15:])
fig1 = go.Figure()
fig1=px.bar(x=data2['job_title'],y=data2['salary_in_usd'],color=data2['salary_in_usd'],
      
       #color = remote_type,
       #color_discrete_sequence=px.colors.sequential.dense,
text_auto=True,
title = 'Top 15 Highest paid Job Roles',
      # title = 'Remote Working Ratio Distribution',
       #template='plotly_dark'
            )
    
fig1.update_layout(xaxis_title="",
yaxis_title="Salary",
font = dict(size=17,family="Franklin Gothic"))    
# showing the plot
st.plotly_chart(fig1,use_container_width=True)
c1,c2,c3=st.columns((2,2,2))
with c1:
    st.markdown("<h4 style='text-align: left; color: black;'>Observations : </h4>", unsafe_allow_html=True) 
    st.markdown("<h6 style='text-align: left; color: black;'> Large Sized company and Full time :</h6>", unsafe_allow_html=True) 
    st.markdown("<h6 style='text-align: left; color: black;'>In Senior level :Data Scientist and Data Analytics lead are highly paid </h6>", unsafe_allow_html=True) 
    st.markdown("<h6 style='text-align: left; color: black;'>In Entry level :Machine Learning Engineer and Machine Learning  Scientist are highly paid </h6>", unsafe_allow_html=True)   
    st.markdown("<h6 style='text-align: left; color: black;'>In Medium  level :Applied Machine Learning  Scientist is highly paid </h6>", unsafe_allow_html=True)     

with c2 : 
     st.markdown("<h4 style='text-align: left; color: black;'> </h4>", unsafe_allow_html=True) 
     st.markdown("<h6 style='text-align: left; color: black;'> Only Full time employees in Executive level and in small sized companied contract employees are also there in executive level  </h6>", unsafe_allow_html=True)    
c1,c2,c3=st.columns((2,2,2))
with c1:
    st.markdown("<h4 style='text-align: left; color: black;'>Employment Type Vs Count </h4>", unsafe_allow_html=True)    
    data_fulltimevsParttime = data[['employment_type', 'salary_in_usd']].groupby('employment_type').count().rename(columns={'salary_in_usd': 'No_of_jobs'}).sort_values('No_of_jobs', ascending=False)
    data_fulltimevsParttime.reset_index(inplace=True)
    st.write(data_fulltimevsParttime.head()  )
    st.markdown("<h4 style='text-align: left; color: black;'>Most of the Jobs are Full time next followed by Contract  </h4>", unsafe_allow_html=True)        
with c2 :
        st.markdown("<h4 style='text-align: left; color: black;'>Designations that pay more than the Average  </h4>", unsafe_allow_html=True)    
    
    # Jobs that pay morethan median
    
        avg_median=data[(data['salary_in_usd'] >= np.median(data['salary_in_usd']))][['job_title']]
        avg_median=avg_median.groupby(['job_title']).size().sort_values(ascending=False)  
        st.write(avg_median[:5] )  
        st.markdown("<h4 style='text-align: left; color: black;'>Data Scientist job role ranks first in paying the employees more than the Average salaries related to Data Science Domain </h4>", unsafe_allow_html=True)        
with c3 : 
        st.markdown("<h8 style='text-align: left; color: black;'>Next 5.....  </h8>", unsafe_allow_html=True)    
        st.write(avg_median[5:10] )  
st.markdown("<h4 style='text-align: left; color: black;'>Jobs Vs Salary more than Average Salary  </h4>", unsafe_allow_html=True)          
c1,c2,c3=st.columns((2,2,2))

with c1:

    Experience_level = st.selectbox('Employment level ',options= data['experience_level'].unique())
    st.write("PT - Part time, FT - Full time, CT - Contract, FL - Freelance")
    med_data = data[(data['experience_level'] == Experience_level) & (data['salary_in_usd'] >= np.median(data['salary_in_usd']))][['job_title']]
    med_data=med_data.groupby(['job_title']).size() .sort_values(ascending=False) .reset_index(name='count') 
    med_data=med_data[:10]

    fig = go.Figure(
            data=[go.Pie(labels=med_data["job_title"], values=med_data['count'])]
        )

    st.plotly_chart(fig, use_container_width=True)

with c2:
    Experience_level = st.selectbox('Employment level',options= data['experience_level'].unique(),key=1)
    Remote_status = st.selectbox(' Remote Status ',options= data['remote_ratio'].unique(),key=2)
    st.write("0 - No Remote Work (less than 20%), 50 - Partially Remote, 100 - Fully Remote (more than 80%) ")
    exp_remote_jobs = data[(data['experience_level'] == Experience_level) & (data['remote_ratio']==Remote_status) &(data['salary_in_usd'] >= np.median(data['salary_in_usd']))][['job_title']]    
    exp_remote_jobs=exp_remote_jobs.groupby(['job_title']).size() .sort_values(ascending=False) .reset_index(name='count') 
    exp_remote_jobs=exp_remote_jobs[:5]
    fig = go.Figure(
            data=[go.Pie(labels=exp_remote_jobs["job_title"], values=exp_remote_jobs['count'])]
        )

    st.plotly_chart(fig, use_container_width=True)
with c3:
    Experience_level = st.selectbox('Employment level ',options= data['experience_level'].unique(),key=3)
    employment_type = st.selectbox('Employment Type',options= data['employment_type'].unique(),key=4)
    st.write("PT - Part time, FT - Full time, CT - Contract, FL - Freelance")
    exp_remote_jobs = data[(data['experience_level'] == Experience_level) & (data['employment_type']==employment_type) &(data['salary_in_usd'] >= np.median(data['salary_in_usd']))][['job_title']]    
    exp_remote_jobs=exp_remote_jobs.groupby(['job_title']).size() .sort_values(ascending=False) .reset_index(name='count') 
    exp_remote_jobs=exp_remote_jobs[:5]
    fig = go.Figure(
            data=[go.Pie(labels=exp_remote_jobs["job_title"], values=exp_remote_jobs['count'])]
        )

    st.plotly_chart(fig, use_container_width=True)

    
from PIL import Image
image = Image.open('C:/Users/Neeraja/Desktop/IUB-DS/SEMESTER 3/DATA VIS/project/word_cloud.PNG')  

st.markdown("<h4 style='text-align: left; color: black;'>Word Cloud - Designation</h4>", unsafe_allow_html=True)
new_image = image.resize((600, 600)) 
st.image(image)


