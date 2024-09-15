import pandas as pd
import streamlit as st
import pickle
import numpy as np
import xgboost
from xgboost import XGBRegressor
import sklearn
# print(sklearn.__version__)

#loading the pickle file
pipe=pickle.load(open('pipe.pkl','rb'))
# print(pipe)

#cricket playing nation
teams=[
    'Australia',
    'India',
    'New Zealand',
    'South Africa',
    'Bangladesh',
    'England',
    'West Indies',
    'Ireland',
    'Afghanistan',
    'Pakistan',
]

#cricket stadium
cities=['Dubai',
 'Mirpur',
 'Johannesburg',
 'Auckland',
 'Cape Town',
 'Colombo',
 'Barbados',
 'London',
 'Durban',
 'Sydney',
 'Wellington',
 'St Lucia',
 'Hamilton',
 'Melbourne',
 'Manchester',
 'Lauderhill',
 'Centurion',
 'Abu Dhabi',
 'Nottingham',
 'Mumbai',
 'Mount Maunganui',
 'Pallekele',
 'Southampton',
 'Cardiff',
 'Kolkata',
 'St Kitts',
 'Greater Noida',
 'Christchurch',
 'Nagpur',
 'Trinidad']

#web pag development
st.title('Cricket Score Predictor')
page_bg_img = f"""
    <style>
        [data-testid="stAppViewContainer"] > .main {{
        background-image: url("https://i.postimg.cc/4xgNnkfX/Untitled-design.png");
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        background-attachment: fixed; /* Change to fixed if preferred */
        }}
        [data-testid="stHeader"] {{
        background: rgba(0,203,0,0);
        }}
    </style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

col1,col2=st.columns(2)
with col1:
    batting_team=st.selectbox('Select batting team',sorted(teams))
with col2:
    bowling_team=st.selectbox('Select bowling team',sorted(teams))

city=st.selectbox('Select City',sorted(cities))

col3,col4,col5=st.columns(3)
with col3:
    current_score=st.number_input('Current Score')
with col4:
    overs=st.number_input('Overs Done(works for more than 5 overs)')
with col5:
    wickets=st.number_input('Wickets out')

last_five=st.number_input('Runs Scored in last 5 overs')

if st.button('Predict Score'):
    balls_left=120-overs*6
    wickets_left=10-wickets
    crr=current_score/overs

    input_df=pd.DataFrame({
        'batting_team':[batting_team],
        'bowling_team':[bowling_team],
        'city':[city],
        'Current_score':[current_score],
        'balls_left':[balls_left],
        'wickets_left':[wickets_left],
        'curr':[crr],
        'last_five':[last_five]
    })

    # st.table(input_df)
    # st.text(xgboost.__version__)
    # st.text(sklearn.__version__)
    result=pipe.predict(input_df)
    st.header('Predicted Score:'+str(int(result[0])))
