import numpy as np
import streamlit as st
# import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# st.title("ipl win predictor")
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="centered"
)
# model =pickle.load(open('pipe2.pkl','rb'))
# print("model uploaded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0f1e;
    color: #e8eaf0;
}

.stApp {
    background: linear-gradient(160deg, #0a0f1e 0%, #0d1a2e 60%, #0a1628 100%);
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero banner ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid rgba(255,140,0,0.2);
    margin-bottom: 2rem;
}

.hero-eyebrow {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.25em;
    color: #ff8c00;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1.05;
    color: #ffffff;
    margin: 0;
}

.hero-title span {
    color: #ff8c00;
}

.hero-sub {
    font-size: 0.9rem;
    color: #7a8499;
    margin-top: 0.6rem;
    font-weight: 300;
}

/* ── Section label ── */
.section-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    color: #ff8c00;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    margin-top: 1.8rem;
}

/* ── Card panels ── */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* ── Selectbox & number input overrides ── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #111827 !important;
    border-color: rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
}

div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
    border-color: #ff8c00 !important;
    box-shadow: 0 0 0 2px rgba(255,140,0,0.15) !important;
}

/* ── Labels ── */
label[data-testid="stWidgetLabel"] p {
    color: #9aa3b5 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #ff8c00, #e65c00) !important;
    color: #ffffff !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2.5rem !important;
    width: 100%;
    margin-top: 1.2rem;
    transition: opacity 0.2s, transform 0.15s;
}

div[data-testid="stButton"] > button:hover {
    opacity: 0.88;
    transform: translateY(-1px);
}

/* ── Result bar ── */
.result-wrapper {
    margin-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.07);
    padding-top: 1.5rem;
}

.result-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: #7a8499;
    text-transform: uppercase;
    margin-bottom: 1rem;
    text-align: center;
}

.team-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.75rem;
}

.team-name {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #e8eaf0;
    min-width: 160px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.bar-track {
    flex: 1;
    height: 10px;
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    overflow: hidden;
}

.bar-fill-win {
    height: 100%;
    background: linear-gradient(90deg, #ff8c00, #ffcc00);
    border-radius: 999px;
    transition: width 0.6s ease;
}

.bar-fill-loss {
    height: 100%;
    background: rgba(255,255,255,0.15);
    border-radius: 999px;
}

.pct-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    min-width: 3.5rem;
    text-align: right;
}

.pct-win  { color: #ff8c00; }
.pct-loss { color: #7a8499; }

/* ── Error box ── */
div[data-testid="stAlert"] {
    background: rgba(220, 53, 69, 0.1) !important;
    border: 1px solid rgba(220,53,69,0.3) !important;
    border-radius: 8px !important;
    color: #f8838e !important;
}

/* divider */
hr { border-color: rgba(255,255,255,0.06); }
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">🏏 Live Match Analysis</div>
    <h1 class="hero-title">IPL Win <span>Predictor</span></h1>
    <p class="hero-sub">Enter mid-match stats to calculate win probability in real time</p>
</div>
""", unsafe_allow_html=True)


df=pd.read_csv('a.csv')
@st.cache_resource
def train_model():
    ipl =pd.read_csv('a.csv')

    ipl.drop(columns=['Unnamed: 0'],inplace=True)

    x =ipl.drop(columns=['result'])
    y=ipl['result']

    xtr,xte,ytr,yte =train_test_split(x,y,test_size=0.2,random_state=2)

    ohe =OneHotEncoder()

    ohe.fit(x[['batting_team','bowling_team','venue']])

    trf = ColumnTransformer([
        ('trf', OneHotEncoder(sparse_output=False, drop='first',categories=ohe.categories_), ['batting_team', 'bowling_team', 'venue'])
        ,('trf2',StandardScaler(),['runs_left','balls_left','wickets_left','runs_target_y','crr','rrr'])
    ]
        , remainder='passthrough')

    # trf2=ColumnTransformer(transformers=[('trf2',StandardScaler(),['runs_left','balls_left','wickets_left','runs_target_y','crr','rrr'])],remainder='passthrough')

    pipe2 = Pipeline(steps=[
        ('step1', trf),

        ('step3', LogisticRegression(solver='liblinear'))
    ])

    pipe2.fit(xtr,ytr)

    print(pipe2.predict(xte),yte)

    return pipe2

pipe=train_model()

####UI

teams=df['batting_team'].unique()

venue=df['venue'].unique()

col1,col2 =st.columns(2)

with col1:
    batting_team=st.selectbox('select batting team',sorted(teams))
with col2:
    bowling_team=st.selectbox('select bowling team',sorted(teams))



select_venue =st.selectbox('select venue',sorted(venue))

target = st.number_input('Target',0,300)

col3,col4,col5 =st.columns(3)

with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs completed',0,20)
with col5:
    wickets = st.number_input('Wickets out')
# if batting_team!=bowling_team :
#
#
#     runs_left = target - score
#     wickets_left = 10 - wickets
#     balls_left = 120 - (overs * 6)
#     crr = score / overs if overs > 0 else 0
#     rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
#



if st.button('predict probability'):
    if batting_team==bowling_team:
        st.error("batting team and bowling team can not be equal")

    else:
        if target !=0:

            if score<= target:
                
                
                runs_left = target - score
                wickets_left = 10 - wickets
                balls_left = 120 - (overs * 6)
                crr = score / overs if overs > 0 else 0
                rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
                

                inp = pd.DataFrame({'batting_team': [batting_team],
                                'bowling_team': [bowling_team],
                                'venue': [select_venue],
                                'runs_left': [runs_left],
                                'wickets_left': [wickets_left],
                                'balls_left': [balls_left],
                                'runs_target_y': [target],
                                'crr': [crr],
                                'rrr': [rrr]

                                })

                 result = pipe.predict_proba(inp)
                
                 loss = result[0][0]
                 win = result[0][1]
                 st.subheader(f"current run rate is {np.round(crr, 1)}")
                
                 st.subheader(f"required run rate is {np.round(rrr, 1)}")
                

                 st.header(batting_team + "- " + str(round(win * 100)) + "%")
                 st.header(bowling_team + "- " + str(round(loss * 100)) + "%")

        

        

        
                

            # st.header(f"current run rate is {crr}")
            # st.header(f"required run rate is {rrr}")


                  
                
                               

          
                
 
            
         

             
            
        else:
            st.error("score can not be greater than target")

        else:
            st.error("please enter target")




