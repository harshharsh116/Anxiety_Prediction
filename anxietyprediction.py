import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns   
import plotly.express as px
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
import streamlit as st
import io
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

st.title("🧠 Anxiety Prediction System")
@st.cache_data
def load_data():
    df = pd.read_csv('enhanced_anxiety.csv')
    return df
df = load_data()
data = st.sidebar.checkbox('ORIGINAL DATA_FRAME')
if data:
    st.dataframe(df)
    st.write('Shape of dataframe',df.shape)
numeric_columns = ['Age','Sleep Hours','Physical Activity (hrs/week)','Caffeine Intake (mg/day)','Alcohol Consumption (drinks/week)','Stress Level (1-10)','Heart Rate (bpm)','Breathing Rate (breaths/min)','Sweating Level (1-5)','Therapy Sessions (per month)','Diet Quality (1-10)','Anxiety Level (1-10)']    
categorocal_columns = ['Gender','Occupation','Smoking','Family History of Anxiety','Dizziness','Medication','Recent Major Life Event']
eda = st.sidebar.checkbox('EDA AND DATA PROCESSING')
if eda:
    st.markdown(
    "<h3 style='text-align: center;'>EDA and DATA PREPROCESSING</h3>",
    unsafe_allow_html=True
    )

    st.markdown(
    "<h4 style='text-align: center;'>First 5 records</h4>",
    unsafe_allow_html=True
    )
    st.write(df.head(5))
    st.markdown(
    "<h4 style='text-align: center;'>Last 5 records</h4>",
    unsafe_allow_html=True
    )

    st.write(df.tail(5))
    st.markdown(
    "<h4 style='text-align: center;'>Descriptive Analysis</h4>",
    unsafe_allow_html=True
    )
    st.write(df.describe())
    
    st.markdown(
    "<h4 style='text-align: center;'>Column Information</h4>",
    unsafe_allow_html=True
    )
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.code(buffer.getvalue())

    st.markdown(
    "<h4 style='text-align: center;'>Null check</h4>",
    unsafe_allow_html=True
    ) 
    st.write(df.isnull().sum())  
    st.markdown(
    "<h4 style='text-align: center;'>Shape After Removing Duplicated Values</h4>",
    unsafe_allow_html=True
    ) 
    df = df.drop_duplicates()
    st.write(df.shape)

    df = df[df['Sleep Hours']>2]
    
    for i in numeric_columns:
        fig =  px.histogram(
            df,
            x=i,
            nbins=10,
        )
        fig.update_traces(
        marker_line_width=2,
        marker_line_color='black'
        )
        st.plotly_chart(fig)
    
    corr = df.corr(numeric_only=True)

    fig = px.imshow(
    corr,
    text_auto='.2f',
    aspect='auto'
    )

    st.plotly_chart(fig)
       
    import plotly.express as px

    fig = px.pie(
    names=df['Gender'],
    title='Gender Distribution'
    )

    st.plotly_chart(fig)
    
    fig = px.pie(
    names=df['Occupation'],
    title='Occupation Distribution'
    )

    st.plotly_chart(fig)
    
    fig = px.pie(
    names=df['Smoking'],
    title='Smoking Distribution'
    )

    st.plotly_chart(fig)
    
    fig = px.pie(
    names=df['Family History of Anxiety'],
    title='Family History of Anxiety Distribution',
    )
    st.plotly_chart(fig)


    for i in numeric_columns:
        fig = px.box(
            df,
            x=df[i]
        )
        st.plotly_chart(fig)

    fig = px.histogram(
        df,
        x='Medication',
        title='Medication Distribution'
    )

    st.plotly_chart(fig)

    fig = px.histogram(
        df,
        x='Dizziness',
        title='Dizziness Distribution'
    )

    st.plotly_chart(fig)

    family = df.groupby('Family History of Anxiety')['Anxiety Level (1-10)'].mean().reset_index()

    fig = px.bar(
        family,
        x='Family History of Anxiety',
        y='Anxiety Level (1-10)',
        title='Average Anxiety by Family History'
    )

    st.plotly_chart(fig)

    medication = df.groupby('Medication')['Anxiety Level (1-10)'].mean().reset_index()

    fig = px.bar(
        medication,
        x='Medication',
        y='Anxiety Level (1-10)',
        title='Average Anxiety by Medication'
    )

    st.plotly_chart(fig)

    life_event = df.groupby('Recent Major Life Event')['Anxiety Level (1-10)'].mean().reset_index()
    fig = px.bar(
        life_event,
        x='Recent Major Life Event',
        y='Anxiety Level (1-10)',
        title='Average Anxiety by Recent Major Life Event'
    )
    st.plotly_chart(fig)
df_before_encode  = df.copy()
encoders = {}

for col in categorocal_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

x = df.drop('Anxiety Level (1-10)',axis=1)
y = df['Anxiety Level (1-10)']


t = st.sidebar.number_input('Test Size',.1,.5)
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=t,random_state=42)
v3 = st.sidebar.checkbox('Train Test Shape')
if v3:
    st.write('xtrain',xtrain.shape)
    st.write('xtest',xtest.shape)
    st.write('ytrain',ytrain.shape)
    st.write('ytest',ytest.shape)

scalling = StandardScaler()
xtrainscale = scalling.fit_transform(xtrain)
xtestscale = scalling.transform(xtest)

if st.sidebar.button('Train Model xgboost'):
     xg = XGBRegressor(random_state=42)
     param_grid = {'n_estimators':[100,200,300],
                   'learning_rate': [0.1,.2,.3],
                   'max_depth': [3,4,5],
                   'min_child_weight': [1,2,3],
                  }
     model = GridSearchCV(estimator=xg, param_grid=param_grid,cv=5,n_jobs=-1,scoring='r2')

     model.fit(xtrainscale,ytrain)
     st.session_state['model'] = model
     ypred = model.predict(xtestscale)

     r2 = r2_score(ytest,ypred)
     mae = mean_absolute_error(ytest,ypred)
     st.write('XG MAE',mae)
     st.write('XG R2',r2)

if st.sidebar.button('Train Model Random Forest'):
    rf = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }

    model = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='r2'
    )

    model.fit(xtrainscale, ytrain)
    st.session_state['model'] = model
    ypred = model.predict(xtestscale)

    r2 = r2_score(ytest, ypred)
    mae = mean_absolute_error(ytest, ypred)

    st.write("Best Parameters:", model.best_params_)
    st.write("RF MAE:", mae)
    st.write("RF R2:", r2)


q = st.sidebar.checkbox('Prediction')
if q:
    age = st.number_input("Age", min_value=1, max_value=100)

    gender = st.selectbox("Gender", df_before_encode['Gender'].unique())

    occupation = st.selectbox("Occupation", df_before_encode['Occupation'].unique())

    sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0)

    physical_activity = st.number_input("Physical Activity (hrs/week)", min_value=0.0)

    caffeine = st.number_input("Caffeine Intake (mg/day)", min_value=0)

    alcohol = st.number_input("Alcohol Consumption (drinks/week)", min_value=0)

    smoking = st.selectbox("Smoking", df_before_encode['Smoking'].unique())

    family_history = st.selectbox("Family History of Anxiety", df_before_encode['Family History of Anxiety'].unique())

    stress = st.slider("Stress Level", 1, 10)

    heart_rate = st.number_input("Heart Rate (bpm)", min_value=30)

    breathing_rate = st.number_input("Breathing Rate (breaths/min)", min_value=5)

    sweating = st.slider("Sweating Level", 1, 5)

    dizziness = st.selectbox("Dizziness", df_before_encode['Dizziness'].unique())

    medication = st.selectbox("Medication", df_before_encode['Medication'].unique())

    therapy = st.number_input("Therapy Sessions (per month)", min_value=0)

    life_event = st.selectbox("Recent Major Life Event", df_before_encode['Recent Major Life Event'].unique())

    diet = st.slider("Diet Quality (1-10)", 1, 10)

    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Occupation': [occupation],
        'Sleep Hours': [sleep_hours],
        'Physical Activity (hrs/week)': [physical_activity],
        'Caffeine Intake (mg/day)': [caffeine],
        'Alcohol Consumption (drinks/week)': [alcohol],
        'Smoking': [smoking],
        'Family History of Anxiety': [family_history],
        'Stress Level (1-10)': [stress],
        'Heart Rate (bpm)': [heart_rate],
        'Breathing Rate (breaths/min)': [breathing_rate],
        'Sweating Level (1-5)': [sweating],
        'Dizziness': [dizziness],
        'Medication': [medication],
        'Therapy Sessions (per month)': [therapy],
        'Recent Major Life Event': [life_event],
        'Diet Quality (1-10)': [diet]
    })

    for col in categorocal_columns:
        input_data[col] = encoders[col].transform(input_data[col])

    input_scaled = scalling.transform(input_data)

    if st.button("Predict Anxiety Level"):

        if 'model' not in st.session_state:
            st.error("Please train a model first.")
        else:
            prediction = st.session_state['model'].predict(input_scaled)

            st.success(f"Predicted Anxiety Level: {round(prediction[0])}")