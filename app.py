import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,classification_report,confusion_matrix

st.set_page_config(page_title="Life Expectancy ML",page_icon="📊",layout="wide")

PROJECT_DIR=os.path.dirname(os.path.abspath(__file__))
DATA_PATH=os.path.join(PROJECT_DIR,"dataset","Life Expectancy Data.csv")

@st.cache_data
def load_data():
    df=pd.read_csv(DATA_PATH)
    df.columns=df.columns.str.strip()
    return df

@st.cache_resource
def train_models():
    df=load_data()
    df=df.dropna(subset=["Life expectancy"]).copy()
    selected_features=[
    "Country",
    "Year",
    "Status",
    "Adult Mortality",
    "Alcohol",
    "BMI",
    "HIV/AIDS",
    "GDP",
    "Income composition of resources",
    "Schooling"
]

    X=df[selected_features]
    y=df["Life expectancy"]
    numeric_features=X.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_features=X.select_dtypes(include=["object"]).columns.tolist()
    numeric_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])
    categorical_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor=ColumnTransformer([
        ("numeric",numeric_pipeline,numeric_features),
        ("categorical",categorical_pipeline,categorical_features)
    ])
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)
    linear_model=Pipeline([
        ("preprocessor",preprocessor),
        ("model",LinearRegression())
    ])
    linear_model.fit(X_train,y_train)
    linear_predictions=linear_model.predict(X_test)
    linear_mae=mean_absolute_error(y_test,linear_predictions)
    linear_rmse=np.sqrt(mean_squared_error(y_test,linear_predictions))
    linear_r2=r2_score(y_test,linear_predictions)
    y_class=(y>=65).astype(int)
    X_train_c,X_test_c,y_train_c,y_test_c=train_test_split(X,y_class,test_size=0.20,random_state=42,stratify=y_class)
    logistic_model=Pipeline([
        ("preprocessor",preprocessor),
        ("model",LogisticRegression(max_iter=2000))
    ])
    logistic_model.fit(X_train_c,y_train_c)
    logistic_predictions=logistic_model.predict(X_test_c)
    logistic_accuracy=accuracy_score(y_test_c,logistic_predictions)
    report=classification_report(y_test_c,logistic_predictions,target_names=["Below 65","65 or above"],output_dict=True)
    matrix=confusion_matrix(y_test_c,logistic_predictions)
    return linear_model,logistic_model,linear_mae,linear_rmse,linear_r2,logistic_accuracy,report,matrix,X.columns.tolist()

df=load_data()
linear_model,logistic_model,linear_mae,linear_rmse,linear_r2,logistic_accuracy,report,matrix,features=train_models()

st.markdown("""
<style>
.main-title{font-size:42px;font-weight:700}
.subtitle{font-size:18px;color:#64748b}
.block{padding:18px;border-radius:12px;border:1px solid #e2e8f0;background:#f8fafc}
</style>
""",unsafe_allow_html=True)

st.markdown('<div class="main-title">Life Expectancy ML Dashboard</div>',unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning Analysis using Linear and Logistic Regression</div>',unsafe_allow_html=True)

c1,c2,c3,c4=st.columns(4)
c1.metric("Dataset Rows",len(df))
c2.metric("Features",len(features))
c3.metric("Linear R²",f"{linear_r2:.4f}")
c4.metric("Logistic Accuracy",f"{logistic_accuracy*100:.2f}%")

st.divider()

tab1,tab2,tab3=st.tabs(["Dataset","Prediction","Model Performance"])

with tab1:
    st.subheader("Dataset")
    st.dataframe(df,use_container_width=True)
    st.subheader("Dataset Statistics")
    st.dataframe(df.describe(include="all").transpose(),use_container_width=True)

with tab2:
    st.subheader("Predict Life Expectancy")
    input_data={}
    left,right=st.columns(2)

    with left:
        input_data["Country"]=st.selectbox("Country",sorted(df["Country"].dropna().unique()))
        input_data["Year"]=st.number_input("Year",int(df["Year"].min()),int(df["Year"].max()),int(df["Year"].max()))
        input_data["Status"]=st.selectbox("Status",sorted(df["Status"].dropna().unique()))
        input_data["Adult Mortality"]=st.number_input("Adult Mortality",0.0,float(df["Adult Mortality"].median()))
        input_data["infant deaths"]=st.number_input("Infant Deaths",0,int(df["infant deaths"].median()))
        input_data["Alcohol"]=st.number_input("Alcohol",0.0,float(df["Alcohol"].median()))
        input_data["percentage expenditure"]=st.number_input("Percentage Expenditure",0.0,float(df["percentage expenditure"].median()))
        input_data["Hepatitis B"]=st.number_input("Hepatitis B",0.0,100.0,float(df["Hepatitis B"].median()))
        input_data["Measles"]=st.number_input("Measles",0,int(df["Measles"].median()))
        input_data["BMI"]=st.number_input("BMI",0.0,float(df["BMI"].median()))
        input_data["under-five deaths"]=st.number_input("Under-Five Deaths",0,int(df["under-five deaths"].median()))

    with right:
        input_data["Polio"]=st.number_input("Polio",0.0,100.0,float(df["Polio"].median()))
        input_data["Total expenditure"]=st.number_input("Total Expenditure",0.0,float(df["Total expenditure"].median()))
        input_data["Diphtheria"]=st.number_input("Diphtheria",0.0,100.0,float(df["Diphtheria"].median()))
        input_data["HIV/AIDS"]=st.number_input("HIV/AIDS",0.0,float(df["HIV/AIDS"].median()))
        input_data["GDP"]=st.number_input("GDP",0.0,float(df["GDP"].median()))
        input_data["Population"]=st.number_input("Population",0.0,float(df["Population"].median()))
        input_data["thinness  1-19 years"]=st.number_input("Thinness 1-19 Years",0.0,float(df["thinness  1-19 years"].median()))
        input_data["thinness 5-9 years"]=st.number_input("Thinness 5-9 Years",0.0,float(df["thinness 5-9 years"].median()))
        input_data["Income composition of resources"]=st.number_input("Income Composition",0.0,1.0,float(df["Income composition of resources"].median()))
        input_data["Schooling"]=st.number_input("Schooling",0.0,float(df["Schooling"].median()))

    if st.button("Predict Life Expectancy",use_container_width=True):
        input_df=pd.DataFrame([input_data],columns=features)
        linear_prediction=linear_model.predict(input_df)[0]
        logistic_prediction=logistic_model.predict(input_df)[0]
        st.success(f"Predicted Life Expectancy: {linear_prediction:.2f} years")
        if logistic_prediction==1:
            st.info("Classification: 65 years or above")
        else:
            st.warning("Classification: Below 65 years")

with tab3:
    st.subheader("Linear Regression")
    c1,c2,c3=st.columns(3)
    c1.metric("MAE",f"{linear_mae:.4f}")
    c2.metric("RMSE",f"{linear_rmse:.4f}")
    c3.metric("R²",f"{linear_r2:.4f}")
    st.divider()
    st.subheader("Logistic Regression")
    st.metric("Accuracy",f"{logistic_accuracy*100:.2f}%")
    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(report).transpose(),use_container_width=True)
    st.subheader("Confusion Matrix")
    cm_df=pd.DataFrame(matrix,index=["Actual: Below 65","Actual: 65 or above"],columns=["Predicted: Below 65","Predicted: 65 or above"])
    st.dataframe(cm_df,use_container_width=True)