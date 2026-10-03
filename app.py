import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,classification_report,confusion_matrix

PROJECT_DIR=os.path.dirname(os.path.abspath(__file__))
DATA_PATH=os.path.join(PROJECT_DIR,"Dataset","Life Expectancy Data.csv")

selected_features=["Country","Year","Status","Adult Mortality","Alcohol","BMI","HIV/AIDS","GDP","Income composition of resources","Schooling"]

@st.cache_data
def load_data():
    df=pd.read_csv(DATA_PATH)
    df.columns=df.columns.str.strip()
    return df

@st.cache_resource
def train_models():
    df=load_data()
    df=df.dropna(subset=["Life expectancy"]).copy()
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

    mae=mean_absolute_error(y_test,linear_predictions)
    rmse=np.sqrt(mean_squared_error(y_test,linear_predictions))
    r2=r2_score(y_test,linear_predictions)

    y_class=(y>=65).astype(int)

    X_train_c,X_test_c,y_train_c,y_test_c=train_test_split(
        X,y_class,test_size=0.20,random_state=42,stratify=y_class
    )

    logistic_preprocessor=ColumnTransformer([
        ("numeric",numeric_pipeline,numeric_features),
        ("categorical",categorical_pipeline,categorical_features)
    ])

    logistic_model=Pipeline([
        ("preprocessor",logistic_preprocessor),
        ("model",LogisticRegression(max_iter=2000))
    ])

    logistic_model.fit(X_train_c,y_train_c)
    logistic_predictions=logistic_model.predict(X_test_c)

    accuracy=accuracy_score(y_test_c,logistic_predictions)
    report=classification_report(
        y_test_c,
        logistic_predictions,
        target_names=["Below 65","65 or above"],
        output_dict=True
    )
    matrix=confusion_matrix(y_test_c,logistic_predictions)

    return linear_model,logistic_model,mae,rmse,r2,accuracy,report,matrix

st.set_page_config(page_title="Life Expectancy Prediction",page_icon="📊",layout="wide")

st.title("Life Expectancy Prediction")
st.write("Predict life expectancy using important health, economic and development features.")

df=load_data()
linear_model,logistic_model,mae,rmse,r2,accuracy,report,matrix=train_models()

tab1,tab2,tab3=st.tabs(["Dataset","Prediction","Model Performance"])

with tab1:
    st.subheader("Original Dataset")
    st.write("The original dataset contains all available features.")
    st.dataframe(df,use_container_width=True)
    st.write("Features used for prediction:")
    st.write(selected_features)

with tab2:
    st.subheader("Predict Life Expectancy")

    col1,col2=st.columns(2)

    with col1:
        country=st.selectbox("Country",sorted(df["Country"].dropna().unique()))
        year=st.number_input("Year",min_value=int(df["Year"].min()),max_value=int(df["Year"].max()),value=int(df["Year"].max()))
        status=st.selectbox("Status",sorted(df["Status"].dropna().unique()))
        adult_mortality=st.number_input("Adult Mortality",min_value=0.0,value=float(df["Adult Mortality"].median()))
        alcohol=st.number_input("Alcohol",min_value=0.0,value=float(df["Alcohol"].median()))
    
    with col2:
        bmi=st.number_input("BMI",min_value=0.0,value=float(df["BMI"].median()))
        hiv=st.number_input("HIV/AIDS",min_value=0.0,value=float(df["HIV/AIDS"].median()))
        gdp=st.number_input("GDP",min_value=0.0,value=float(df["GDP"].median()))
        income=st.number_input("Income composition of resources",min_value=0.0,max_value=1.0,value=float(df["Income composition of resources"].median()))
        schooling=st.number_input("Schooling",min_value=0.0,value=float(df["Schooling"].median()))

    if st.button("Predict Life Expectancy"):
        input_data=pd.DataFrame([{
            "Country":country,
            "Year":year,
            "Status":status,
            "Adult Mortality":adult_mortality,
            "Alcohol":alcohol,
            "BMI":bmi,
            "HIV/AIDS":hiv,
            "GDP":gdp,
            "Income composition of resources":income,
            "Schooling":schooling
        }])

        prediction=linear_model.predict(input_data)[0]
        class_prediction=logistic_model.predict(input_data)[0]

        st.success(f"Predicted Life Expectancy: {prediction:.2f} years")

        if class_prediction==1:
            st.info("Predicted category: 65 or above")
        else:
            st.info("Predicted category: Below 65")

with tab3:
    st.subheader("Linear Regression Performance")

    c1,c2,c3=st.columns(3)
    c1.metric("MAE",f"{mae:.4f}")
    c2.metric("RMSE",f"{rmse:.4f}")
    c3.metric("R² Score",f"{r2:.4f}")

    st.subheader("Logistic Regression Performance")
    st.metric("Accuracy",f"{accuracy:.2%}")

    st.subheader("Classification Report")
    report_df=pd.DataFrame(report).transpose()
    st.dataframe(report_df,use_container_width=True)

    st.subheader("Confusion Matrix")
    st.dataframe(
        pd.DataFrame(
            matrix,
            index=["Actual Below 65","Actual 65 or above"],
            columns=["Predicted Below 65","Predicted 65 or above"]
        ),
        use_container_width=True
    )