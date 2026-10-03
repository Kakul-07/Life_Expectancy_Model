import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,classification_report,confusion_matrix
import numpy as np

st.set_page_config(
    page_title="Life Expectancy Prediction",
    page_icon="📊",
    layout="wide"
)

PROJECT_DIR=os.path.dirname(os.path.abspath(__file__))
DATA_PATH=os.path.join(PROJECT_DIR,"Dataset","Life Expectancy Data.csv")

selected_features=[
    "HIV/AIDS",
    "Adult Mortality",
    "Income composition of resources",
    "BMI",
    "under-five deaths",
    "thinness 5-9 years",
    "Year",
    "Schooling",
    "Country",
    "Alcohol"
]

@st.cache_data
def load_data():
    df=pd.read_csv(DATA_PATH)
    df.columns=df.columns.str.strip()
    df=df.dropna(subset=["Life expectancy"]).copy()
    return df

@st.cache_resource
def train_models():
    df=load_data()

    X=df[selected_features]
    y_regression=df["Life expectancy"]
    y_classification=(df["Life expectancy"]>=65).astype(int)

    X_train_reg,X_test_reg,y_train_reg,y_test_reg=train_test_split(
        X,
        y_regression,
        test_size=0.20,
        random_state=42
    )

    X_train_cls,X_test_cls,y_train_cls,y_test_cls=train_test_split(
        X,
        y_classification,
        test_size=0.20,
        random_state=42,
        stratify=y_classification
    )

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

    linear_model=Pipeline([
        ("preprocessor",preprocessor),
        ("model",LinearRegression())
    ])

    logistic_model=Pipeline([
        ("preprocessor",preprocessor),
        ("model",LogisticRegression(max_iter=2000))
    ])

    linear_model.fit(X_train_reg,y_train_reg)
    logistic_model.fit(X_train_cls,y_train_cls)

    regression_predictions=linear_model.predict(X_test_reg)
    classification_predictions=logistic_model.predict(X_test_cls)

    mae=mean_absolute_error(y_test_reg,regression_predictions)
    rmse=np.sqrt(mean_squared_error(y_test_reg,regression_predictions))
    r2=r2_score(y_test_reg,regression_predictions)

    accuracy=accuracy_score(
        y_test_cls,
        classification_predictions
    )

    report=classification_report(
        y_test_cls,
        classification_predictions,
        target_names=["Below 65","65 or above"],
        output_dict=True
    )

    matrix=confusion_matrix(
        y_test_cls,
        classification_predictions
    )

    return (
        linear_model,
        logistic_model,
        mae,
        rmse,
        r2,
        accuracy,
        report,
        matrix
    )

df=load_data()

linear_model,logistic_model,mae,rmse,r2,accuracy,report,matrix=train_models()

st.title("Life Expectancy Prediction")
st.write("Predict life expectancy using important health, education and demographic features.")

st.sidebar.header("Project Information")
st.sidebar.write("Dataset: Life Expectancy Data")
st.sidebar.write("Models: Linear Regression and Logistic Regression")
st.sidebar.write("Selected Features: 10")

tab1,tab2,tab3=st.tabs([
    "Prediction",
    "Model Performance",
    "Dataset"
])

with tab1:
    st.header("Predict Life Expectancy")

    st.write("The prediction uses the 10 features selected using permutation importance.")

    col1,col2=st.columns(2)

    with col1:
        country=st.selectbox(
            "Country",
            sorted(df["Country"].dropna().unique())
        )

        hiv_aids=st.number_input(
            "HIV/AIDS",
            min_value=0.0,
            value=0.0
        )

        adult_mortality=st.number_input(
            "Adult Mortality",
            min_value=0.0,
            value=0.0
        )

        income_composition=st.number_input(
            "Income composition of resources",
            min_value=0.0,
            value=0.0
        )

        bmi=st.number_input(
            "BMI",
            min_value=0.0,
            value=0.0
        )

    with col2:
        under_five_deaths=st.number_input(
            "under-five deaths",
            min_value=0.0,
            value=0.0
        )

        thinness_5_9=st.number_input(
            "thinness 5-9 years",
            min_value=0.0,
            value=0.0
        )

        year=st.number_input(
            "Year",
            min_value=2000,
            max_value=2020,
            value=2015,
            step=1
        )

        schooling=st.number_input(
            "Schooling",
            min_value=0.0,
            value=0.0
        )

        alcohol=st.number_input(
            "Alcohol",
            min_value=0.0,
            value=0.0
        )

    if st.button("Predict Life Expectancy",type="primary"):
        input_data=pd.DataFrame([{
            "HIV/AIDS":hiv_aids,
            "Adult Mortality":adult_mortality,
            "Income composition of resources":income_composition,
            "BMI":bmi,
            "under-five deaths":under_five_deaths,
            "thinness 5-9 years":thinness_5_9,
            "Year":year,
            "Schooling":schooling,
            "Country":country,
            "Alcohol":alcohol
        }])

        predicted_value=linear_model.predict(input_data)[0]
        predicted_class=logistic_model.predict(input_data)[0]

        st.success(
            f"Predicted Life Expectancy: {predicted_value:.2f} years"
        )

        if predicted_class==1:
            st.info("Classification: 65 years or above")
        else:
            st.info("Classification: Below 65 years")

with tab2:
    st.header("Model Performance")

    st.subheader("Linear Regression")

    col1,col2,col3=st.columns(3)

    with col1:
        st.metric("MAE",f"{mae:.4f}")

    with col2:
        st.metric("RMSE",f"{rmse:.4f}")

    with col3:
        st.metric("R²",f"{r2:.4f}")

    st.subheader("Logistic Regression")

    st.metric(
        "Accuracy",
        f"{accuracy*100:.2f}%"
    )

    report_df=pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.subheader("Confusion Matrix")

    fig,ax=plt.subplots(figsize=(6,5))

    ax.imshow(matrix)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Logistic Regression - Confusion Matrix")

    ax.set_xticks([0,1])
    ax.set_xticklabels(["Below 65","65 or above"])

    ax.set_yticks([0,1])
    ax.set_yticklabels(["Below 65","65 or above"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                matrix[i,j],
                ha="center",
                va="center"
            )

    st.pyplot(fig)

with tab3:
    st.header("Dataset")

    st.write(
        f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns."
    )

    st.dataframe(df)

    st.subheader("Selected Features")

    st.write(
        "The following 10 features are used by the final models:"
    )

    for feature in selected_features:
        st.write(f"- {feature}")