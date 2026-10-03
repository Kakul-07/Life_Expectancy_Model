import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

PROJECT_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH=os.path.join(PROJECT_DIR,"Dataset","Life Expectancy Data.csv")

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

def load_data():
    df=pd.read_csv(DATA_PATH)
    df.columns=df.columns.str.strip()
    df=df.dropna(subset=["Life expectancy"]).copy()
    return df
def create_preprocessor(X):
    numeric_features=X.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_features=X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline=Pipeline([("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])

    categorical_pipeline=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([("numeric",numeric_pipeline,numeric_features),
        ("categorical",categorical_pipeline,categorical_features)
    ])

def prepare_regression_data():
    df=load_data()
    X=df[selected_features]
    y=df["Life expectancy"]

    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.20,random_state=42
    )

    return X_train,X_test,y_train,y_test,create_preprocessor(X)

def prepare_classification_data():
    df=load_data()
    X=df[selected_features]
    y=(df["Life expectancy"]>=65).astype(int)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
    return X_train,X_test,y_train,y_test,create_preprocessor(X)