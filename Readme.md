# Life Expectancy Prediction using Machine Learning

## About the Project

This project is based on the **Life Expectancy Data** dataset. The main aim of the project is to use machine learning to predict life expectancy and also classify countries into two life expectancy groups.

I have used two machine learning algorithms in this project:

- Linear Regression
- Logistic Regression

Linear Regression is used to predict the actual life expectancy value, while Logistic Regression is used to classify whether the life expectancy is below 65 or 65 and above.

The project also includes data analysis, preprocessing, feature selection, model evaluation and a Streamlit web application.

---

## Dataset

The dataset used in this project is:

`Life Expectancy Data.csv`

It contains information about different countries over different years. The dataset includes health, economic, education and demographic-related columns.

The dataset has:

- 2938 rows
- 22 columns

The main target column is:

`Life expectancy`

The original dataset is kept unchanged in the project.

---

## What I Did in This Project

The project was completed in the following steps:

1. Loaded the dataset
2. Explored the data using a Jupyter Notebook
3. Checked missing values and duplicates
4. Studied the distributions and relationships between variables
5. Analyzed feature importance
6. Selected important features
7. Preprocessed the data
8. Split the data into training and testing sets
9. Trained Linear Regression
10. Trained Logistic Regression
11. Evaluated both models
12. Created a Streamlit application for prediction

---

## Exploratory Data Analysis

The EDA notebook is used to understand the dataset before training the models.

Some of the things checked during EDA were:

- Dataset shape
- Column names
- Data types
- Missing values
- Duplicate rows
- Statistical summary
- Unique countries
- Years
- Country status
- Zero values
- Outliers
- Feature distributions
- Correlations

Some relationships were also checked, such as:

- Schooling and life expectancy
- Adult mortality and life expectancy
- Health-related variables and life expectancy
- Economic variables and life expectancy

The notebook also contains some exploratory feature engineering. These features were created for analysis and are not used as inputs in the final models.

---

## Feature Selection

I did not select the final features randomly.

Feature importance was checked using **Random Forest with permutation importance**. The features were ranked according to how much the model's performance changed when a feature was shuffled.

The final selected features are:

```text
FEATURES=[
    "HIV/AIDS",
    "Schooling",
    "Adult Mortality",
    "Income composition of resources",
    "under-five deaths",
    "infant deaths",
    "BMI",
    "GDP",
    "Diphtheria",
    "thinness 1-19 years"
]
These features were selected based on the permutation-importance analysis performed in this project. The feature importance values can depend on the machine-learning model and the data used for the analysis, so these features are considered important for this project rather than universally important factors.

The original dataset is not modified. The selected features are chosen in the Python code and are used by the final machine-learning models.

---

## Data Preprocessing

The selected features contain numerical as well as categorical data, so different preprocessing methods are used.

### Numerical Features

Missing numerical values are filled using the median.

The numerical features are then standardized using `StandardScaler`.

### Categorical Features

The `Country` column is a categorical feature.

Missing categorical values are filled using the most frequent value and then converted into numerical form using `OneHotEncoder`.

The preprocessing is implemented using a Scikit-learn pipeline so that the same preprocessing steps are applied during training and prediction.

---

## Train-Test Split

The dataset is divided into:

- 80% training data
- 20% testing data

The project uses `random_state=42` so that the same split can be reproduced.

For Logistic Regression, stratified splitting is used to maintain the proportion of the two classes.

---

## Linear Regression

Linear Regression is used to predict the actual life expectancy value.

For example:

```text
Predicted Life Expectancy = 68.4 years