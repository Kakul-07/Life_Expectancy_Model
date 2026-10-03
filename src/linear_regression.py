from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import numpy as np
from preprocessing import prepare_regression_data
def train_linear_model():
    X_train,X_test,y_train,y_test,preprocessor=prepare_regression_data()
    model=Pipeline([("preprocessor",preprocessor),("model",LinearRegression())])
    model.fit(X_train,y_train)
    predictions=model.predict(X_test)
    mae=mean_absolute_error(y_test,predictions)
    rmse=np.sqrt(mean_squared_error(y_test,predictions))
    r2=r2_score(y_test,predictions)
    print("\nLINEAR REGRESSION")
    print("MAE:",round(mae,4))
    print("RMSE:",round(rmse,4))
    print("R2:",round(r2,4))
    return model