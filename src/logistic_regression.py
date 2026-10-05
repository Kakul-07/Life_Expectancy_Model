from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from preprocessing import prepare_classification_data
def train_logistic_model():
    X_train,X_test,y_train,y_test,preprocessor=prepare_classification_data()
    model=Pipeline([("preprocessor",preprocessor), ("model",LogisticRegression(max_iter=2000))])
    model.fit(X_train,y_train)
    predictions=model.predict(X_test)
    print("\nLOGISTIC REGRESSION")
    print("Accuracy:",round(model.score(X_test,y_test)*100,2),"%")
    return model,X_test,y_test,predictions