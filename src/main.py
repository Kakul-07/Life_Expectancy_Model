from linear_regression import train_linear_model
from logistic_regression import train_logistic_model
from evaluate import evaluate_logistic_model
print("\nLIFE EXPECTANCY ML")
linear_model=train_linear_model()
logistic_model,X_test,y_test,predictions=train_logistic_model()
accuracy=evaluate_logistic_model(y_test,predictions)
print("\nMODEL INFORMATION")
print("Linear Regression: Continuous prediction")
print("Logistic Regression: Classification")
print("Linear Regression metrics: MAE, RMSE, R2")
print("Logistic Regression metrics: Accuracy, Precision, Recall, F1-score")