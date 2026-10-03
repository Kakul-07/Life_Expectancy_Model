import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
def evaluate_logistic_model(y_test,predictions):
    accuracy=accuracy_score(y_test,predictions)
    report=classification_report(y_test,predictions,target_names=["Below 65","65 or above"],digits=4)
    matrix=confusion_matrix(y_test,predictions)
    print("\nACCURACY")
    print(round(accuracy*100,2),"%")
    print("\nCLASSIFICATION REPORT")
    print(report)
    print("\nCONFUSION MATRIX")
    print(matrix)
    plt.figure(figsize=(6,5))
    plt.imshow(matrix)
    plt.title("Logistic Regression - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.xticks([0,1],["Below 65","65 or above"])
    plt.yticks([0,1],["Below 65","65 or above"])
    for i in range(2):
        for j in range(2):
            plt.text(j,i,matrix[i,j],ha="center",va="center")
    plt.tight_layout()
    plt.show()
    return accuracy