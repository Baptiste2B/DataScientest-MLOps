import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

with mlflow.start_run():
    model = LogisticRegression()
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)

    mlflow.log_param("model_type", "logistic_regression")
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")