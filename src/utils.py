import os
import sys
import pickle

from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Train and evaluate multiple models.

    Returns:
        report: dict containing model name and R2 score
    """
    try:
        report = {}

        for model_name, model in models.items():

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

            print(
                f"{model_name} -> "
                f"Train R2: {train_model_score:.4f}, "
                f"Test R2: {test_model_score:.4f}"
            )

        return report

    except Exception as e:
        raise CustomException(e, sys)