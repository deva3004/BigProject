import os
import sys
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
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


def evaluate_models(X_train, y_train, X_test, y_test, models, parameters):
    try:

        report = {}

        for model_name in models:

            model = models[model_name]
            param = parameters[model_name]

            gs = GridSearchCV(
                model,
                param_grid=param,
                cv=3,
                scoring="r2",
                n_jobs=-1
            )

            gs.fit(X_train, y_train)

            # Update model with best parameters
            model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

            print(f"\n{'='*50}")
            print(f"Model: {model_name}")
            print(f"Best Parameters: {gs.best_params_}")
            print(f"Train R2 Score: {train_model_score:.4f}")
            print(f"Test R2 Score: {test_model_score:.4f}")
            print(f"{'='*50}")

        return report

    except Exception as e:
        raise CustomException(e, sys)