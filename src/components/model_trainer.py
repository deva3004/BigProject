import os
import sys

from dataclasses import dataclass

from sklearn.linear_model import Lasso, LinearRegression
LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):
        try:
            logging.info("Splitting training and test input data")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {
                    "Linear Regression": LinearRegression(),
                    "Lasso": Lasso(),
                    "Ridge": Ridge(),
                    "KNeighbors Regressor": KNeighborsRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Random Forest": RandomForestRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "XGBoost": XGBRegressor(),
                    "CatBoost": CatBoostRegressor(verbose=False),
                    "AdaBoost": AdaBoostRegressor()
            }
            params = {

                "Linear Regression": {
                    "fit_intercept": [True, False]
                },

                "Lasso": {
                    "alpha": [0.001, 0.01, 0.1, 1, 10]
                },

                "Ridge": {
                    "alpha": [0.001, 0.01, 0.1, 1, 10]
                },

                "KNeighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9, 11]
                },

                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse"],
                    "max_depth": [5, 10, 15, 20]
                },

                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20, None]
                },

                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1]
                },

                "XGBoost": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [100, 200]
                },

                "CatBoost": {
                    "depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.1]
                },

                "AdaBoost": {
                    "n_estimators": [50, 100, 200]
                }
            }

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                parameters=params
            )

            best_model_score = max(
                sorted(model_report.values())
            )

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            logging.info(
                f"Best Model Found: {best_model_name}"
            )

            logging.info(
                f"Best Model Score: {best_model_score}"
            )

            if best_model_score < 0.60:
                raise CustomException(
                    "No best model found with acceptable score",
                    sys
                )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Trained model saved successfully")

            predict_test = best_model.predict(X_test)
            r2_square = r2_score(y_test, predict_test)

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)