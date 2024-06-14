import sys
import os
from src.exception import CustomException
from src.logger import logging
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from dataclasses import dataclass
from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class Modeltrainer_config:
    model_trainer_path=os.path.join("artifacts","trainer.pkl")

class ModelTrainer:
    def __init__(self):
        self.modeltrainer_config=Modeltrainer_config()
    
    def inititate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("SPLITTING TRAIN ANAD TEST SPLIT")
            X_train,Y_train,X_test,Y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            logging.info("EVALUATING MODELS")
            model_report:dict=evaluate_models( X_train,Y_train,X_test,Y_test,models)

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("BAD MODEL")
            logging.info("BEST MODEL FOUND")

            logging.info("SAVING THE BEST MODEL")
            save_object(
                file_path=self.modeltrainer_config.model_trainer_path,
                obj=best_model
            )

            predict=best_model.predict(X_test)
            r2=r2_score(Y_test,predict)

            return (r2,best_model)
        except Exception as e:
            raise CustomException(e,sys) 