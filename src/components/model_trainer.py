import os 
import sys
from dataclasses import dataclass 

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor

from exception import CustomException
from logger import logging

from utils import save_object,evaluate_model
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self) :
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and testing input data.")
            X_train,Y_train,X_test,Y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "LinearRegression": LinearRegression(),
                "K-Neighbours Classifier": KNeighborsRegressor(),
                "XGBClassifier":XGBRegressor(),
                "AdaboostClassifier": AdaBoostRegressor(),
            }

            model_report=evaluate_model(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,models=models)
            #To get best model scores from dict
            best_model_score=max(sorted(model_report.values()))
            #To get the best model name
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("No best model found")
            logging.info("Best moel is found on both training and testing data.")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2score=r2_score(Y_test,predicted)
            return r2score 
        except Exception as e:
            raise CustomException(e,sys)

