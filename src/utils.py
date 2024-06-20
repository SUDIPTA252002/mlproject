import os 
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score

def save_object(file_path,obj):
    dir_name=os.path.dirname(file_path)
    os.makedirs(dir_name,exist_ok=True)
    with open(file_path,"wb") as file_obj:
        dill.dump(obj,file_obj)

def evaluate_models( X_train,Y_train,X_test,Y_test,models,param):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            paras=list(param.values())[i]
            gs=GridSearchCV(model,paras,cv=3)
            gs.fit(X_train,Y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,Y_train)

            train_y_pred=model.predict(X_train)
            test_y_pred=model.predict(X_test)

            model_score=r2_score(Y_test,test_y_pred)
            report[list(models.keys())[i]]=model_score
            return report
    
    except Exception as e:
        raise CustomException(e,sys)


