import pandas as pd
import numpy as np
import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file=os.path.join("artifacts",'preprocessor.pkl')

class Data_transformation:
    def __init__(self):
        self.data_transformation_obj=DataTransformationConfig()
    
    def get_data_transformation_obj(self):
        num_features=["writing_score","reading_score"]
        cat_features=[
            "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
        ]
        num_pipeline=Pipeline(steps=[
            ("Imputer",SimpleImputer(strategy="median")),
            ("scaling",StandardScaler())
        ])
        cat_pipeline=Pipeline(steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("Onehot",OneHotEncoder())
        ])
        logging.info(f"Categorical columns: {cat_features}")
        logging.info(f"Numerical columns: {num_features}")

        preprocessor=ColumnTransformer([("num",num_pipeline,num_features),
                                        ("cat",cat_pipeline,cat_features)])
        
        return preprocessor
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("READIN TRAIN AND TEST DATA COMPLETED")

            logging.info("OBTAINING PREPROCESSING OBJECT")
            preprocessor_obj=self.get_data_transformation_obj()
            
            target="math_score"
            input_train_feature=train_df.drop(columns=target)
            target_train_feature=train_df[target]

            input_test_feature=test_df.drop(columns=target)
            target_test_feature=test_df[target]

            logging.info("APPLYING PREPROCESSING OBJECT ON TRAINA AND TEST DATA")
            input_train_feature_arr=preprocessor_obj.fit_transform(input_train_feature)

            input_test_feature_arr=preprocessor_obj.transform(input_test_feature)


            train_arr=np.c_[input_train_feature_arr,np.array(target_train_feature)]
            test_arr=np.c_[input_test_feature_arr,np.array(target_test_feature)]

            logging.info(f"SAVING PREPROCESSING OBJECT.")
            
            save_object(
                file_path=self.data_transformation_obj.preprocessor_obj_file,
                obj=preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_obj.preprocessor_obj_file
            )
        except Exception as e:
            raise CustomException(e,sys)


