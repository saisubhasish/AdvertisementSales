import os,sys 
import numpy as np
from dask import dataframe as dd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from typing import Optional

from advertisement import utils
from advertisement.entity import artifact_entity,config_entity
from advertisement.exception import AdvertisementException
from advertisement.logger import logging
from advertisement.config import TARGET_COLUMN
from advertisement.config import FEATURE_COLUMN



class DataTransformation:

    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,
                    data_validation_artifact:artifact_entity.DataValidationArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config=data_transformation_config
            self.data_validation_artifact=data_validation_artifact
        except Exception as e:
            raise AdvertisementException(e, sys)

    @classmethod
    def get_transformer_object(cls)->Pipeline:     # Attributes of this class will be same across all the object 
        try:
            scaler = StandardScaler()
            pipeline = Pipeline(steps=[
                    ('scaler', scaler)  
                ])
            return pipeline

        except Exception as e:
            raise AdvertisementException(e, sys)
    

    def initiate_data_transformation(self,) -> artifact_entity.DataTransformationArtifact:
        try:
            # Reading training and testing file
            logging.info("Reading training and testing file")
            train_df = dd.read_csv(self.data_validation_artifact.train_file_path).head(n=200)
            test_df = dd.read_csv(self.data_validation_artifact.test_file_path).head(n=200)
            
            # Selecting input feature for train and test dataframe
            logging.info("Selecting input feature for train and test dataframe")
            input_feature_train_df=train_df[[FEATURE_COLUMN]]
            input_feature_test_df=test_df[[FEATURE_COLUMN]]

            # Selecting target feature for train and test dataframe
            logging.info("Selecting target feature for train and test dataframe")
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            # Imputing null values with KNNImputer
            transformation_pipeline = DataTransformation.get_transformer_object()
            transformation_pipeline.fit(input_feature_train_df)
            logging.info(input_feature_train_df.columns)
            
            # Imputing null values
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)  
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)        

            # Target encoder
            train_arr = np.c_[input_feature_train_arr, target_feature_train_df]    # concatenated transpose array
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df]

            # Save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path, array=train_arr)
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path, array=test_arr)

            # Saving object
            utils.save_object(file_path=self.data_transformation_config.transformer_object_path, obj=transformation_pipeline)

            # Preparing Artifact
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transformer_object_path=self.data_transformation_config.transformer_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path)

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact
            
        except Exception as e:
            raise AdvertisementException(e, sys)
