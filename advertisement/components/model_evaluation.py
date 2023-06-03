import os, sys
import pandas as pd
import numpy as np

from sklearn.metrics import r2_score

from advertisement import utils
from advertisement.logger import logging
from advertisement.utils import load_object
from advertisement.config import TARGET_COLUMN
from advertisement.predictor import ModelResolver
from advertisement.exception import AdvertisementException
from advertisement.entity import config_entity, artifact_entity
from advertisement.components.data_transformation import DataTransformation
 

class ModelEvaluation:

    def __init__(self,
        model_eval_config:config_entity.ModelEvaluationConfig,
        data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
        data_transformation_artifact:artifact_entity.DataTransformationArtifact,
        model_trainer_artifact:artifact_entity.ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*20}  Model Evaluation {'<<'*20}")
            logging.info("___________________________________________________________________________________________________________")
            self.model_eval_config=model_eval_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.data_transformation= DataTransformation(data_transformation_config=config_entity.DataTransformationConfig, data_validation_artifact=artifact_entity.DataValidationArtifact)
            self.model_resolver = ModelResolver()

        except Exception as e:
            raise AdvertisementException(e, sys)


    def initiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:
        try:
            # If saved model folder has model then we will compare which model is best
            # Trained model from artifact folder or the model from saved model folder
            logging.info("___________________________________________________________________________________________________________")
            logging.info("If saved model folder has model then we will compare which model is best, "
            "Trained model from artifact folder or the model from saved model folder")
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path==None:                                 # If there is no saved_models then we will accept the currnt model
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
                improved_accuracy=None)                                           
                logging.info(f"Model evaluation artifact: {model_eval_artifact}")
                return model_eval_artifact

            # Finding location of model and target encoder
            logging.info("Finding location of knn_imputer, model and target encoder")
            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()

            # Loading objects
            logging.info("Previous trained objects of knn_imputer, model and target encoder")
            # Previous trained  objects
            model = load_object(file_path=model_path)
            transformer = load_object(file_path=transformer_path)
            
            logging.info("Currently trained model objects")
            # Currently trained model objects
            current_transformer = load_object(file_path=self.data_transformation_artifact.transformer_object_path)
            current_model  = load_object(file_path=self.model_trainer_artifact.model_path)
            
            # Reading test file
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            # output label
            target_df = test_df[TARGET_COLUMN]
            y_true =target_df
            
            # Accuracy using previous trained model
            exclude_columns = [TARGET_COLUMN]
            input_feature_name = list(transformer.feature_names_in_)
            input_feature_test_df= test_df[input_feature_name]

            input_arr= transformer.transform(input_feature_test_df)
            y_pred = model.predict(input_arr)

            # Label decoding with 5 values to get actual string
            print(f"Prediction using previous model: {transformer.inverse_transform(y_pred[:5])}")
            previous_model_score = r2_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"Accuracy using previous trained model: {previous_model_score}")

            # Accuracy using current trained model
            logging.info(f"Accuracy using current trained model")
            exclude_columns = [TARGET_COLUMN]
            input_feature_name = list(current_transformer.feature_names_in_)
            input_feature_test_df= test_df[input_feature_name]

            input_arr= current_transformer.transform(input_feature_test_df)
            y_pred= current_model.predict(input_arr)
            y_true= target_df
            # Label decoding with 5 values to get actual string 
            print(f"Prediction using trained model: {transformer.inverse_transform(y_pred[:5])}")
            current_model_score = r2_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"Accuracy using current trained model: {current_model_score}")
            if current_model_score<=previous_model_score:
                logging.info("Current trained model is not better than previous model")
                raise Exception("Current trained model is not better than previous model")

            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
            improved_accuracy=current_model_score-previous_model_score)
            
            # Improved accuracy
            logging.info(f"Model eval artifact: {model_eval_artifact}")
            return model_eval_artifact
            
        except Exception as e:
            raise AdvertisementException(e,sys)
