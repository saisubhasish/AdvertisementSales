import os,sys 

from typing import Optional
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from advertisement import utils
from advertisement.logger import logging
from advertisement.exception import AdvertisementException
from advertisement.entity import artifact_entity,config_entity


class ModelTrainer:

    def __init__(self, model_trainer_config:config_entity.ModelTrainerConfig,
                data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact

        except Exception as e:
            raise AdvertisementException(e, sys)

    def train_model(self,x,y):
        """
        Model training
        """
        try:
            lr =  LinearRegression()
            lr.fit(x,y)
            return lr

        except Exception as e:
            raise AdvertisementException(e, sys)

    def initiate_model_trainer(self,)->artifact_entity.ModelTrainerArtifact:
        """
        Preparing dataset
        """
        try:
            logging.info("Loading train and test array.")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info("Splitting input and target feature from both train and test arr.")
            x_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]


            logging.info("Train the model")
            model = self.train_model(x=x_train,y=y_train)

            # Prediction and accuracy using training data
            logging.info("Calculating f1 train score")
            yhat_train = model.predict(x_train)
            r2_train_score= r2_score(y_true=y_train, y_pred=yhat_train)

            # Prediction and acuracy using test data
            logging.info("Calculating f1 test score")
            yhat_test = model.predict(x_test)
            r2_test_score= r2_score(y_true=y_test, y_pred=yhat_test)
            
            logging.info(f"train score:{r2_train_score} and tests score {r2_test_score}")
            logging.info("Checking if our model is a good model or not")
            if r2_test_score<self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {r2_test_score}")

            logging.info("Checking if our model is overfiiting or not")
            diff = abs(r2_train_score-r2_test_score)   # Checking the difference by removing -ve

            # Check for overfitting or underfiiting on threshold
            if diff>self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            # Saving trained model if it passes using utils
            logging.info("Saving mode object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            # Prepare artifact
            logging.info("Prepare the artifact")
            model_trainer_artifact  = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path, 
            r2_train_score=r2_train_score, r2_test_score=r2_test_score)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
            
        except Exception as e:
            raise AdvertisementException(e, sys)

