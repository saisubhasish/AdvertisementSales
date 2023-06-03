import os, sys

from advertisement.logger import logging
from advertisement.exception import AdvertisementException

from advertisement.entity import config_entity, artifact_entity
from advertisement.entity.config_entity import DataIngestionConfig

from advertisement.components.data_ingestion import DataIngestion
from advertisement.components.data_validation import DataValidation
from advertisement.components.data_transformation import DataTransformation
from advertisement.components.model_trainer import ModelTrainer

from advertisement.entity.config_entity import DATA_FILE_PATH

if __name__ == '__main__':
    try:
        training_pipeline_config = config_entity.TrainingPipelineConfig()

        #data ingestion  
        data_ingestion_config  = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        print(data_ingestion_config.to_dict())
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion(DATA_FILE_PATH)

        #data validation
        data_validation_config = config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(data_validation_config=data_validation_config,
                    data_ingestion_artifact=data_ingestion_artifact)

        data_validation_artifact = data_validation.initiate_data_validation()

        #data transformation
        data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation = DataTransformation(data_transformation_config=data_transformation_config, 
        data_validation_artifact=data_validation_artifact)
        data_transformation_artifact = data_transformation.initiate_data_transformation()

        #model trainer
        model_trainer_config = config_entity.ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()

    except Exception as e:
        raise AdvertisementException(e, sys)