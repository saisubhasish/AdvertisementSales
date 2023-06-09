import os,sys 
import numpy as np
from dask import dataframe as dd
from advertisement import utils
from typing import Optional
from advertisement.logger import logging
from advertisement.config import TARGET_COLUMN
from advertisement.exception import AdvertisementException
from advertisement.entity import artifact_entity,config_entity



class DataValidation:


    def __init__(self,
                    data_validation_config:config_entity.DataValidationConfig,
                    data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.validation_error=dict()
        except Exception as e:
            raise AdvertisementException(e, sys)

    

    def drop_missing_values_columns(self,df:dd.DataFrame,report_key_name:str)->Optional[dd.DataFrame]:
        """
        This function will drop column which contains missing value more than specified threshold

        df : Accepts a dask dataframe
        =========================================================================================
        returns dask Dataframe if atleast a single column is available after missing columns drop else None
        """
        try:
            
            threshold = self.data_validation_config.missing_threshold
            null_report = df.isna().sum()/df.shape[0]
            # Selecting column name which contains null
            logging.info(f"selecting column name which contains null above to {threshold}")
            drop_column_names = null_report[null_report>threshold].index

            logging.info(f"Columns to drop: {list(drop_column_names)}")
            self.validation_error[report_key_name]=list(drop_column_names)
            df.drop(list(drop_column_names),axis=1,inplace=True)

            # Return None if no columns left
            if len(df.columns)==0:
                return None
            return df
        except Exception as e:
            raise AdvertisementException(e, sys)


    def is_required_columns_exists(self,base_df:dd.DataFrame,current_df:dd.DataFrame,report_key_name:str)->bool:
        """
        This function checks if required columns exists or not by comparing current df with base df and returns
        output as True and False
        """
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns

            missing_columns = []
            for base_column in base_columns:    
                if base_column not in current_columns:
                    logging.info(f"Column: [{base_column} is not available.]")
                    missing_columns.append(base_column)

            # Return False if there are missing columns in current df other wise True
            if len(missing_columns)>0:
                self.validation_error[report_key_name]=missing_columns
                return False    
            return True
            
        except Exception as e:
            raise AdvertisementException(e, sys)

    def data_drift(self,base_df:dd.DataFrame,current_df:dd.DataFrame,report_key_name:str):
        try:
            drift_report=dict()

            base_columns = base_df.columns
            current_columns = current_df.columns

            for base_column in base_columns:
                base_data,current_data = base_df[base_column],current_df[base_column]
                # Null hypothesis : Both column data has same distribution
                
                logging.info(f"Checking Data Types of '{base_column}': {base_data.dtype}, {current_data.dtype} ")
                
                if base_df[base_column].dtype == current_df[base_column].dtype:
                    drift_report[base_column] = {"Same data type": True}
                else:
                    drift_report[base_column] = {"Same data type": False}

                logging.info(f"Checking number of classes in in {base_column} column\n: {base_df[base_column].value_counts(), current_df[base_column].value_counts()}")
                if len(base_df[base_column].value_counts()) == len(current_df[base_column].value_counts()):
                    drift_report[base_column] = {"Column has equal number of classes": True}
                else:
                    drift_report[base_column] = {"Column has equal number of classes": False} 

            self.validation_error[report_key_name]=drift_report
            
        except Exception as e:
            raise AdvertisementException(e, sys)

    def initiate_data_validation(self)->artifact_entity.DataValidationArtifact:
        try:
            logging.info("Reading base dataframe")
            base_df = dd.read_csv(self.data_validation_config.base_file_path, dtype={'Age': 'float64',
                        'EstimatedSalary': 'float64'}).head(n=200)

            logging.info("Drop null values colums from base df")
            base_df=self.drop_missing_values_columns(df=base_df,report_key_name="missing_values_within_base_dataset")

            logging.info("Reading train dataframe")
            train_df = dd.read_csv(self.data_ingestion_artifact.train_file_path).head(n=200)
            logging.info("Reading test dataframe")
            test_df = dd.read_csv(self.data_ingestion_artifact.test_file_path).head(n=200)

            logging.info("Drop null values colums from train df")
            train_df = self.drop_missing_values_columns(df=train_df,report_key_name="missing_values_within_train_dataset")
            logging.info("Drop null values colums from test df")
            test_df = self.drop_missing_values_columns(df=test_df,report_key_name="missing_values_within_test_dataset")

            logging.info("Is all required columns present in train df")
            train_df_columns_status = self.is_required_columns_exists(base_df=base_df, current_df=train_df,report_key_name="missing_columns_within_train_dataset")
            logging.info("Is all required columns present in test df")
            test_df_columns_status = self.is_required_columns_exists(base_df=base_df, current_df=test_df,report_key_name="missing_columns_within_test_dataset")

            if train_df_columns_status:     # If True
                logging.info("As all column are available in train df hence detecting data drift in train dataframe")
                self.data_drift(base_df=base_df, current_df=train_df,report_key_name="data_drift_within_train_dataset")
            if test_df_columns_status:     # If True
                logging.info("As all column are available in test df hence detecting data drift test dataframe")
                self.data_drift(base_df=base_df, current_df=test_df,report_key_name="data_drift_within_test_dataset")

            logging.info("create dataset directory folder if not available for validated train file and test file")
            # Create dataset directory folder if not available
            dataset_dir = os.path.dirname(self.data_validation_config.train_file_path)
            os.makedirs(dataset_dir,exist_ok=True)

            logging.info("Saving validated train df and test df to dataset folder")
            # Saving validated train df and test df to dataset folder
            train_df.to_csv(path_or_buf=self.data_validation_config.train_file_path,index=False,header=True)
            test_df.to_csv(path_or_buf=self.data_validation_config.test_file_path,index=False,header=True)
            
            # Write the report
            logging.info("Writing report in yaml file")
            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path,
            data=self.validation_error)   # valiadtion_error: drop columns, missing columns, drift report

            data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path, 
            train_file_path=self.data_validation_config.train_file_path, test_file_path=self.data_validation_config.test_file_path)
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise AdvertisementException(e, sys)