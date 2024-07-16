import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exception import CustomException
from logger import logging
import pandas as pd # type: ignore

from sklearn.model_selection import train_test_split # type: ignore
from dataclasses import dataclass
from data_transformation import DataTransformation
from data_transformation import DataTransfromationConfig

from model_trainer import ModelTrainerConfig
from model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    """Any input that iw required will be given to this class"""
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    """This class will ingest the data from the source and save it in the artifacts folder"""
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()   

    def initiate_data_ingestion(self):
            logging.info("Enter the data ingestion method or component")
            try:
                df=pd.read_csv('notebook/data/stud.csv')
                logging.info("Read the dataset as dataframe")

                os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
                df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

                logging.info("Train test split initiated")
                train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

                train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
                test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

                logging.info("Data ingestion completed")

                return(
                    self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
                )
            except Exception as e:
                raise CustomException(e,sys)
            
if __name__=='__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_array,test_array,_=data_transformation.initate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_array,test_array))