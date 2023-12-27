import os
import pandas as pd
import urllib.request as request
import zipfile
from HeartDisease import logger
from HeartDisease.utils.common import get_size
from pathlib import Path
from HeartDisease.entity.config_entity import DataTransformationConfig
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        
    def pre_process(self) :        
        # Replacing
        data = pd.read_csv(self.config.data_path)
        data['HeartDisease'] = data['HeartDisease'].replace({'No': 0, 'Yes': 1})
        
        # Train Test Splitting
        train, test = train_test_split(data, test_size=0.3, random_state=0)
    
        # Encoding
        catcols = list(data.select_dtypes(exclude='number').columns)
        
        encoder = TargetEncoder()
        train[catcols] = encoder.fit_transform(train[catcols], train['HeartDisease'])
        test[catcols] = encoder.transform(test[catcols])
        
        # Uploading csvs 
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
        
