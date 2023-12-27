import os
import pandas as pd
import urllib.request as request
import zipfile
from HeartDisease import logger
from HeartDisease.utils.common import get_size
from pathlib import Path
from HeartDisease.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config


    def validate_all_columns(self)-> bool:
        try:
            validation_status = True

            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()

            
            for col in all_cols:
                if col not in all_schema:
                    validation_status = False and validation_status
                else:
                    validation_status = True and validation_status
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")
            return validation_status
        
        except Exception as e:
            raise e

