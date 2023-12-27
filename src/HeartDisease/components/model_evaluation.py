import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from HeartDisease.utils.common import save_json
from pathlib import Path
from HeartDisease.entity.config_entity import ModelEvaluationConfig
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import torchvision.models as models
from torch import Tensor
from transformers import AdamW

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def prepare_data(self):
        
        test_data = pd.read_csv(self.config.test_data_path)

        X_test = test_data.drop([self.config.target_column], axis=1)
        y_test = test_data[[self.config.target_column]]

        test_target = torch.tensor(y_test.values.astype(np.float32))
        test = torch.tensor(X_test.values.astype(np.float32)) 
        dataset_test = data_utils.TensorDataset(test, test_target)
        
        test_dataloader = DataLoader(dataset_test, batch_size=64, shuffle=True)
        
        return test_dataloader
    
    def log_into_mlflow(self, dataloader):

        model = joblib.load(self.config.model_path)


        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print(mlflow.get_tracking_uri())

        with mlflow.start_run():

            accuracy, test_loss = self.test_loop(dataloader, model)
            
            # Saving metrics as local
            scores = {"accuracy": accuracy, "test_loss": test_loss}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)
            print("Done")
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("test_loss", test_loss)


            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                mlflow.sklearn.log_model(model, "model", registered_model_name="NeuralNetwork")
            else:
                mlflow.sklearn.log_model(model, "model")
                
    
    def test_loop(self,dataloader, model):
        loss_fn = nn.BCELoss()
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                test_loss += loss_fn(pred, y.reshape(-1,1)).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return 100*correct, test_loss

    def predict(row, model):
        # convert row to data
        row = Tensor([row])
        # make prediction
        yhat = model(row)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        return yhat

    
