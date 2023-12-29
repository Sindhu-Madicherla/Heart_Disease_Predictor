import pandas as pd
import os
from HeartDisease import logger
from sklearn.linear_model import ElasticNet
import joblib
from HeartDisease.components.NeuralNetwork import NeuralNetwork
from HeartDisease.entity.config_entity import ModelTrainerConfig
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import torchvision.models as models
from torch import Tensor
from transformers import AdamW
import pickle

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def prepare_data(self):
        train_data = pd.read_csv(self.config.train_data_path)


        X_train = train_data.drop([self.config.target_column], axis=1)
        y_train = train_data[[self.config.target_column]]
        
        train_target = torch.tensor(y_train.values.astype(np.float32))
        train = torch.tensor(X_train.values.astype(np.float32)) 
        dataset_train = data_utils.TensorDataset(train, train_target)
        
        train_dataloader = DataLoader(dataset_train, batch_size=64, shuffle=True)
        
        return train_dataloader
    
    
    def train(self, dataloader):

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {device} device")
        
        model = NeuralNetwork().to(device)
        print(model)
        
        learning_rate = self.config.learning_rate
        epochs = self.config.epochs
        batch_size = 64
        loss_fn = nn.BCELoss()
        optimizer = AdamW(model.parameters(), lr = 5e-5)
        
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train_loop(dataloader, model, loss_fn, optimizer)

        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))
        print("Done!")
        
        
    def train_loop(self, dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        losses = []
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y.reshape(-1,1))

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 1000 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                losses.append(loss)
