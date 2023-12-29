import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import pickle


class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        file = open('artifacts/data_transformation/encoder.obj','rb')
        self.encoder = pickle.load(file)
        file.close()

    
    def predict(self, data):
        
        # Applying categroy transformation
        cols = ['BMI','Smoking','AlcoholDrinking','Stroke','PhysicalHealth','MentalHealth','DiffWalking','Sex','AgeCategory','Race','Diabetic','PhysicalActivity','GenHealth','SleepTime','Asthma','KidneyDisease','SkinCancer']
        catcols = ['Smoking','AlcoholDrinking','Stroke','DiffWalking','Sex','AgeCategory','Race','Diabetic','PhysicalActivity','GenHealth','Asthma','KidneyDisease','SkinCancer']
        data = pd.DataFrame(data,columns=cols)
        data[catcols] = self.encoder.transform(data[catcols])
        print("Data after tranformation is ")
        print(data)        
        data = torch.tensor(data.values.astype(np.float32))
        prediction = self.model(data)
        print("Prediction is")
        print(prediction)
        return 'Yes' if prediction[0][0]>0.5 else 'No'