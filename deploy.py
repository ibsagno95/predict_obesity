from fastapi import FastAPI
import joblib
from pydantic import BaseModel

model = joblib.load('logistic_regression_model.pkl')

## initaliser l'API

app = FastAPI()

