from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from sklearn.tree import DecisionTreeRegressor
# import numpy as np
import os
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model_path = os.path.join(os.getcwd(), "DecisionTree.pkl")  
model = joblib.load(model_path)


class WeatherData(BaseModel):
    holiday: str  
    temp: float
    rain_1h: float
    snow_1h: float
    clouds_all: int
    weather_main: str
    weather_description: str
    day: str
    month: int
    year: int
    hour: int

@app.post("/predict")
async def predict_weather(data: WeatherData):
    if model is None:
        return {"error": "Le modèle n'a pas pu être chargé."}
    try:
        # Préparer les données pour la prédiction
        input_data = pd.DataFrame([{
            "holiday": data.holiday,
            "temp": data.temp,
            "rain_1h": data.rain_1h,
            "snow_1h": data.snow_1h,
            "clouds_all": data.clouds_all,
            "weather_main": data.weather_main,
            "weather_description": data.weather_description,
            "day": data.day,
            "month": data.month,
            "year": data.year,
            "hour": data.hour
        }])
        
        prediction = model.predict(input_data)

        return {"prediction": prediction.tolist()}

    except Exception as e:
        return {"error": f"Erreur lors de la prédiction : {str(e)}"}


@app.get("/predict_test")
async def predict_test():
    if model is None:
        return {"error": "Le modele n'a pas pu être chargeé."}
    try:
        data = {
            "holiday": None,
            "temp": 269.04,
            "rain_1h": 0.0,
            "snow_1h": 0.0,
            "clouds_all": 90,
            "weather_main": "Clouds",
            "weather_description": "overcast clouds",
            "day": "Tuesday",
            "month": 11,
            "year": 2012,
             "hour": 13
        }

        sample_data = pd.DataFrame([data])

        prediction = model.predict(sample_data)

        return { "prediction": prediction.tolist()} 

    except Exception as e:
        return {"error": f"Erreur lors de la prediction : {str(e)}"}

    


@app.get("/")
async def read_root():
    return {"message": "Bienvenue sur FastAPI avec Docker !"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
