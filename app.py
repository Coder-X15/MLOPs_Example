#! /usr/bin/env python3

import mlflow
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mlflow.set_tracking_uri("http://localhost:5000")
model = mlflow.pyfunc.load_model(os.getenv("MODEL_URI", "models:/Iris_MLP_Classifier/Production"))


@app.route("/predict", methods=["POST"])
async def predict(request):
    # loading the last saved model from MLflow Model Registry

    data = await request.json()
    print(data)
    input_df = pd.DataFrame(dict(data))
    # make prediction
    predictions = model.predict(input_df)
    # return predictions as a HTTP response
    from fastapi.responses import JSONResponse
    response = {"predictions": predictions.tolist()}
    return JSONResponse(content=response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)