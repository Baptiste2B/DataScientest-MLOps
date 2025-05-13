from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from app.predict_model import predict_binary as pb, predict_multiclass as pm

app = FastAPI()

class Comment(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "API de prédiction de sentiment prête 🎯"}

@app.post("/predict/binary")
def predict_binary(comment: Comment):
    df = pd.DataFrame([{
        "Content": comment.text,
        "dates.experiencedDate": pd.Timestamp.now(),
        "dates.publishedDate": pd.Timestamp.now()
    }])
    sentiment = pb(df)
    if sentiment is not None:
        return {"prediction": sentiment.tolist() if hasattr(sentiment, "tolist") else sentiment}
    else:
        return {"error": "Prediction failed"}

@app.post("/predict/multiclass")
def predict(comment: Comment):
    df = pd.DataFrame([{
        "Content": comment.text,
        "dates.experiencedDate": pd.Timestamp.now(),
        "dates.publishedDate": pd.Timestamp.now()
    }])
    sentiment = pm(df)
    if sentiment is not None:
        return {"prediction": sentiment.tolist() if hasattr(sentiment, "tolist") else sentiment}
    else:
        return {"error": "Prediction failed"}