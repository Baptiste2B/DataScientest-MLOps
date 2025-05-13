from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API de prédiction de sentiment prête 🎯"}

def test_predict_binary_valid():
    response = client.post(
        "/predict/binary",
        json={"text": "Good service, but the food was not great."}
    )
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_binary_empty_text():
    response = client.post(
        "/predict/binary",
        json={"text": ""}
    )
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_multiclass_valid():
    response = client.post(
        "/predict/multiclass",
        json={"text": "I love the ambiance and the food was delicious!"}
    )
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_multiclass_empty():
    response = client.post(
        "/predict/multiclass",
        json={"text": ""}
    )
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_binary_missing_field():
    response = client.post(
        "/predict/binary",
        json={}
    )
    assert response.status_code == 422  # FastAPI renvoie 422 pour validation manquante

def test_invalid_route():
    response = client.get("/invalid")
    assert response.status_code == 404