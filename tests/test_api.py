from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
SAMPLE = Path(__file__).parent / "sample.jpg"

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_predict_valid_image():
    with open(SAMPLE, "rb") as f:
        r = client.post("/predict", files={"file": ("sample.jpg", f, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert set(body["probabilities"]) == {"glioma", "meningioma", "notumor", "pituitary"}
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3
    assert body["prediction"] in body["probabilities"]
    assert "Not a medical device" in body["disclaimer"]

def test_predict_rejects_non_image():
    r = client.post("/predict", files={"file": ("fake.jpg", b"hello", "image/jpeg")})
    assert r.status_code == 400


def test_root_redirects_to_docs():
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/docs"