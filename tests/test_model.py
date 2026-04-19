from src.train import train
from src.predict import predict

def test_model():
    train()
    sample = [5,5000,1,0,1,0,2]
    result = predict(sample)
    assert result is not None
