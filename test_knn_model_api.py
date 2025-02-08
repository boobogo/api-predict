import pytest
from flask import Flask
from flask.testing import FlaskClient
import json
from knn_model_api import app  # Import the Flask app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_predict(client: FlaskClient):
    # Define the input data
    input_data = {
        'features': [5.1, 3.5, 1.4, 0.2]
    }

    # Send a POST request to the /predict endpoint
    response = client.post('/predict', data=json.dumps(input_data), content_type='application/json')

    # Assert the response status code
    assert response.status_code == 200

    # Assert the response data
    response_data = response.get_json()
    assert 'prediction' in response_data
    assert isinstance(response_data['prediction'], int)

if __name__ == '__main__':
    pytest.main()