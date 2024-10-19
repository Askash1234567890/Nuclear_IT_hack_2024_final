from fastapi.testclient import TestClient

from scripts.api import app

client = TestClient(app)


def test_load_data_successfully():
    request_data = {
        "data": [
            {"column1": "value1", "column2": "value2"},
            {"column1": "value3", "column2": "value4"}
        ]
    }
    response = client.post("/load_data", json=request_data)
    assert response.status_code == 200
    assert response.json() == {"message": "Data loaded successfully"}


def test_load_data_with_invalid_data():
    request_data = {
        "data": "invalid_data"
    }
    response = client.post("/load_data", json=request_data)
    assert response.status_code == 422


def test_get_personal_statistic_successfully():
    request_data = {"id": 0}
    response = client.post("/get_personal_statistic", json=request_data)
    assert response.status_code == 200
    assert "personal_statistic" in response.json()


def test_get_personal_statistic_with_invalid_id():
    request_data = {"id": -1}
    response = client.post("/get_personal_statistic", json=request_data)
    assert response.status_code == 500


def test_get_graphics_successfully():
    response = client.post("/get_graphics")
    assert response.status_code == 200
    assert response.json() == {"message": "Graphics generated successfully"}
