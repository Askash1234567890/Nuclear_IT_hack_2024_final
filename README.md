# Exit Interview Analyzer

This project provides a FastAPI-based web service for analyzing exit interview data. It includes functionalities for
loading data, preprocessing, clustering, generating statistics, and creating visual graphics.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/exit-interview-analyzer.git
    cd exit-interview-analyzer
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Start the FastAPI server:
    ```sh
    uvicorn scripts.api:app --reload
    ```

2. Open your browser and navigate to `http://127.0.0.1:8000/docs` to access the interactive API documentation.

## API Endpoints

- **POST /load_data**: Load data into the analyzer.
- **POST /preprocess**: Preprocess the given words.
- **POST /clustering**: Perform clustering on the given words.
- **POST /get_statistic**: Get statistics for the given words.
- **POST /get_personal_statistic**: Get personal statistics for a given ID.
- **POST /get_graphics**: Generate graphics for the loaded data.

### Example Requests

- **Load Data**
    ```sh
    curl -X 'POST' \
      'http://127.0.0.1:8000/load_data' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "data": [{"column1": "value1", "column2": "value2"}]
    }'
    ```

- **Preprocess Data**
    ```sh
    curl -X 'POST' \
      'http://127.0.0.1:8000/preprocess' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "words": ["word1", "word2"]
    }'
    ```

## Project Structure

```plaintext
.
├── scripts/
│   ├── api.py                # FastAPI application
│   ├── analyzer.py           # Analyzer class with data processing methods
│   ├── grad_search.py        # Gradient search implementation
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
