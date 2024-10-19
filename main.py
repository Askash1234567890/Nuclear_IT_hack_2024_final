import os

import pandas as pd
import uvicorn

from flask import Flask, jsonify, request

app = Flask(__name__)

# Directory to save uploaded files
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/index.html', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    print("MEOWMEOW")
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Read the CSV file
    try:
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file_path)
            return jsonify({'message': 'File uploaded successfully', 'data': data.to_dict(orient='records')}), 200
        else:
            return jsonify({'error': 'Unsupported file type. Please upload a CSV file.'}), 400
    except Exception as e:
        print(f"Error reading the file: {e}")  # Log the error
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    uvicorn.run('main:app', host='localhost', port=3000, reload=True)
