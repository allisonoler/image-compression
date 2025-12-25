from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow React frontend to make requests

@app.route('/api/hello')
def hello():
    return jsonify({'message': 'Hello from Flask!', 'status': 'success'})

@app.route('/api/data')
def get_data():
    return jsonify({
        'items': ['Item 1', 'Item 2', 'Item 3'],
        'count': 3
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)