from flask import Flask, request, jsonify
from mcq_gen import mcqGen
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Welcome to MCQ generation API"

@app.route('/api/mcq', methods=['POST'])
def mcq():
    data = request.form
    topic = data.get('topic').strip()
    quantity = data.get('quantity')
    difficulty = data.get('difficulty')
    file = request.files['file'] if 'file' in request.files else None
    inputText = data.get('inputText')
    status = data.get('status')

    try:
        mcqs = mcqGen(topic, quantity, difficulty, file, inputText, status)
        return jsonify({'mcqs': mcqs})
    except ValueError:
        return jsonify({'error': 'Error'}), 400


if __name__ == '__main__':
    app.run()