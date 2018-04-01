from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/')
def get_recommendations():
    return jsonify({
        'total': 50,
        'recommendations': [{ 'ticker': 'ABC', 'amount': 1.56 }, { 'ticker': 'CBC', 'amount': 0 }]
    })
