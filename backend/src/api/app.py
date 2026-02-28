from flask import Flask, request, jsonify
from flask_cors import CORS

counter = 0 # testing 

app = Flask(__name__)
CORS(app)

@app.route("/commentary", methods=["POST"])
def commentary ():
    data = request.json
    fen_before = data["fen_before"]
    move = data["move"]
    fen_after = data["fen_after"]

    # TO DO PLUG IN MODEL
    global counter
    
    comment = counter
    counter += 1

    return jsonify({ "comment": str(comment) })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)