from dotenv_vault import load_dotenv
from flask import Flask, request, jsonify, render_template
from ice_breaker import ice_break_with

load_dotenv()

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    name = request.form["name"]
    summary, picture_url = ice_break_with(name)
    return jsonify({"summary_and_facts": summary.to_dict(), "picture_url": picture_url})

if __name__ == "__main__":
    app.run(debug=True)