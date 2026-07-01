"""
model_server/app.py
━━━━━━━━━━━━━━━━━━
Flask 앱 초기화 및 Blueprint 등록
포트 8000에서 실행
"""
from flask import Flask, jsonify
from routes.chat import bp as chat_bp
from routes.curriculum import bp as curriculum_bp
from routes.suggestions import bp as suggestions_bp

app = Flask(__name__)


# ── CORS ───────────────────────────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

@app.route("/api/<path:subpath>", methods=["OPTIONS"])
def options_handler(subpath):
    return "", 200


# ── Blueprint 등록 ─────────────────────────────────────────────────
app.register_blueprint(chat_bp)
app.register_blueprint(curriculum_bp)
app.register_blueprint(suggestions_bp)


# ── 헬스 체크 ──────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
