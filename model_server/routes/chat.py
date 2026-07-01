from flask import Blueprint, request, jsonify
from qa_pipeline import process_question

bp = Blueprint("chat", __name__)


@bp.route("/api/chat", methods=["POST"])
def api_chat():
    data       = request.get_json(force=True)
    question   = data.get("question", "").strip()
    history    = data.get("history", [])
    ncs_detail = data.get("ncs_detail", True)

    if not question:
        return jsonify({"error": "질문이 없습니다."}), 400

    try:
        answer, meta = process_question(question, history=history or None, ncs_detail=ncs_detail)
        return jsonify({"answer": answer, "meta": meta})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
