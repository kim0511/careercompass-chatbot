from flask import Blueprint, request, jsonify

bp = Blueprint("suggestions", __name__)

ORG_SUGGESTIONS = {
    "한국자산관리공사": ["한국자산관리공사(캠코) 주요 채용 직무는?"],
    "부산항만공사":     ["부산항만공사 지원 자격 조건은?"],
    "한국해양진흥공사": ["한국해양진흥공사 신입 채용도 해?"],
    "주택도시보증공사": ["주택도시보증공사(HUG) 어떤 직무를 채용해?"],
    "기술보증기금":     ["기술보증기금 자격증 조건은?"],
}


@bp.route("/api/suggestions")
def api_suggestions():
    org = request.args.get("org", "")
    suggestions = ORG_SUGGESTIONS.get(
        org,
        [f"{org} 채용 자격 조건은?", f"{org} 전형 절차는?", f"{org} 우대사항은?", f"{org} NCS 영역은?"],
    )
    return jsonify({"suggestions": suggestions})
