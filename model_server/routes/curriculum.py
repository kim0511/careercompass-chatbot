from flask import Blueprint, jsonify
from services.curriculum import get_curriculum_data
from services.job_tracks import get_job_tracks_data

bp = Blueprint("curriculum", __name__)


@bp.route("/api/curriculum")
def api_curriculum():
    data = get_curriculum_data()
    data["job_tracks"] = get_job_tracks_data()
    return jsonify(data)
