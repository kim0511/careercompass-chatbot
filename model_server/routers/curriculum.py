from fastapi import APIRouter

router = APIRouter()


@router.get("/curriculum")
async def api_curriculum():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from server import get_curriculum_data, get_job_tracks_data
    data = get_curriculum_data()
    data["job_tracks"] = get_job_tracks_data()
    return data
