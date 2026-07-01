from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []
    ncs_detail: bool = True


@router.post("/chat")
async def api_chat(req: ChatRequest):
    if not req.question.strip():
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="질문이 없습니다.")
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from qa_pipeline import process_question
        answer, meta = process_question(
            req.question,
            history=req.history or None,
            ncs_detail=req.ncs_detail,
        )
        return {"answer": answer, "meta": meta}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))
