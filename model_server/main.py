import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import chat, curriculum, suggestions

app = FastAPI(title="Jobseeker QnA Model Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

app.include_router(chat.router,        prefix="/api")
app.include_router(curriculum.router,  prefix="/api")
app.include_router(suggestions.router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}
