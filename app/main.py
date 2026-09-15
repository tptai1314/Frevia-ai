from fastapi import FastAPI

from app.api.v1.ai import router as ai_router


app = FastAPI(
    title="Frevia AI",
    description="AI service for Frevia CV-JD matching",
    version="0.1.0",
)

app.include_router(
    ai_router,
    prefix="/api/v1",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "frevia-ai",
    }