from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.cv import CVAnalysisResponse
from app.schemas.job import JobAnalysisResponse
from app.services.cv_parser import extract_cv_text

router = APIRouter(prefix="/ai", tags=["AI"])


@router.post("/analyze-cv")
async def analyze_cv(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="CV filename is required.",
        )

    if not file.filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are supported.",
        )

    file_bytes = await file.read()

    try:
        text = extract_cv_text(
            file_bytes=file_bytes,
            filename=file.filename,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from CV.",
        )

    return {
        "filename": file.filename,
        "text": text,
    }


@router.post("/analyze-job", response_model=JobAnalysisResponse)
def analyze_job():
    return JobAnalysisResponse(
        role="Backend Developer",
        domain="Web Development",
        responsibilities=[
            "Develop REST APIs",
            "Design database schemas",
            "Maintain backend services",
        ],
        required_skills=[
            "NestJS",
            "PostgreSQL",
            "REST API",
        ],
        preferred_skills=[
            "Docker",
            "Redis",
        ],
        experience_requirements=[
            "2+ years of backend development experience",
        ],
    )   