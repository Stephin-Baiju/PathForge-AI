# main.py
# Backend for PathForge AI.
# Serves your index.html AND provides the endpoints it actually calls:
#   POST /api/analyze  -> { currentSkills, requiredSkills }
#   POST /api/roadmap  -> { roadmap }
# Also keeps a PDF resume upload endpoint for later, once you want
# to wire a real file-upload button into the frontend.

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import extractor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Request shapes (must match what index.html sends) ----------

class AnalyzeRequest(BaseModel):
    targetRole: str
    resumeText: str = ""
    skillsText: str = ""
    projectsText: str = ""


class GapItem(BaseModel):
    skill: str
    status: str
    currentLevel: int
    importance: int


class RoadmapRequest(BaseModel):
    targetRole: str
    gaps: list[GapItem]


# ---------- Endpoints index.html actually calls ----------

@app.post("/api/analyze")
def api_analyze(payload: AnalyzeRequest):
    has_info = payload.resumeText.strip() or payload.skillsText.strip()
    if not has_info:
        raise HTTPException(status_code=400, detail="Provide resume text or skills text.")
    try:
        return extractor.analyze_skills(
            target_role=payload.targetRole,
            resume_text=payload.resumeText,
            skills_text=payload.skillsText,
            projects_text=payload.projectsText,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/roadmap")
def api_roadmap(payload: RoadmapRequest):
    try:
        gaps_as_dicts = [g.model_dump() for g in payload.gaps]
        return extractor.generate_roadmap(
            target_role=payload.targetRole,
            gaps=gaps_as_dicts,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Optional: PDF resume upload (not yet called by index.html,
# but kept here so you can wire a real upload button in later) ----------

@app.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...),
    target_role: str = Form("Software Engineer"),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are currently supported.")
    try:
        file_bytes = await file.read()
        extracted_text = extractor.extract_text_from_pdf(file_bytes)
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")
        result = extractor.analyze_skills(target_role=target_role, resume_text=extracted_text)
        return {"status": "success", "filename": file.filename, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Serve index.html itself (must be LAST, after all routes above) ----------

app.mount("/", StaticFiles(directory=".", html=True), name="static")
