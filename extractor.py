# extractor.py
# Handles: reading text out of uploaded PDFs, and calling Gemini to
# produce the exact JSON shapes PathForge AI's frontend expects.

import os
import json
import re
import io
import google.generativeai as genai
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3.6-flash")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Takes raw PDF bytes (from an uploaded file) and returns plain text."""
    reader = PdfReader(io.BytesIO(file_bytes))
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts).strip()


def _extract_json(text: str):
    """Strips ```json fences etc. so json.loads() doesn't choke on
    formatting Gemini sometimes adds even when told not to."""
    cleaned = re.sub(r"^```json|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def analyze_skills(target_role: str, resume_text: str = "", skills_text: str = "", projects_text: str = "") -> dict:
    """Returns {"currentSkills": [...], "requiredSkills": [...]} —
    exactly the shape index.html's runAnalysis() expects."""
    prompt = f"""You are a career skills analyst. Based on the learner's input below,
return ONLY valid JSON (no markdown, no explanation) in this exact shape:

{{
  "currentSkills": [{{"name": "Skill Name", "level": 0-100, "note": "short reason"}}],
  "requiredSkills": [{{"name": "Skill Name", "importance": 0-100}}]
}}

Target role: {target_role}
Resume: {resume_text}
Self-listed skills: {skills_text}
Projects: {projects_text}

- currentSkills: extract skills evident from the input, estimate level 0-100.
- requiredSkills: 6-10 key skills needed for the target role, importance 0-100.
Return ONLY the JSON object, nothing else."""

    response = model.generate_content(prompt)
    return _extract_json(response.text)


def generate_roadmap(target_role: str, gaps: list) -> dict:
    """Takes a list of gap dicts (skill, status, currentLevel, importance)
    and returns {"roadmap": [...]} — exactly what index.html expects."""
    gaps_desc = "\n".join(
        f"- {g['skill']} (status: {g['status']}, current: {g['currentLevel']}, importance: {g['importance']})"
        for g in gaps
    )
    prompt = f"""You are a learning roadmap designer. For the target role "{target_role}",
sequence a roadmap for these skill gaps, highest priority first.

Return ONLY valid JSON (no markdown, no explanation) in this exact shape:

{{
  "roadmap": [
    {{
      "skill": "Skill Name",
      "status": "critical or developing",
      "weeks": 1-6,
      "learn": "short instruction for the learn stage",
      "practice": "short instruction for the practice stage",
      "build": "short instruction for the build stage",
      "validate": "short instruction for the validate stage"
    }}
  ]
}}

Gaps to sequence:
{gaps_desc}

Return ONLY the JSON object, nothing else."""

    response = model.generate_content(prompt)
    return _extract_json(response.text)
