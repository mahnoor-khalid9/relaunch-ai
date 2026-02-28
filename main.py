import json, logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from agents import run_analysis

log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(handlers=[logging.FileHandler(log_dir / "relaunch_main.json")], level=logging.INFO)

app = FastAPI(title="relaunch.ai")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

cache: dict[str, dict] = {}


class AnalyseRequest(BaseModel):
    # Stage 1 — required
    startup_name:        str
    industry:            str  = ""
    country:             str  = ""
    year_founded:        str  = ""
    year_shutdown:       str  = ""
    funding_range:       str  = ""
    product_description: str  = ""
    # Stage 2 — optional
    startup_overview:    str  = ""
    why_failed_shutdown: str  = ""
    founder_why_failed:  str  = ""
    customer_feedback:   str  = ""
    pivots_tried:        str  = ""
    what_different:      str  = ""
    # Stage 3 — optional checkboxes
    context_signals:     List[str] = []


@app.post("/analyse")
async def analyse(req: AnalyseRequest):
    if not req.startup_name.strip():
        raise HTTPException(400, "startup_name is required")
    key = req.startup_name.strip().lower()
    result = run_analysis(req.model_dump())
    cache[key] = result
    return JSONResponse({
        "startup_name":       req.startup_name,
        "research":           result.get("research", {}),
        "autopsy":            result.get("autopsy", {}),
        "revival":            result.get("revival", {}),
        "copywriter_outputs": result.get("copywriter_outputs", {}),
        "marketing_html":     result.get("marketing_html", ""),
        "progress":           result.get("progress", []),
        "data_confidence":    result.get("data_confidence", "medium"),
    })


@app.get("/preview/{startup_name}", response_class=HTMLResponse)
async def preview(startup_name: str):
    key = startup_name.strip().lower()
    if key not in cache:
        raise HTTPException(404, "Run /analyse first.")
    return HTMLResponse(cache[key].get("marketing_html", "<p>No page.</p>"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse((Path(__file__).parent / "static" / "index.html").read_text())


app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
