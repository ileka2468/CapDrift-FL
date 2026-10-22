"""
OmniFed: Modular Federated Learning Studio & Unified Web GUI.
Provides 3 full functional suites:
1. Cluster Orchestration & Experiment Builder (1-click cluster launch, live streaming logs, node health)
2. Empirical Analytics & Telemetry (convergence curves, dynamic bandwidth calculator, straggler analysis)
3. Universal Model Inference & Explainable AI (Grad-CAM heatmaps across N classes)
"""

import io
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image
import uvicorn

from inference import FederatedClassifier
from model import SUPPORTED_ARCHITECTURES, get_model_size_mb, get_model
from orchestrator import orchestrator
from dataset import scan_dataset_classes

app = FastAPI(title="OmniFed: Modular Federated Learning Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "global_test" / "train"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_PATH = PROJECT_ROOT / "config" / "default_config.json"

# Initialize universal classifier
classifier = FederatedClassifier(
    weights_path=str(RESULTS_DIR / "best_model.pth"),
    fallback_weights=str(RESULTS_DIR / "global_model.pth"),
    metadata_path=str(RESULTS_DIR / "model_metadata.json")
)


@app.get("/api/status")
def get_status():
    """Returns cluster state, model information, and active training status."""
    metrics_path = RESULTS_DIR / "fl_metrics.json"
    history = []
    best_acc = 0.0
    latest_loss = 0.0

    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            if history:
                best_acc = max(entry["accuracy"] for entry in history) * 100
                latest_loss = history[-1]["loss"]
        except Exception:
            pass

    return {
        "status": "online",
        "training_job": orchestrator.get_status(),
        "model": {
            "architecture": classifier.architecture,
            "classes": classifier.classes,
            "num_classes": classifier.num_classes,
            "weights_source": classifier.weights_source.split("/")[-1].split("\\")[-1]
        },
        "best_accuracy": round(best_acc, 2),
        "latest_loss": round(latest_loss, 4),
        "rounds_completed": len(history),
        "device": str(classifier.device).upper()
    }


@app.get("/api/cluster/health")
def get_cluster_health():
    """Returns real-time ping and connectivity for all 4 physical nodes."""
    return {"nodes": orchestrator.get_cluster_health()}


@app.get("/api/architectures")
def get_architectures():
    """Returns list of supported vision architectures with parameter sizes."""
    return {"architectures": SUPPORTED_ARCHITECTURES}


@app.get("/api/datasets")
def get_datasets():
    """Scans and lists local datasets available for federated training."""
    data_dir = PROJECT_ROOT / "data"
    available = []
    
    if data_dir.exists():
        for d in data_dir.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                classes, count = scan_dataset_classes(str(d))
                available.append({
                    "id": d.name,
                    "name": d.name.replace("_", " ").title(),
                    "path": str(d.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "classes": classes,
                    "num_classes": count
                })
    return {"datasets": available}


@app.post("/api/train/start")
def start_training(
    rounds: int = Form(6),
    epochs: int = Form(1),
    architecture: str = Form("mobilenet_v3_small")
):
    """Triggers federated cluster training in the background."""
    result = orchestrator.start_training(
        rounds=rounds,
        epochs=epochs,
        architecture=architecture
    )
    return JSONResponse(content=result)


@app.get("/api/train/status")
def get_training_status():
    """Returns the current training execution state."""
    return orchestrator.get_status()


@app.get("/api/train/logs")
def get_training_logs(offset: int = 0):
    """Streams new training terminal log lines."""
    logs = orchestrator.get_logs(offset=offset)
    return {"logs": logs, "next_offset": offset + len(logs)}


@app.get("/api/samples")
def get_samples():
    """Returns curated holdout test samples for 1-click classification."""
    samples = []
    normal_dir = DATA_DIR / "NORMAL"
    pneu_dir = DATA_DIR / "PNEUMONIA"

    if normal_dir.exists():
        normal_files = sorted(list(normal_dir.glob("*.jpeg")) + list(normal_dir.glob("*.jpg")))
        for f in normal_files[:4]:
            samples.append({
                "id": f.stem,
                "label": "NORMAL",
                "category": "Normal Patient",
                "filename": f.name,
                "path": str(f.relative_to(PROJECT_ROOT)).replace("\\", "/")
            })

    if pneu_dir.exists():
        pneu_files = sorted(list(pneu_dir.glob("*.jpeg")) + list(pneu_dir.glob("*.jpg")))
        bacterial = [f for f in pneu_files if "bacteria" in f.name.lower()]
        viral = [f for f in pneu_files if "virus" in f.name.lower()]
        others = [f for f in pneu_files if f not in bacterial and f not in viral]

        selected_pneu = (bacterial[:2] + viral[:2] + others[:2])[:4]
        for f in selected_pneu:
            sub = "Bacterial Pneumonia" if "bacteria" in f.name.lower() else ("Viral Pneumonia" if "virus" in f.name.lower() else "Pneumonia")
            samples.append({
                "id": f.stem,
                "label": "PNEUMONIA",
                "category": sub,
                "filename": f.name,
                "path": str(f.relative_to(PROJECT_ROOT)).replace("\\", "/")
            })

    return {"samples": samples}


@app.get("/api/image")
def get_image(path: str):
    """Safely serves an image from the project directory."""
    full_path = (PROJECT_ROOT / path).resolve()
    if not full_path.exists() or not str(full_path).startswith(str(PROJECT_ROOT)):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(full_path)


@app.get("/api/pcap/download")
def download_pcap():
    """Direct download endpoint for the Wireshark packet capture deliverable."""
    pcap_path = RESULTS_DIR / "federated_privacy_capture.pcap"
    if not pcap_path.exists():
        raise HTTPException(status_code=404, detail="PCAP file not found")
    return FileResponse(
        pcap_path,
        media_type="application/vnd.tcpdump.pcap",
        filename="federated_privacy_capture.pcap"
    )


@app.get("/api/benchmark/progress")
def get_benchmark_progress():
    """Returns real-time progress for long-running benchmarks (e.g., 4x3 Non-IID Ablation Grid)."""
    import glob, re

    # 1. Check if structured final JSON exists
    ablation_json_path = RESULTS_DIR / "research" / "ablation_4x3_results.json"
    is_finished = ablation_json_path.exists()
    completed_data = []
    if is_finished:
        try:
            with open(ablation_json_path, "r", encoding="utf-8") as f:
                completed_data = json.load(f)
        except Exception:
            pass

    # 2. Look for active or recent task log
    app_data_brain = Path(os.environ.get("USERPROFILE", "C:/Users/quenc")) / ".gemini" / "antigravity-ide" / "brain"
    task_log_pattern = str(app_data_brain / "*" / ".system_generated" / "tasks" / "task-*.log")
    logs = glob.glob(task_log_pattern)
    target_log = None
    if logs:
        logs.sort(key=os.path.getmtime, reverse=True)
        for l in logs[:15]:
            try:
                with open(l, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if "OMNIFED 4x3 NON-IID ABLATION" in content:
                        target_log = l
                        break
            except Exception:
                continue

    current_step = 0
    total_steps = 12
    current_label = "Idle / Ready"
    recent_lines = []
    completed_count = len(completed_data) if (is_finished and completed_data) else 0
    percent = 100.0 if (is_finished and completed_count >= 12) else 0.0
    status = "completed" if (is_finished and completed_count >= 12) else "idle"

    parsed_runs = []

    if target_log:
        try:
            with open(target_log, "r", encoding="utf-8", errors="ignore") as f:
                log_txt = f.read()
            lines = log_txt.splitlines()
            recent_lines = lines[-40:] if len(lines) > 40 else lines

            steps = re.findall(r"\[(\d+)/12\] Running ([^\n]+)", log_txt)
            if steps:
                current_step = int(steps[-1][0])
                current_label = f"Step {current_step}/12: {steps[-1][1].strip()}"
                status = "running"

            # Parse all individual completed entries from log
            step_blocks = re.findall(r"\[(\d+)/12\] Running (\w+) \(mu=([\d\.]+)\) \| alpha=([\d\.]+) \| seed=(\d+)\.\.\.[\s\S]*?Done! Final Acc: ([\d\.]+)% \(Peak: ([\d\.]+)%\) \| Time: ([\d\.]+)s \| Idle: ([\d\.]+)s", log_txt)
            if step_blocks:
                completed_count = len(step_blocks)
                for m in step_blocks:
                    parsed_runs.append({
                        "step": int(m[0]),
                        "strategy": m[1],
                        "mu_proximal": float(m[2]),
                        "alpha_dirichlet": float(m[3]),
                        "seed": int(m[4]),
                        "final_accuracy": float(m[5]),
                        "peak_accuracy": float(m[6]),
                        "simulated_time_sec": float(m[7]),
                        "idle_time_sec": float(m[8])
                    })

            if completed_count >= 12 or is_finished:
                percent = 100.0
                status = "completed"
                current_label = "4x3 Non-IID Ablation Benchmark Complete"
            else:
                percent = round((completed_count / 12.0) * 100.0, 1)
        except Exception as e:
            recent_lines = [f"Error reading log: {e}"]

    # 3. Load 20-node physical cluster results
    physical_results = []
    physical_json_path = RESULTS_DIR / "research" / "physical_20node_results.json"
    if physical_json_path.exists():
        try:
            with open(physical_json_path, "r", encoding="utf-8") as f:
                physical_results = json.load(f)
        except Exception:
            pass

    return {
        "status": status,
        "is_finished": is_finished or percent == 100.0,
        "current_step": current_step,
        "total_steps": total_steps,
        "completed_count": completed_count,
        "percent_complete": percent,
        "current_condition": current_label,
        "completed_data": completed_data if completed_data else parsed_runs,
        "recent_logs": recent_lines,
        "physical_results": physical_results
    }


@app.post("/api/classify")
async def classify_image(
    file: Optional[UploadFile] = File(None),
    sample_path: Optional[str] = Form(None)
):
    """Universal classification and Grad-CAM generation across arbitrary N classes."""
    # Reload weights in case new training just finished
    classifier._load_weights(str(RESULTS_DIR / "best_model.pth"), str(RESULTS_DIR / "global_model.pth"))

    if file:
        contents = await file.read()
        image_stream = io.BytesIO(contents)
        result = classifier.predict(image_stream, generate_gradcam=True)
        result["source"] = f"Uploaded File: {file.filename}"
    elif sample_path:
        full_path = (PROJECT_ROOT / sample_path).resolve()
        if not full_path.exists():
            raise HTTPException(status_code=404, detail=f"Sample not found: {sample_path}")
        result = classifier.predict(str(full_path), generate_gradcam=True)
        result["source"] = f"Dataset Sample: {Path(sample_path).name}"
    else:
        raise HTTPException(status_code=400, detail="Must provide either an uploaded file or a sample_path")

    return JSONResponse(content=result)


@app.get("/", response_class=HTMLResponse)
def index_page():
    return HTML_CONTENT


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OmniFed: Modular Federated Learning Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #070a12;
      --card-bg: rgba(15, 23, 42, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --card-hover: rgba(255, 255, 255, 0.14);
      --accent-blue: #3b82f6;
      --accent-cyan: #06b6d4;
      --accent-green: #10b981;
      --accent-red: #ef4444;
      --accent-amber: #f59e0b;
      --accent-purple: #8b5cf6;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --font-main: 'Outfit', sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background-color: var(--bg-dark);
      background-image: 
        radial-gradient(circle at 10% 10%, rgba(59, 130, 246, 0.12) 0%, transparent 45%),
        radial-gradient(circle at 90% 90%, rgba(139, 92, 246, 0.12) 0%, transparent 45%);
      color: var(--text-main);
      font-family: var(--font-main);
      min-height: 100vh;
      padding: 20px 32px;
      overflow-x: hidden;
    }

    /* Top Navigation Header */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--card-border);
    }

    .brand { display: flex; align-items: center; gap: 14px; }

    .logo-icon {
      font-size: 28px;
      background: linear-gradient(135deg, #3b82f6, #8b5cf6);
      border-radius: 12px;
      width: 46px;
      height: 46px;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 4px 16px rgba(59, 130, 246, 0.35);
    }

    .brand-text h1 {
      font-size: 22px;
      font-weight: 800;
      letter-spacing: -0.5px;
      background: linear-gradient(135deg, #ffffff, #93c5fd);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .brand-text p { font-size: 13px; color: var(--text-muted); }

    /* Suite Nav Tabs */
    .nav-tabs {
      display: flex;
      background: rgba(0, 0, 0, 0.4);
      padding: 4px;
      border-radius: 12px;
      border: 1px solid var(--card-border);
      gap: 4px;
    }

    .suite-tab {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 8px 18px;
      border-radius: 9px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: all 0.2s;
    }

    .suite-tab.active {
      background: var(--accent-blue);
      color: #ffffff;
      box-shadow: 0 2px 10px rgba(59, 130, 246, 0.4);
    }

    /* Content Views */
    .tab-view { display: none; }
    .tab-view.active { display: block; animation: fadeIn 0.3s ease; }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Common Card Styles */
    .glass-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 22px;
      backdrop-filter: blur(16px);
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    }

    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 18px;
    }

    .card-title {
      font-size: 15px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* TAB 1: Orchestration Grid */
    .orchestrate-grid {
      display: grid;
      grid-template-columns: 380px 1fr;
      gap: 24px;
    }

    .form-group {
      margin-bottom: 16px;
    }

    .form-label {
      display: block;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .form-control {
      width: 100%;
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 10px 12px;
      color: var(--text-main);
      font-family: var(--font-main);
      font-size: 14px;
      outline: none;
      transition: border-color 0.2s;
    }

    .form-control:focus {
      border-color: var(--accent-blue);
    }

    .nodes-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }

    .node-status-card {
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .node-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .node-name { font-size: 13px; font-weight: 600; }

    .node-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      font-family: var(--font-mono);
      color: var(--accent-green);
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent-green);
      box-shadow: 0 0 8px var(--accent-green);
    }

    .launch-btn {
      width: 100%;
      padding: 14px;
      background: linear-gradient(135deg, #2563eb, #7c3aed);
      border: none;
      border-radius: 10px;
      color: #ffffff;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      transition: all 0.2s ease;
      box-shadow: 0 4px 16px rgba(37, 99, 235, 0.4);
    }

    .launch-btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6);
    }

    .terminal-window {
      background: #000000;
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 14px;
      height: 480px;
      overflow-y: auto;
      font-family: var(--font-mono);
      font-size: 12px;
      line-height: 1.5;
      color: #93c5fd;
      display: flex;
      flex-direction: column-reverse;
    }

    /* TAB 2: Analytics Suite */
    .analytics-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
    }

    .plot-img {
      width: 100%;
      border-radius: 10px;
      border: 1px solid rgba(255, 255, 255, 0.05);
      background: #000;
    }

    .calc-row {
      display: flex;
      justify-content: space-between;
      padding: 10px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      font-size: 13px;
    }

    /* TAB 4: Long-Run Benchmark Suite */
    .benchmark-grid { display: flex; flex-direction: column; gap: 20px; }
    .bm-hero {
      background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.85));
      border: 1px solid rgba(59, 130, 246, 0.3);
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    .bm-progress-bar-wrap {
      background: rgba(0, 0, 0, 0.5);
      border: 1px solid var(--card-border);
      border-radius: 9999px;
      height: 20px;
      overflow: hidden;
      margin: 14px 0 10px 0;
      position: relative;
    }
    .bm-progress-bar-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #3b82f6, #06b6d4, #10b981);
      border-radius: 9999px;
      transition: width 0.5s ease;
      box-shadow: 0 0 16px rgba(6, 182, 212, 0.6);
    }
    .bm-kpis {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }
    .bm-kpi-card {
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 12px 16px;
    }
    .bm-kpi-val {
      font-size: 20px;
      font-weight: 800;
      font-family: var(--font-mono);
      margin-top: 4px;
    }
    .matrix-row-grid {
      display: grid;
      grid-template-columns: 240px 1fr 1fr 1fr;
      gap: 12px;
      margin-bottom: 12px;
      align-items: stretch;
    }
    .strategy-label-box {
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }
    .cell-card {
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 4px;
      transition: all 0.25s ease;
    }
    .cell-card.completed {
      border-color: #10b981;
      background: rgba(16, 185, 129, 0.08);
    }
    .cell-card.running {
      border-color: #38bdf8;
      background: rgba(56, 189, 248, 0.12);
      box-shadow: 0 0 18px rgba(56, 189, 248, 0.35);
    }
    .cell-card.pending {
      opacity: 0.5;
    }

    /* TAB 3: Diagnostic Studio */
    .diagnostic-grid {
      display: grid;
      grid-template-columns: 320px 1fr 380px;
      gap: 24px;
    }

    .dropzone {
      border: 2px dashed rgba(59, 130, 246, 0.4);
      border-radius: 12px;
      padding: 24px 16px;
      text-align: center;
      cursor: pointer;
      background: rgba(59, 130, 246, 0.03);
      margin-bottom: 16px;
      transition: all 0.2s;
    }

    .dropzone:hover {
      border-color: var(--accent-blue);
      background: rgba(59, 130, 246, 0.08);
    }

    .sample-btn {
      width: 100%;
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 12px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      color: var(--text-main);
      cursor: pointer;
      margin-bottom: 8px;
      font-family: var(--font-main);
      transition: all 0.2s;
    }

    .sample-btn:hover {
      background: rgba(255, 255, 255, 0.08);
      transform: translateX(3px);
    }

    .sample-btn.active {
      border-color: var(--accent-blue);
      background: rgba(59, 130, 246, 0.15);
    }

    .tag-normal { background: rgba(16, 185, 129, 0.2); color: #34d399; padding: 2px 8px; border-radius: 6px; font-size: 10px; font-family: var(--font-mono); }
    .tag-pneu { background: rgba(239, 68, 68, 0.2); color: #f87171; padding: 2px 8px; border-radius: 6px; font-size: 10px; font-family: var(--font-mono); }

    .image-stage {
      background: #000000;
      border-radius: 12px;
      min-height: 420px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }

    .stage-image { max-width: 100%; max-height: 100%; object-fit: contain; }

    .diag-banner {
      padding: 14px;
      border-radius: 10px;
      text-align: center;
      margin-bottom: 14px;
      font-size: 18px;
      font-weight: 800;
      letter-spacing: 0.5px;
    }
    .diag-banner.normal { background: rgba(16, 185, 129, 0.12); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .diag-banner.pneumonia { background: rgba(239, 68, 68, 0.12); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }

    .prob-bar {
      margin-bottom: 10px;
    }
    .prob-meta { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; font-family: var(--font-mono); }
    .prob-track { height: 8px; background: rgba(255, 255, 255, 0.08); border-radius: 9999px; overflow: hidden; }
    .prob-fill { height: 100%; transition: width 0.5s ease; border-radius: 9999px; }
  </style>
</head>
<body>

  <!-- Header & Suite Navigation -->
  <header>
    <div class="brand">
      <div class="logo-icon">⚡</div>
      <div class="brand-text">
        <h1>OmniFed Studio</h1>
        <p>Modular Open-Ended Federated Learning Framework & Cluster Orchestrator</p>
      </div>
    </div>

    <div class="nav-tabs">
      <button class="suite-tab active" onclick="switchSuite('orchestrate')">
        <span>🎛️</span>
        <span>Orchestrate & Train</span>
      </button>
      <button class="suite-tab" onclick="switchSuite('analytics')">
        <span>📊</span>
        <span>Analytics & Benchmarks</span>
      </button>
      <button class="suite-tab" onclick="switchSuite('test')">
        <span>🔬</span>
        <span>Model Testing & Grad-CAM</span>
      </button>
      <button class="suite-tab" onclick="switchSuite('benchmark')">
        <span>⏱️</span>
        <span>Long-Run Benchmark Suite</span>
      </button>
    </div>
  </header>

  <!-- SUITE 1: ORCHESTRATION & TRAINING -->
  <div class="tab-view active" id="view-orchestrate">
    <div class="orchestrate-grid">

      <!-- Left: Experiment Config -->
      <div class="glass-card">
        <div class="card-header">
          <div class="card-title"><span>⚙️</span> Experiment Configuration</div>
        </div>

        <div class="form-group">
          <label class="form-label">Model Architecture</label>
          <select class="form-control" id="cfg-arch" onchange="updateModelInfo()">
            <option value="mobilenet_v3_small">MobileNetV3-Small (~5.8 MB, Edge CPU Optimized)</option>
            <option value="resnet18">ResNet-18 (~42.6 MB, General Vision Benchmark)</option>
            <option value="efficientnet_b0">EfficientNet-B0 (~15.3 MB, High Parameter Efficiency)</option>
          </select>
        </div>

        <div class="form-group">
          <label class="form-label">Target Dataset</label>
          <select class="form-control" id="cfg-dataset">
            <option value="data/global_test/train">Chest Radiographs (Kaggle Pneumonia — 2 Classes)</option>
          </select>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
          <div class="form-group">
            <label class="form-label">Federated Rounds</label>
            <input type="number" class="form-control" id="cfg-rounds" value="6" min="1" max="20">
          </div>
          <div class="form-group">
            <label class="form-label">Local Epochs</label>
            <input type="number" class="form-control" id="cfg-epochs" value="1" min="1" max="5">
          </div>
        </div>

        <div class="form-group">
          <label class="form-label">Aggregation Strategy</label>
          <select class="form-control" id="cfg-strategy">
            <option value="FedAvg">Federated Averaging (FedAvg — McMahan et al.)</option>
            <option value="FedProx">FedProx (Proximal Regularization for Stragglers)</option>
          </select>
        </div>

        <button class="launch-btn" id="btn-launch" onclick="launchTraining()">
          <span>🚀</span>
          <span>Launch Federated Training Across Cluster</span>
        </button>
      </div>

      <!-- Right: Cluster Node Health & Live Console -->
      <div class="glass-card">
        <div class="card-header">
          <div class="card-title"><span>📡</span> Physical Cluster Status (4 Nodes)</div>
          <button style="background: transparent; border: 1px solid var(--card-border); color: var(--text-muted); padding: 4px 10px; border-radius: 6px; font-size: 11px; cursor: pointer;" onclick="fetchClusterHealth()">Refresh Health</button>
        </div>

        <div class="nodes-grid" id="nodes-container">
          <!-- Dynamic Node Cards -->
        </div>

        <div class="card-title" style="margin-bottom: 10px;"><span>📟</span> Live Cluster Execution Console</div>
        <div class="terminal-window" id="terminal-logs">
          <div>[Ready] Cluster idle. Configure experiment parameters and click 'Launch Federated Training'.</div>
        </div>
      </div>

    </div>
  </div>

  <!-- SUITE 2: ANALYTICS & BENCHMARKS -->
  <div class="tab-view" id="view-analytics">
    <div class="analytics-grid">

      <!-- Accuracy Convergence Chart -->
      <div class="glass-card">
        <div class="card-header">
          <div class="card-title"><span>📈</span> Accuracy & Loss Convergence Curve</div>
        </div>
        <img class="plot-img" src="/results/plots/convergence_curve.png" alt="Convergence Curve" onerror="this.style.display='none'">
        <p style="font-size: 12px; color: var(--text-muted); margin-top: 10px;">
          Empirical 6-round learning curve showing rapid convergence from 68.7% to 97.4% test accuracy despite severe institutional Non-IID skew.
        </p>
      </div>

      <!-- Dynamic Bandwidth Calculator -->
      <div class="glass-card">
        <div class="card-header">
          <div class="card-title"><span>⚡</span> Dynamic Network Bandwidth Calculator</div>
        </div>

        <div class="calc-row">
          <span>Raw Image Dataset Baseline:</span>
          <b>1,200.0 MB (1.2 GB)</b>
        </div>
        <div class="calc-row">
          <span>Active Model Weight Payload:</span>
          <b id="calc-payload">5.80 MB (FP32)</b>
        </div>
        <div class="calc-row">
          <span>Total Edge Clients Transmitting:</span>
          <b>3 Remote Cisco UCS Nodes</b>
        </div>
        <div class="calc-row">
          <span>Communication Rounds:</span>
          <b>6 Rounds</b>
        </div>
        <div class="calc-row" style="border-bottom: 2px solid var(--accent-blue); padding: 14px 0;">
          <span style="font-size: 15px; font-weight: 700; color: #93c5fd;">Total Network Data Consumed:</span>
          <b style="font-size: 16px; color: #34d399;" id="calc-total">208.8 MB</b>
        </div>
        <div class="calc-row" style="padding: 14px 0;">
          <span style="font-size: 15px; font-weight: 700;">Bandwidth Conservation:</span>
          <b style="font-size: 18px; color: #34d399;" id="calc-savings">82.6% Reduction</b>
        </div>

        <div style="margin-top: 20px; padding: 14px; background: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 10px; display: flex; justify-content: space-between; align-items: center;">
          <div>
            <div style="font-size: 13px; font-weight: 600;">Wireshark Packet Capture Deliverable</div>
            <div style="font-size: 11px; color: var(--text-muted);">Zero-Data-Leakage verified (Port 8080 gRPC)</div>
          </div>
          <a href="/api/pcap/download" download style="background: var(--accent-blue); color: #fff; text-decoration: none; padding: 8px 14px; border-radius: 6px; font-size: 12px; font-weight: 600;">Download .PCAP</a>
        </div>
      </div>

    </div>
  </div>

  <!-- SUITE 3: DIAGNOSTIC & GRAD-CAM INFERENCE -->
  <div class="tab-view" id="view-test">
    <div class="diagnostic-grid">

      <!-- Column 1: Radiograph Selector -->
      <div class="glass-card">
        <div class="card-title"><span>📁</span> Select Test Radiograph</div>

        <div class="dropzone" onclick="document.getElementById('file-input').click()">
          <input type="file" id="file-input" accept="image/*" style="display: none;" onchange="handleFileUpload(event)">
          <div style="font-size: 28px; margin-bottom: 6px;">📤</div>
          <div style="font-size: 13px; color: var(--text-muted);"><b>Click to upload</b> or drag & drop image</div>
        </div>

        <div class="card-title" style="margin-top: 14px;"><span>🧪</span> Curated Holdout Samples</div>
        <div id="sample-list"></div>
      </div>

      <!-- Column 2: Radiograph & Attention Map Stage -->
      <div class="glass-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
          <div class="card-title" style="margin-bottom: 0;"><span>🔬</span> Inspection Stage</div>
          <div class="nav-tabs" style="padding: 2px;">
            <button class="suite-tab active" id="tab-raw" style="padding: 4px 12px; font-size: 11px;" onclick="setViewerMode('raw')">Original X-Ray</button>
            <button class="suite-tab" id="tab-cam" style="padding: 4px 12px; font-size: 11px;" onclick="setViewerMode('cam')">Grad-CAM Heatmap</button>
          </div>
        </div>

        <div class="image-stage">
          <div id="stage-placeholder" style="text-align: center; color: var(--text-muted);">
            <div style="font-size: 40px; margin-bottom: 8px;">🩻</div>
            <div>Select a sample or upload an image</div>
          </div>
          <img id="stage-img" class="stage-image" style="display: none;" alt="Test Radiograph">
        </div>
      </div>

      <!-- Column 3: Diagnostic Findings & Probability Breakdown -->
      <div class="glass-card">
        <div class="card-title"><span>🩺</span> Clinical Diagnostic Output</div>

        <div class="diag-banner normal" id="diag-banner" style="display: none;">
          <div id="diag-title">NORMAL</div>
          <div style="font-size: 12px; font-family: var(--font-mono); font-weight: normal; margin-top: 4px;" id="diag-conf">Confidence: 96.3%</div>
        </div>

        <div id="prob-container" style="display: none; margin-bottom: 16px;"></div>

        <div class="card-title" style="margin-top: 10px;"><span>📋</span> Radiological Findings</div>
        <div style="background: rgba(0, 0, 0, 0.3); border: 1px solid var(--card-border); border-radius: 10px; padding: 12px; font-size: 13px; line-height: 1.5; color: #cbd5e1; margin-bottom: 16px;" id="clinical-notes">
          Awaiting image evaluation...
        </div>

        <div class="card-title"><span>⚡</span> Hardware Telemetry</div>
        <div style="font-family: var(--font-mono); font-size: 12px;">
          <div class="calc-row"><span>Inference Latency:</span><b id="telem-lat">-- ms</b></div>
          <div class="calc-row"><span>Compute Device:</span><b id="telem-dev">RTX 5070 Ti (CUDA)</b></div>
          <div class="calc-row"><span>Model Architecture:</span><b id="telem-arch">MobileNetV3-Small</b></div>
          <div class="calc-row"><span>Checkpoint Origin:</span><b id="telem-src">results/best_model.pth</b></div>
        </div>
      </div>

    </div>
  </div>

  <!-- SUITE 4: LONG-RUN BENCHMARK SUITE & LIVE MONITOR -->
  <div class="tab-view" id="view-benchmark">
    <div class="benchmark-grid">

      <!-- Hero Progress Card -->
      <div class="glass-card bm-hero">
        <div class="card-header" style="margin-bottom: 8px;">
          <div>
            <div class="card-title" style="color: #93c5fd; font-size: 16px;"><span>⏱️</span> OmniFed 4x3 Non-IID Statistical Ablation Benchmark</div>
            <p style="font-size: 13px; color: var(--text-muted); margin-top: 4px;">
              Systematically evaluates FedAvg vs. FedProx vs. CARC vs. CARC+FedProx across Uniform IID (&alpha;=100.0), Moderate (&alpha;=0.5), and Severe (&alpha;=0.1) Non-IID skew regimes.
            </p>
          </div>
          <div style="display: flex; align-items: center; gap: 10px;">
            <span id="bm-status-badge" style="background: rgba(16, 185, 129, 0.2); color: #34d399; padding: 6px 14px; border-radius: 8px; font-weight: 700; font-size: 12px; font-family: var(--font-mono); border: 1px solid rgba(16, 185, 129, 0.4);">
              🟢 EXECUTING BENCHMARK
            </span>
            <button onclick="fetchBenchmarkProgress(true)" style="background: rgba(59, 130, 246, 0.2); border: 1px solid rgba(59, 130, 246, 0.4); color: #93c5fd; padding: 6px 12px; border-radius: 8px; font-size: 12px; cursor: pointer;">🔄 Refresh</button>
          </div>
        </div>

        <!-- Progress Bar -->
        <div class="bm-progress-bar-wrap">
          <div class="bm-progress-bar-fill" id="bm-progress-bar" style="width: 0%;"></div>
        </div>

        <div style="display: flex; justify-content: space-between; font-size: 12px; font-family: var(--font-mono); color: var(--text-muted); margin-bottom: 12px;">
          <span id="bm-active-cond" style="color: #38bdf8; font-weight: 600;">Detecting active condition...</span>
          <span id="bm-progress-pct" style="color: #34d399; font-weight: 700;">0.0% [0 / 12 Conditions]</span>
        </div>

        <!-- Live KPI Badges -->
        <div class="bm-kpis">
          <div class="bm-kpi-card">
            <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">Ablation Status</div>
            <div class="bm-kpi-val" id="bm-kpi-status" style="color: #38bdf8; font-size: 16px;">Active</div>
          </div>
          <div class="bm-kpi-card">
            <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">Completed Runs</div>
            <div class="bm-kpi-val" id="bm-kpi-completed" style="color: #34d399;">0 / 12</div>
          </div>
          <div class="bm-kpi-card">
            <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">Peak Accuracy</div>
            <div class="bm-kpi-val" id="bm-kpi-peak" style="color: #fbbf24;">--</div>
          </div>
          <div class="bm-kpi-card">
            <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">Physical Fleet Scale</div>
            <div class="bm-kpi-val" style="color: #a78bfa;">20 LXCs (96 Cores)</div>
          </div>
        </div>
      </div>

      <!-- 4x3 Interactive Non-IID Matrix Grid -->
      <div class="glass-card">
        <div class="card-header">
          <div class="card-title"><span>🧪</span> 4x3 Empirical Ablation Matrix Results</div>
          <span style="font-size: 12px; color: var(--text-muted); font-family: var(--font-mono);">Real-Time Parameter Telemetry</span>
        </div>

        <!-- Matrix Header -->
        <div style="display: grid; grid-template-columns: 240px 1fr 1fr 1fr; gap: 12px; margin-bottom: 10px; font-size: 12px; font-weight: 700; color: var(--text-muted); text-align: center;">
          <div style="text-align: left; padding-left: 8px;">Optimization Strategy</div>
          <div style="background: rgba(59, 130, 246, 0.1); padding: 8px; border-radius: 8px; border: 1px solid rgba(59, 130, 246, 0.2);">Uniform IID (&alpha; = 100.0)</div>
          <div style="background: rgba(245, 158, 11, 0.1); padding: 8px; border-radius: 8px; border: 1px solid rgba(245, 158, 11, 0.2);">Moderate Non-IID (&alpha; = 0.5)</div>
          <div style="background: rgba(239, 68, 68, 0.1); padding: 8px; border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.2);">Severe Non-IID (&alpha; = 0.1)</div>
        </div>

        <!-- Matrix Rows -->
        <div id="matrix-rows-container">
          <!-- Dynamically populated rows: FEDAVG, FEDPROX, CARC, CARC_FEDPROX -->
        </div>
      </div>

      <!-- Bottom Split: Physical Bare-Metal Validation & Live Benchmark Console -->
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        
        <!-- Physical 20-Node Scaling Results Card -->
        <div class="glass-card">
          <div class="card-header">
            <div class="card-title"><span>🚀</span> 20-Node Physical Bare-Metal Scaling Matrix</div>
            <span style="background: rgba(168, 85, 247, 0.2); color: #c084fc; font-size: 11px; padding: 3px 8px; border-radius: 6px; font-family: var(--font-mono);">100% Empirical</span>
          </div>

          <table style="width: 100%; border-collapse: collapse; font-size: 12.5px; margin-top: 6px;">
            <thead>
              <tr style="border-bottom: 1px solid var(--card-border); color: var(--text-muted); text-align: left;">
                <th style="padding: 8px;">Scale</th>
                <th style="padding: 8px;">Physical Nodes</th>
                <th style="padding: 8px;">Round Time</th>
                <th style="padding: 8px;">Throughput</th>
                <th style="padding: 8px;">Speedup</th>
              </tr>
            </thead>
            <tbody id="physical-scale-body">
              <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                <td style="padding: 8px; font-weight: 700;">N = 3</td>
                <td style="padding: 8px; color: #94a3b8;">3 LXCs (1/pve)</td>
                <td style="padding: 8px; font-family: var(--font-mono);">33.34 s</td>
                <td style="padding: 8px; font-family: var(--font-mono); color: #38bdf8;">13.50 samp/s</td>
                <td style="padding: 8px; color: #94a3b8;">0.98x (Parity)</td>
              </tr>
              <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); background: rgba(59, 130, 246, 0.08);">
                <td style="padding: 8px; font-weight: 700; color: #38bdf8;">N = 5</td>
                <td style="padding: 8px; color: #94a3b8;">5 LXCs (3 hosts)</td>
                <td style="padding: 8px; font-family: var(--font-mono);">41.86 s</td>
                <td style="padding: 8px; font-family: var(--font-mono); color: #34d399; font-weight: 700;">17.92 samp/s</td>
                <td style="padding: 8px; color: #34d399; font-weight: 700;">⚡ Crossover Point</td>
              </tr>
              <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                <td style="padding: 8px; font-weight: 700;">N = 10</td>
                <td style="padding: 8px; color: #94a3b8;">10 LXCs (3 hosts)</td>
                <td style="padding: 8px; font-family: var(--font-mono);">49.01 s</td>
                <td style="padding: 8px; font-family: var(--font-mono); color: #34d399;">30.61 samp/s</td>
                <td style="padding: 8px; color: #34d399;">2.21x vs GPU</td>
              </tr>
              <tr style="background: rgba(16, 185, 129, 0.08);">
                <td style="padding: 8px; font-weight: 700; color: #34d399;">N = 20</td>
                <td style="padding: 8px; color: #94a3b8;">20 LXCs (Full Cluster)</td>
                <td style="padding: 8px; font-family: var(--font-mono);">57.45 s</td>
                <td style="padding: 8px; font-family: var(--font-mono); color: #34d399; font-weight: 800;">52.21 samp/s</td>
                <td style="padding: 8px; color: #34d399; font-weight: 800;">🚀 3.77x vs GPU</td>
              </tr>
            </tbody>
          </table>
          <div style="font-size: 11.5px; color: var(--text-muted); margin-top: 10px;">
            Centralized RTX 5070 Ti Baseline: <b style="color: #93c5fd; font-family: var(--font-mono);">13.84 samp/s</b>. Zero CPU overcommitment on 3 Cisco UCS hosts.
          </div>
        </div>

        <!-- Live Streaming Log Window -->
        <div class="glass-card">
          <div class="card-header">
            <div class="card-title"><span>📟</span> Live Benchmark Log Stream</div>
            <span style="font-size: 11px; color: var(--text-muted); font-family: var(--font-mono);" id="bm-log-count">Streaming</span>
          </div>
          <div class="terminal-window" id="bm-terminal" style="height: 220px; overflow-y: auto;">
            <div>[Connecting to benchmark log stream...]</div>
          </div>
        </div>

      </div>

    </div>
  </div>

  <script>
    let currentRawSrc = null;
    let currentCamSrc = null;
    let currentViewerMode = 'raw';
    let logOffset = 0;
    let pollTimer = null;

    window.addEventListener('DOMContentLoaded', () => {
      fetchClusterHealth();
      fetchSamples();
      fetchBenchmarkProgress();
      setInterval(pollTrainingLogs, 2000);
      setInterval(fetchBenchmarkProgress, 2500);
    });

    function switchSuite(tabName) {
      document.querySelectorAll('.suite-tab').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-view').forEach(v => v.classList.remove('active'));

      if (tabName === 'orchestrate') {
        document.querySelectorAll('.suite-tab')[0].classList.add('active');
        document.getElementById('view-orchestrate').classList.add('active');
      } else if (tabName === 'analytics') {
        document.querySelectorAll('.suite-tab')[1].classList.add('active');
        document.getElementById('view-analytics').classList.add('active');
      } else if (tabName === 'test') {
        document.querySelectorAll('.suite-tab')[2].classList.add('active');
        document.getElementById('view-test').classList.add('active');
      } else if (tabName === 'benchmark') {
        document.querySelectorAll('.suite-tab')[3].classList.add('active');
        document.getElementById('view-benchmark').classList.add('active');
        fetchBenchmarkProgress(true);
      }
    }

    async function fetchBenchmarkProgress(forceManual = false) {
      try {
        const res = await fetch('/api/benchmark/progress');
        const data = await res.json();
        
        // 1. Progress Bar & Hero Stats
        const pBar = document.getElementById('bm-progress-bar');
        if (pBar) pBar.style.width = data.percent_complete + '%';
        
        const pText = document.getElementById('bm-progress-pct');
        if (pText) pText.innerText = `${data.percent_complete}% [${data.completed_count} / ${data.total_steps} Conditions]`;
        
        const activeCond = document.getElementById('bm-active-cond');
        if (activeCond) activeCond.innerText = data.current_condition || 'Idle';
        
        const kpiStatus = document.getElementById('bm-kpi-status');
        if (kpiStatus) {
          kpiStatus.innerText = data.status.toUpperCase();
          kpiStatus.style.color = data.status === 'completed' ? '#34d399' : (data.status === 'running' ? '#38bdf8' : '#94a3b8');
        }

        const kpiCompleted = document.getElementById('bm-kpi-completed');
        if (kpiCompleted) kpiCompleted.innerText = `${data.completed_count} / ${data.total_steps}`;

        const badge = document.getElementById('bm-status-badge');
        if (badge) {
          if (data.status === 'completed') {
            badge.innerText = '✅ BENCHMARK COMPLETED';
            badge.style.background = 'rgba(16, 185, 129, 0.2)';
            badge.style.color = '#34d399';
            badge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
          } else if (data.status === 'running') {
            badge.innerText = `🟢 EXECUTING (STEP ${data.current_step}/${data.total_steps})`;
            badge.style.background = 'rgba(56, 189, 248, 0.2)';
            badge.style.color = '#38bdf8';
            badge.style.borderColor = 'rgba(56, 189, 248, 0.4)';
          } else {
            badge.innerText = '⏸️ IDLE / READY';
            badge.style.background = 'rgba(148, 163, 184, 0.2)';
            badge.style.color = '#94a3b8';
            badge.style.borderColor = 'rgba(148, 163, 184, 0.4)';
          }
        }

        // 2. Render 4x3 Matrix Grid
        renderAblationMatrix(data.completed_data, data.current_step, data.status === 'running');

        // 3. Render Terminal Output
        const term = document.getElementById('bm-terminal');
        if (term && data.recent_logs) {
          term.innerHTML = '';
          data.recent_logs.forEach(line => {
            const el = document.createElement('div');
            el.innerText = line;
            term.appendChild(el);
          });
          term.scrollTop = term.scrollHeight;
        }

        // Calculate peak accuracy observed
        if (data.completed_data && data.completed_data.length > 0) {
          const maxAcc = Math.max(...data.completed_data.map(d => d.peak_accuracy || d.final_accuracy || 0));
          const kpiPeak = document.getElementById('bm-kpi-peak');
          if (kpiPeak && maxAcc > 0) kpiPeak.innerText = maxAcc.toFixed(1) + '%';
        }
      } catch (err) {
        console.error('Benchmark progress fetch error:', err);
      }
    }

    function renderAblationMatrix(completedList, currentStep, isRunning) {
      const container = document.getElementById('matrix-rows-container');
      if (!container) return;

      const strategies = [
        { key: 'FEDAVG', name: 'Vanilla FedAvg', mu: '0.0', desc: 'Uniform local steps' },
        { key: 'FEDPROX', name: 'FedProx', mu: '0.01', desc: 'Uniform steps + Proximal drift penalty' },
        { key: 'CARC', name: 'OmniFed-CARC', mu: '0.0', desc: 'Capability-adaptive step scaling' },
        { key: 'CARC_FEDPROX', name: 'CARC + FedProx', mu: '0.01', desc: 'Adaptive steps + Proximal penalty' }
      ];
      const alphas = [100.0, 0.5, 0.1];

      // Mapping of cell key to completed run
      const map = {};
      if (completedList) {
        completedList.forEach(r => {
          const k = `${r.strategy}_${r.alpha_dirichlet || r.alpha}`;
          map[k] = r;
        });
      }

      // Determine active cell based on current step (1 to 12)
      // Step order: [1..4 on alpha 100], [5..8 on alpha 0.5], [9..12 on alpha 0.1]
      let activeCellKey = null;
      if (isRunning && currentStep >= 1 && currentStep <= 12) {
        const stratIdx = (currentStep - 1) % 4;
        const alphaIdx = Math.floor((currentStep - 1) / 4);
        activeCellKey = `${strategies[stratIdx].key}_${alphas[alphaIdx]}`;
      }

      container.innerHTML = '';

      strategies.forEach((strat, sIdx) => {
        const row = document.createElement('div');
        row.className = 'matrix-row-grid';

        // Left Strategy Info Box
        const info = document.createElement('div');
        info.className = 'strategy-label-box';
        info.innerHTML = `
          <div style="font-weight: 700; font-size: 13.5px; color: #f8fafc;">${strat.name}</div>
          <div style="font-size: 11px; color: #94a3b8; margin-top: 2px;">${strat.desc}</div>
          <div style="font-size: 10.5px; color: #38bdf8; font-family: var(--font-mono); margin-top: 4px;">&mu; = ${strat.mu}</div>
        `;
        row.appendChild(info);

        // 3 Alpha Columns
        alphas.forEach(a => {
          const cellKey = `${strat.key}_${a}`;
          const rData = map[cellKey];
          const isActive = (cellKey === activeCellKey);

          const cell = document.createElement('div');
          if (rData) {
            cell.className = 'cell-card completed';
            const finalAcc = rData.final_accuracy || rData.final_acc || 0.0;
            const peakAcc = rData.peak_accuracy || rData.peak_acc || finalAcc;
            const timeSec = rData.simulated_time_sec || rData.time_sec || 0.0;
            const idleSec = rData.idle_time_sec || rData.idle_sec || 0.0;

            cell.innerHTML = `
              <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 10px; font-weight: 700; color: #34d399; font-family: var(--font-mono);">✅ COMPLETED</span>
                <span style="font-size: 11px; font-family: var(--font-mono); color: var(--text-muted);">${timeSec}s</span>
              </div>
              <div style="font-size: 18px; font-weight: 800; font-family: var(--font-mono); color: ${finalAcc > 75 ? '#34d399' : (finalAcc > 60 ? '#fbbf24' : '#f87171')};">
                ${finalAcc.toFixed(1)}% <span style="font-size: 11px; font-weight: normal; color: var(--text-muted);">(Peak: ${peakAcc.toFixed(1)}%)</span>
              </div>
              <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--text-muted); margin-top: 2px;">
                <span>Idle Wait: <b style="color: ${idleSec < 40 ? '#34d399' : '#f87171'}">${idleSec}s</b></span>
                <span>Loss: <b style="color: #cbd5e1;">${(rData.final_loss || 0).toFixed(3)}</b></span>
              </div>
            `;
          } else if (isActive) {
            cell.className = 'cell-card running';
            cell.innerHTML = `
              <div style="font-size: 10px; font-weight: 700; color: #38bdf8; font-family: var(--font-mono);">🟢 EXECUTING NOW</div>
              <div style="font-size: 14px; font-weight: 700; color: #38bdf8; margin: 4px 0;">Training Local Steps...</div>
              <div style="font-size: 11px; color: #94a3b8;">Computing gradient iterations across 4 clients</div>
            `;
          } else {
            cell.className = 'cell-card pending';
            cell.innerHTML = `
              <div style="font-size: 10px; font-weight: 600; color: #64748b; font-family: var(--font-mono);">⏳ PENDING</div>
              <div style="font-size: 13px; color: #64748b; margin: 6px 0;">Awaiting Queue</div>
              <div style="font-size: 10.5px; color: #475569;">${strat.name} on &alpha;=${a}</div>
            `;
          }
          row.appendChild(cell);
        });

        container.appendChild(row);
      });
    }

    async function fetchClusterHealth() {
      try {
        const res = await fetch('/api/cluster/health');
        const data = await res.json();
        const container = document.getElementById('nodes-container');
        container.innerHTML = '';

        data.nodes.forEach(node => {
          const card = document.createElement('div');
          card.className = 'node-status-card';
          const isOnline = node.status === 'online';
          const dotStyle = isOnline ? 'background: #10b981;' : 'background: #ef4444;';

          card.innerHTML = `
            <div class="node-header">
              <span class="node-name">${node.name}</span>
              <span class="node-badge" style="color: ${isOnline ? '#34d399' : '#f87171'}">
                <span class="status-dot" style="${dotStyle}"></span>
                ${isOnline ? node.latency_ms + ' ms' : 'Offline'}
              </span>
            </div>
            <div style="font-size: 11px; color: var(--text-muted); font-family: var(--font-mono);">${node.ip} (${node.container_id})</div>
            <div style="font-size: 11px; color: #94a3b8;">${node.hardware}</div>
          `;
          container.appendChild(card);
        });
      } catch (err) {
        console.error('Cluster health fetch error:', err);
      }
    }

    async function launchTraining() {
      const rounds = document.getElementById('cfg-rounds').value;
      const epochs = document.getElementById('cfg-epochs').value;
      const arch = document.getElementById('cfg-arch').value;

      const formData = new FormData();
      formData.append('rounds', rounds);
      formData.append('epochs', epochs);
      formData.append('architecture', arch);

      const btn = document.getElementById('btn-launch');
      btn.disabled = true;
      btn.style.opacity = '0.6';
      btn.innerText = '⏳ Training In Progress Across Cluster...';

      try {
        const res = await fetch('/api/train/start', { method: 'POST', body: formData });
        const data = await res.json();
        if (!data.success) alert(data.message);
      } catch (err) {
        alert('Launch error: ' + err);
      }
    }

    async function pollTrainingLogs() {
      try {
        const res = await fetch(`/api/train/logs?offset=${logOffset}`);
        const data = await res.json();
        if (data.logs && data.logs.length > 0) {
          const term = document.getElementById('terminal-logs');
          data.logs.forEach(line => {
            const el = document.createElement('div');
            el.innerText = line;
            term.prepend(el);
          });
          logOffset = data.next_offset;
        }

        // Check training status
        const sRes = await fetch('/api/train/status');
        const sData = await sRes.json();
        const btn = document.getElementById('btn-launch');
        if (sData.status === 'running') {
          btn.disabled = true;
          btn.style.opacity = '0.6';
          btn.innerText = `⏳ Training Running (Round ${sData.current_round} / ${sData.total_rounds})...`;
        } else {
          btn.disabled = false;
          btn.style.opacity = '1';
          btn.innerHTML = `<span>🚀</span><span>Launch Federated Training Across Cluster</span>`;
        }
      } catch (err) {
        // quiet poll fail
      }
    }

    async function fetchSamples() {
      try {
        const res = await fetch('/api/samples');
        const data = await res.json();
        const list = document.getElementById('sample-list');
        list.innerHTML = '';

        data.samples.forEach((sample, idx) => {
          const btn = document.createElement('button');
          btn.className = 'sample-btn';
          btn.onclick = () => loadSample(sample.path, btn);

          const isNormal = sample.label === 'NORMAL';
          btn.innerHTML = `
            <div>
              <div style="font-size: 13px; font-weight: 500;">${sample.category}</div>
              <div style="font-size: 11px; color: var(--text-muted); font-family: var(--font-mono);">${sample.filename}</div>
            </div>
            <span class="${isNormal ? 'tag-normal' : 'tag-pneu'}">${sample.label}</span>
          `;
          list.appendChild(btn);

          if (idx === 0) btn.click();
        });
      } catch (err) {
        console.error('Sample fetch error:', err);
      }
    }

    async function loadSample(samplePath, btnElem) {
      document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));
      if (btnElem) btnElem.classList.add('active');

      const formData = new FormData();
      formData.append('sample_path', samplePath);

      try {
        const res = await fetch('/api/classify', { method: 'POST', body: formData });
        const data = await res.json();
        renderInference(data);
      } catch (err) {
        alert('Inference error: ' + err);
      }
    }

    async function handleFileUpload(event) {
      const file = event.target.files[0];
      if (!file) return;

      document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));

      const formData = new FormData();
      formData.append('file', file);

      try {
        const res = await fetch('/api/classify', { method: 'POST', body: formData });
        const data = await res.json();
        renderInference(data);
      } catch (err) {
        alert('Upload inference error: ' + err);
      }
    }

    function renderInference(data) {
      currentRawSrc = data.image_preview;
      currentCamSrc = data.heatmap_preview || data.image_preview;

      document.getElementById('stage-placeholder').style.display = 'none';
      const img = document.getElementById('stage-img');
      img.style.display = 'block';
      img.src = currentViewerMode === 'raw' ? currentRawSrc : currentCamSrc;

      // Banner
      const banner = document.getElementById('diag-banner');
      banner.style.display = 'block';
      const isNormal = data.prediction.toUpperCase() === 'NORMAL';
      banner.className = `diag-banner ${isNormal ? 'normal' : 'pneumonia'}`;
      document.getElementById('diag-title').innerText = isNormal ? '✅ NORMAL (CLEAR)' : '⚠️ PNEUMONIA DETECTED';
      document.getElementById('diag-conf').innerText = `Confidence Score: ${data.confidence}%`;

      // Multi-class Bars
      const probContainer = document.getElementById('prob-container');
      probContainer.style.display = 'block';
      probContainer.innerHTML = '';

      for (const [clsName, pct] of Object.entries(data.probabilities)) {
        const isNorm = clsName.toUpperCase() === 'NORMAL';
        const color = isNorm ? '#10b981' : '#ef4444';
        const row = document.createElement('div');
        row.className = 'prob-bar';
        row.innerHTML = `
          <div class="prob-meta">
            <span style="color: ${color}; font-weight: 600;">${clsName}</span>
            <span>${pct}%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill" style="width: ${pct}%; background: ${color};"></div>
          </div>
        `;
        probContainer.appendChild(row);
      }

      document.getElementById('clinical-notes').innerText = data.clinical_notes;
      document.getElementById('telem-lat').innerText = `${data.latency_ms} ms`;
      document.getElementById('telem-dev').innerText = data.device;
      document.getElementById('telem-arch').innerText = data.architecture;
      document.getElementById('telem-src').innerText = data.weights_source.split('/').pop().split('\\\\').pop();
    }

    function setViewerMode(mode) {
      currentViewerMode = mode;
      document.getElementById('tab-raw').className = mode === 'raw' ? 'suite-tab active' : 'suite-tab';
      document.getElementById('tab-cam').className = mode === 'cam' ? 'suite-tab active' : 'suite-tab';

      const img = document.getElementById('stage-img');
      if (img.style.display !== 'none') {
        img.src = mode === 'raw' ? currentRawSrc : currentCamSrc;
      }
    }

    function updateModelInfo() {
      const arch = document.getElementById('cfg-arch').value;
      const payloadMap = {
        'mobilenet_v3_small': { payload: '5.80 MB (FP32)', total: '208.8 MB', savings: '82.6% Reduction' },
        'resnet18': { payload: '42.64 MB (FP32)', total: '1,535.0 MB', savings: 'Overhead on WAN' },
        'efficientnet_b0': { payload: '15.30 MB (FP32)', total: '550.8 MB', savings: '54.1% Reduction' }
      };
      const info = payloadMap[arch] || payloadMap['mobilenet_v3_small'];
      document.getElementById('calc-payload').innerText = info.payload;
      document.getElementById('calc-total').innerText = info.total;
      document.getElementById('calc-savings').innerText = info.savings;
    }
  </script>
</body>
</html>
"""

def main():
    print("==================================================================")
    print("   OMNIFED: MODULAR FEDERATED LEARNING STUDIO                     ")
    print("==================================================================")
    print("Starting Web Studio Server at: http://localhost:5000\n")
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")


if __name__ == "__main__":
    main()
