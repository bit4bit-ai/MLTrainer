import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import DATA_DIR, EXPORTS_DIR, CORS_ORIGINS, SUPPORTED_TASKS
from .core.dataset_manager import dataset_manager
from .core.split_manager import SplitManager
from .core.trainer import trainer
from .core.evaluator import evaluator
from .core.exporter import exporter

app = FastAPI(title="MLTrainer Backend API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class SplitRequest(BaseModel):
    train_pct: float = 70.0
    val_pct: float = 15.0
    test_pct: float = 15.0
    stratify: bool = True
    seed: int = 42

class ReassignRequest(BaseModel):
    item_id: str
    new_split: str

class TrainStartRequest(BaseModel):
    task: str
    config: Dict[str, Any]

class UpdateAnnotationRequest(BaseModel):
    label: Optional[str] = None
    boxes: Optional[list] = None
    segmentation: Optional[list] = None
    is_anomaly: Optional[bool] = None
    anomaly_score: Optional[float] = None
    split: Optional[str] = None

class ExportRequest(BaseModel):
    format: str = "onnx"
    quantization: str = "fp32"

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "MLTrainer Vision Engine",
        "supported_tasks": SUPPORTED_TASKS
    }

# ==================== DATASET ENDPOINTS ====================
@app.get("/api/dataset/{task}")
def get_dataset(task: str):
    if task not in SUPPORTED_TASKS:
        raise HTTPException(status_code=400, detail=f"Unsupported task: {task}")
    return dataset_manager.get_dataset(task)

@app.get("/api/dataset/{task}/distribution")
def get_distribution(task: str):
    if task not in SUPPORTED_TASKS:
        raise HTTPException(status_code=400, detail=f"Unsupported task: {task}")
    return SplitManager.get_split_distribution(task)

@app.post("/api/dataset/{task}/upload")
async def upload_dataset_images(task: str, file: UploadFile = File(...), split: str = Form("train"), label: Optional[str] = Form(None)):
    if task not in SUPPORTED_TASKS:
        raise HTTPException(status_code=400, detail="Invalid task")
    
    task_dir = DATA_DIR / task
    task_dir.mkdir(parents=True, exist_ok=True)

    filename = file.filename
    file_path = task_dir / filename

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Check if uploaded file is a ZIP archive
    if filename.lower().endswith(".zip"):
        unzipped_items = []
        with zipfile.ZipFile(file_path, "r") as zf:
            for zip_info in zf.infolist():
                if zip_info.filename.startswith("__MACOSX") or zip_info.is_dir():
                    continue
                ext = Path(zip_info.filename).suffix.lower()
                if ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
                    extracted_path = zf.extract(zip_info, path=task_dir)
                    item = dataset_manager.add_image_item(
                        task=task,
                        file_path=str(extracted_path),
                        filename=Path(extracted_path).name,
                        width=320,
                        height=320,
                        split=split,
                        label=label or dataset_manager.get_dataset(task)["classes"][0]
                    )
                    unzipped_items.append(item)
        file_path.unlink()  # remove zip
        return {"status": "success", "message": f"Extracted {len(unzipped_items)} images", "items": unzipped_items}

    # Single Image
    item = dataset_manager.add_image_item(
        task=task,
        file_path=str(file_path),
        filename=filename,
        width=320,
        height=320,
        split=split,
        label=label or dataset_manager.get_dataset(task)["classes"][0]
    )
    return {"status": "success", "item": item}

@app.post("/api/dataset/{task}/upload-batch")
async def upload_batch_images(
    task: str,
    files: List[UploadFile] = File(...),
    split: str = Form("train"),
    label: Optional[str] = Form(None)
):
    if task not in SUPPORTED_TASKS:
        raise HTTPException(status_code=400, detail="Invalid task")
    task_dir = DATA_DIR / task
    task_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    added_items = []

    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext in valid_exts:
            clean_name = Path(f.filename).name
            file_path = task_dir / clean_name
            with open(file_path, "wb") as out_file:
                shutil.copyfileobj(f.file, out_file)
            item = dataset_manager.add_image_item(
                task=task,
                file_path=str(file_path),
                filename=clean_name,
                width=320,
                height=320,
                split=split,
                label=label or dataset_manager.get_dataset(task)["classes"][0]
            )
            added_items.append(item)

    return {"status": "success", "count": len(added_items), "items": added_items}

class LocalFolderImportRequest(BaseModel):
    folder_path: str
    split: str = "train"
    label: Optional[str] = None

@app.post("/api/dataset/{task}/import-folder")
def import_local_folder(task: str, req: LocalFolderImportRequest):
    if task not in SUPPORTED_TASKS:
        raise HTTPException(status_code=400, detail="Invalid task")
    src_dir = Path(req.folder_path)
    if not src_dir.exists() or not src_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Folder not found: {req.folder_path}")

    task_dir = DATA_DIR / task
    task_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    imported = []
    for root, _, files in os.walk(src_dir):
        for fname in files:
            if Path(fname).suffix.lower() in valid_exts:
                src_file = Path(root) / fname
                dst_file = task_dir / fname
                shutil.copy2(src_file, dst_file)

                # Check for class subfolder name if label not provided
                parent_folder = Path(root).name
                default_classes = dataset_manager.get_dataset(task)["classes"]
                item_label = req.label or (parent_folder if parent_folder in default_classes else default_classes[0])

                item = dataset_manager.add_image_item(
                    task=task,
                    file_path=str(dst_file),
                    filename=fname,
                    width=320,
                    height=320,
                    split=req.split,
                    label=item_label
                )
                imported.append(item)

    dataset_manager.save_metadata()
    return {"status": "success", "count": len(imported), "items": imported}

@app.put("/api/dataset/{task}/items/{item_id}")
def update_item(task: str, item_id: str, req: UpdateAnnotationRequest):
    updated = dataset_manager.update_item_annotation(task, item_id, req.model_dump(exclude_unset=True))
    if not updated:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"status": "success", "item": updated}

@app.delete("/api/dataset/{task}/items/{item_id}")
def delete_item(task: str, item_id: str):
    success = dataset_manager.delete_item(task, item_id)
    if not success:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"status": "success", "message": "Item deleted"}

class BatchDeleteRequest(BaseModel):
    item_ids: list[str]

@app.post("/api/dataset/{task}/batch-delete")
def batch_delete_items(task: str, req: BatchDeleteRequest):
    if task not in SUPPORTED_TASKS:
        raise HTTPException(status_code=400, detail="Invalid task")
    count = dataset_manager.delete_items(task, req.item_ids)
    return {"status": "success", "deleted_count": count}

# ==================== SPLITTING ENDPOINTS ====================
@app.post("/api/split/{task}")
def split_dataset(task: str, req: SplitRequest):
    if task not in SUPPORTED_TASKS:
        raise HTTPException(status_code=400, detail="Invalid task")
    res = SplitManager.split_dataset(
        task=task,
        train_pct=req.train_pct,
        val_pct=req.val_pct,
        test_pct=req.test_pct,
        stratify=req.stratify,
        seed=req.seed
    )
    return res

@app.post("/api/split/{task}/reassign")
def reassign_split(task: str, req: ReassignRequest):
    success = SplitManager.reassign_item(task, req.item_id, req.new_split)
    if not success:
        raise HTTPException(status_code=404, detail="Item not found")
    return SplitManager.get_split_distribution(task)

# ==================== TRAINING ENDPOINTS ====================
@app.get("/api/training/status")
def get_training_status():
    return trainer.get_status()

@app.post("/api/training/start")
async def start_training(req: TrainStartRequest):
    res = await trainer.start(req.task, req.config)
    return res

@app.post("/api/training/pause")
def pause_training():
    return trainer.pause()

@app.post("/api/training/resume")
def resume_training():
    return trainer.resume()

@app.post("/api/training/stop")
def stop_training():
    return trainer.stop()

@app.websocket("/ws/training")
async def websocket_training(websocket: WebSocket):
    await websocket.accept()
    q = trainer.subscribe()
    try:
        # Send initial status
        await websocket.send_json({"type": "initial_status", "data": trainer.get_status()})
        while True:
            msg = await q.get()
            await websocket.send_json(msg)
    except WebSocketDisconnect:
        trainer.unsubscribe(q)
    except Exception:
        trainer.unsubscribe(q)

# ==================== EVALUATION ENDPOINTS ====================
@app.get("/api/evaluation/{task}")
def get_evaluation(task: str):
    if task not in SUPPORTED_TASKS:
        raise HTTPException(status_code=400, detail="Invalid task")
    return evaluator.get_evaluation_metrics(task)

# ==================== EXPORT ENDPOINTS ====================
@app.get("/api/export/code/{task}")
def get_inference_code(task: str, format: str = "onnx"):
    code = exporter.generate_inference_code(task, format)
    snippet = exporter.generate_fastapi_snippet(task)
    return {"inference_code": code, "fastapi_snippet": snippet}

@app.post("/api/export/bundle/{task}")
def export_bundle(task: str, req: ExportRequest):
    res = exporter.create_export_bundle(task, req.format, req.quantization)
    return res

@app.get("/api/export/download/{filename}")
def download_export(filename: str):
    file_path = EXPORTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Export file not found")
    return FileResponse(file_path, filename=filename, media_type="application/zip")

# ==================== STATIC IMAGE SERVING ====================
@app.get("/api/images/{task}/{filename}")
def serve_image(task: str, filename: str):
    path = DATA_DIR / task / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    media_type = "image/png"
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
        media_type = "image/jpeg"
    return FileResponse(path, media_type=media_type)

# ==================== FRONTEND STATIC MOUNT (HF Space / Docker) ====================
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
