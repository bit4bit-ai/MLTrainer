import os
import json
import uuid
import shutil
import zipfile
import math
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image, ImageDraw, ImageFilter
from ..config import DATA_DIR, DEMO_DATA_DIR

class DatasetManager:
    def __init__(self):
        self.metadata_file = DATA_DIR / "dataset_metadata.json"
        self.datasets: Dict[str, Dict[str, Any]] = {
            "classification": {"name": "Industrial Component Defect Classification", "classes": ["Normal", "Scratch", "Dent", "Void"], "items": []},
            "detection": {"name": "PCB Electronic Component Detection", "classes": ["Resistor", "Capacitor", "IC_Chip", "Diode"], "items": []},
            "anomaly": {"name": "Metal Surface Flaw Anomaly Inspection", "classes": ["Normal", "Anomaly"], "items": []},
            "segmentation": {"name": "Precision Part Defect Segmentation", "classes": ["Background", "Defect_Core", "Surface_Flaw", "Boundary_Void"], "items": []},
        }
        self._last_mtime = 0
        self.load_metadata()
        self.ensure_demo_datasets()

    def load_metadata(self):
        if self.metadata_file.exists():
            try:
                mtime = self.metadata_file.stat().st_mtime
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    for task, data in saved.items():
                        if task in self.datasets:
                            self.datasets[task] = data
                self._last_mtime = mtime
            except Exception as e:
                print(f"Error loading metadata: {e}")

    def save_metadata(self):
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.datasets, f, indent=2)
            if self.metadata_file.exists():
                self._last_mtime = self.metadata_file.stat().st_mtime
        except Exception as e:
            print(f"Error saving metadata: {e}")

    def get_dataset(self, task: str) -> Dict[str, Any]:
        if task not in self.datasets:
            raise ValueError(f"Unsupported task: {task}")
        # Always sync with disk if modified externally
        if self.metadata_file.exists():
            try:
                current_mtime = self.metadata_file.stat().st_mtime
                if current_mtime > self._last_mtime:
                    self.load_metadata()
            except Exception:
                pass
        return self.datasets[task]

    def update_item_annotation(self, task: str, item_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ds = self.get_dataset(task)
        for item in ds["items"]:
            if item["id"] == item_id:
                for k, v in updates.items():
                    if k in ["label", "boxes", "segmentation", "anomaly_score", "is_anomaly", "split"]:
                        item[k] = v
                self.save_metadata()
                return item
        return None

    def delete_item(self, task: str, item_id: str) -> bool:
        ds = self.get_dataset(task)
        item_to_remove = None
        for item in ds["items"]:
            if item["id"] == item_id:
                item_to_remove = item
                break
        if item_to_remove:
            file_path = Path(item_to_remove["file_path"])
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception:
                    pass
            ds["items"].remove(item_to_remove)
            self.save_metadata()
    def delete_items(self, task: str, item_ids: List[str]) -> int:
        ds = self.get_dataset(task)
        id_set = set(item_ids)
        remaining = []
        deleted_count = 0
        for item in ds["items"]:
            if item["id"] in id_set:
                file_path = Path(item["file_path"])
                if file_path.exists():
                    try:
                        file_path.unlink()
                    except Exception:
                        pass
                deleted_count += 1
            else:
                remaining.append(item)
        ds["items"] = remaining
        self.save_metadata()
        return deleted_count

    def add_image_item(self, task: str, file_path: str, filename: str, width: int, height: int, split: str = "train", **kwargs) -> Dict[str, Any]:
        item_id = str(uuid.uuid4())[:8]
        ds = self.get_dataset(task)
        
        # default annotation based on task
        default_label = kwargs.get("label", ds["classes"][0])
        boxes = kwargs.get("boxes", [])
        segmentation = kwargs.get("segmentation", [])
        is_anomaly = kwargs.get("is_anomaly", False)
        anomaly_score = kwargs.get("anomaly_score", 0.0)

        item = {
            "id": item_id,
            "filename": filename,
            "file_path": file_path,
            "url": f"/api/images/{task}/{Path(file_path).name}",
            "width": width,
            "height": height,
            "split": split,
            "label": default_label,
            "boxes": boxes,
            "segmentation": segmentation,
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "heatmap_url": kwargs.get("heatmap_url", None)
        }
        ds["items"].append(item)
        self.save_metadata()
        return item

    def ensure_demo_datasets(self):
        """Generates rich, visual demo datasets for each task if empty."""
        for task in ["classification", "detection", "anomaly", "segmentation"]:
            task_dir = DATA_DIR / task
            task_dir.mkdir(parents=True, exist_ok=True)
            if len(self.datasets[task]["items"]) == 0:
                print(f"[DatasetManager] Generating pre-loaded demo dataset for {task}...")
                generator = getattr(self, f"_generate_demo_{task}")
                generator(task_dir)
        self.save_metadata()

    def _generate_demo_classification(self, task_dir: Path):
        classes = self.datasets["classification"]["classes"]
        splits = ["train", "train", "train", "train", "val", "val", "test"]
        
        for idx in range(28):
            cls = classes[idx % len(classes)]
            split = splits[idx % len(splits)]
            filename = f"part_{idx+1:03d}_{cls.lower()}.png"
            file_path = task_dir / filename
            
            # Generate visual image with dark metallic/industrial look
            img = Image.new("RGB", (320, 320), color=(26, 32, 44))
            draw = ImageDraw.Draw(img)
            
            # Background grid/texture
            for x in range(0, 320, 32):
                draw.line([(x, 0), (x, 320)], fill=(34, 42, 56), width=1)
            for y in range(0, 320, 32):
                draw.line([(0, y), (320, y)], fill=(34, 42, 56), width=1)
                
            # Base mechanical component (bearing / gear / metallic plate)
            center = (160, 160)
            radius = 100
            draw.ellipse([center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius], fill=(60, 72, 90), outline=(120, 140, 170), width=4)
            draw.ellipse([center[0]-45, center[1]-45, center[0]+45, center[1]+45], fill=(30, 38, 50), outline=(160, 180, 210), width=3)

            # Draw class-specific visual features
            if cls == "Scratch":
                draw.line([(110, 120), (210, 190)], fill=(255, 75, 75), width=3)
                draw.line([(120, 130), (190, 175)], fill=(255, 180, 180), width=1)
            elif cls == "Dent":
                draw.ellipse([180, 110, 220, 150], fill=(20, 25, 35), outline=(255, 130, 50), width=3)
            elif cls == "Void":
                draw.ellipse([100, 170, 145, 215], fill=(12, 16, 24), outline=(220, 40, 40), width=2)
            # Normal has a pristine clean finish with small bolt markings
            for angle in [0, 60, 120, 180, 240, 300]:
                rad = math.radians(angle)
                bx = int(center[0] + 75 * math.cos(rad))
                by = int(center[1] + 75 * math.sin(rad))
                draw.ellipse([bx-6, by-6, bx+6, by+6], fill=(180, 200, 220), outline=(230, 240, 255), width=1)

            img.save(file_path, "PNG")
            
            self.add_image_item(
                task="classification",
                file_path=str(file_path),
                filename=filename,
                width=320,
                height=320,
                split=split,
                label=cls
            )

    def _generate_demo_detection(self, task_dir: Path):
        splits = ["train", "train", "train", "val", "test"]
        for idx in range(20):
            split = splits[idx % len(splits)]
            filename = f"pcb_board_{idx+1:03d}.png"
            file_path = task_dir / filename
            
            # Dark green PCB substrate
            img = Image.new("RGB", (480, 360), color=(14, 46, 28))
            draw = ImageDraw.Draw(img)
            
            # Traces
            for t in range(12):
                y_pos = 30 + t * 28
                draw.line([(20, y_pos), (200, y_pos), (240, y_pos+20), (450, y_pos+20)], fill=(28, 90, 56), width=2)

            boxes = []
            # Component 1: IC Chip
            ic_x1, ic_y1, ic_x2, ic_y2 = 180, 110, 300, 230
            draw.rectangle([ic_x1, ic_y1, ic_x2, ic_y2], fill=(22, 24, 28), outline=(180, 180, 190), width=2)
            draw.text((ic_x1+20, ic_y1+50), "ARM-CORTEX", fill=(190, 190, 200))
            boxes.append({"label": "IC_Chip", "box": [ic_x1, ic_y1, ic_x2, ic_y2], "confidence": 0.96})

            # Component 2: Resistor
            r_x1, r_y1, r_x2, r_y2 = 60, 80, 130, 120
            draw.rectangle([r_x1, r_y1, r_x2, r_y2], fill=(70, 130, 180), outline=(200, 200, 200), width=2)
            draw.line([(r_x1+20, r_y1), (r_x1+20, r_y2)], fill=(220, 50, 50), width=3)
            draw.line([(r_x1+35, r_y1), (r_x1+35, r_y2)], fill=(240, 200, 50), width=3)
            boxes.append({"label": "Resistor", "box": [r_x1, r_y1, r_x2, r_y2], "confidence": 0.92})

            # Component 3: Capacitor
            c_x1, c_y1, c_x2, c_y2 = 350, 60, 420, 140
            draw.ellipse([c_x1, c_y1, c_x2, c_y2], fill=(180, 140, 50), outline=(230, 200, 100), width=2)
            boxes.append({"label": "Capacitor", "box": [c_x1, c_y1, c_x2, c_y2], "confidence": 0.94})

            # Component 4: Diode (on some images)
            if idx % 2 == 0:
                d_x1, d_y1, d_x2, d_y2 = 70, 240, 140, 290
                draw.rectangle([d_x1, d_y1, d_x2, d_y2], fill=(20, 20, 20), outline=(200, 200, 200), width=2)
                draw.line([(d_x1+15, d_y1), (d_x1+15, d_y2)], fill=(220, 220, 220), width=4)
                boxes.append({"label": "Diode", "box": [d_x1, d_y1, d_x2, d_y2], "confidence": 0.89})

            img.save(file_path, "PNG")
            
            self.add_image_item(
                task="detection",
                file_path=str(file_path),
                filename=filename,
                width=480,
                height=360,
                split=split,
                boxes=boxes
            )

    def _generate_demo_anomaly(self, task_dir: Path):
        splits = ["train", "train", "train", "val", "test"]
        for idx in range(24):
            split = splits[idx % len(splits)]
            is_anomaly = (idx % 3 != 0) if split != "train" else (idx == 11)  # Train mostly normal
            filename = f"texture_{idx+1:03d}_{'anomaly' if is_anomaly else 'good'}.png"
            file_path = task_dir / filename
            heatmap_filename = f"heatmap_{idx+1:03d}.png"
            heatmap_path = task_dir / heatmap_filename
            
            # Brushed metal surface texture
            img = Image.new("RGB", (320, 320), color=(100, 105, 115))
            draw = ImageDraw.Draw(img)
            
            for y in range(0, 320, 4):
                tint = 95 + (y * 3) % 25
                draw.line([(0, y), (320, y)], fill=(tint, tint+4, tint+10), width=1)
                
            # Heatmap mask
            heat = Image.new("RGBA", (320, 320), (0, 0, 0, 0))
            hdraw = ImageDraw.Draw(heat)
            
            score = 0.08
            if is_anomaly:
                score = 0.82 + (idx % 15) * 0.01
                # Draw crack or dent defect
                fx = 120 + (idx * 27) % 100
                fy = 90 + (idx * 31) % 120
                draw.ellipse([fx, fy, fx+45, fy+35], fill=(30, 25, 25), outline=(15, 10, 10), width=2)
                draw.line([(fx+5, fy+10), (fx+55, fy+45)], fill=(20, 15, 15), width=3)
                
                # Glowing anomaly heatmap overlay
                for r in range(40, 0, -5):
                    alpha = int(180 * (1 - r/40))
                    hdraw.ellipse([fx+20-r, fy+20-r, fx+20+r, fy+20+r], fill=(255, int(40 + r*3), 20, alpha))
            
            img.save(file_path, "PNG")
            heat.save(heatmap_path, "PNG")

            self.add_image_item(
                task="anomaly",
                file_path=str(file_path),
                filename=filename,
                width=320,
                height=320,
                split=split,
                label="Anomaly" if is_anomaly else "Normal",
                is_anomaly=is_anomaly,
                anomaly_score=round(score, 3),
                heatmap_url=f"/api/images/anomaly/{heatmap_filename}" if is_anomaly else None
            )

    def _generate_demo_segmentation(self, task_dir: Path):
        splits = ["train", "train", "train", "val", "test"]
        for idx in range(20):
            split = splits[idx % len(splits)]
            filename = f"micro_cell_{idx+1:03d}.png"
            file_path = task_dir / filename
            
            img = Image.new("RGB", (360, 360), color=(18, 22, 32))
            draw = ImageDraw.Draw(img)
            
            # Circular substrate
            draw.ellipse([20, 20, 340, 340], fill=(28, 36, 52), outline=(50, 70, 100), width=2)
            
            segmentation = []
            # Cell / Core region 1
            poly1 = [
                {"x": 100, "y": 90},
                {"x": 160, "y": 70},
                {"x": 210, "y": 110},
                {"x": 190, "y": 170},
                {"x": 120, "y": 160}
            ]
            draw.polygon([(p["x"], p["y"]) for p in poly1], fill=(45, 110, 160), outline=(80, 180, 240))
            segmentation.append({"label": "Defect_Core", "color": "#06b6d4", "polygon": poly1})

            # Flaw region 2
            poly2 = [
                {"x": 220, "y": 190},
                {"x": 280, "y": 210},
                {"x": 260, "y": 270},
                {"x": 180, "y": 250}
            ]
            draw.polygon([(p["x"], p["y"]) for p in poly2], fill=(160, 45, 80), outline=(240, 80, 120))
            segmentation.append({"label": "Surface_Flaw", "color": "#f43f5e", "polygon": poly2})

            img.save(file_path, "PNG")
            
            self.add_image_item(
                task="segmentation",
                file_path=str(file_path),
                filename=filename,
                width=360,
                height=360,
                split=split,
                segmentation=segmentation
            )

dataset_manager = DatasetManager()
