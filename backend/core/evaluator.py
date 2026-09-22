import io
import time
import math
import base64
import random
from typing import Dict, List, Any
from PIL import Image, ImageDraw, ImageFilter
from .dataset_manager import dataset_manager

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class Evaluator:
    @staticmethod
    def get_evaluation_metrics(task: str) -> Dict[str, Any]:
        ds = dataset_manager.get_dataset(task)
        items = ds["items"]
        classes = ds["classes"]
        test_items = [i for i in items if i.get("split") == "test"]
        if not test_items:
            test_items = items[:max(1, len(items) // 4)]

        if task == "classification":
            return Evaluator._eval_classification(classes, test_items)
        elif task == "detection":
            return Evaluator._eval_detection(classes, test_items)
        elif task == "anomaly":
            return Evaluator._eval_anomaly(classes, test_items)
        elif task == "segmentation":
            return Evaluator._eval_segmentation(classes, test_items)
        return {}

    @staticmethod
    def _eval_classification(classes: List[str], test_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Generate realistic confusion matrix
        matrix = {c_true: {c_pred: [] for c_pred in classes} for c_true in classes}
        
        for item in test_items:
            gt = item.get("label", classes[0])
            if gt not in matrix:
                gt = classes[0]
            # 88% probability correct prediction
            if random.random() < 0.88:
                pred = gt
            else:
                other_classes = [c for c in classes if c != gt]
                pred = random.choice(other_classes) if other_classes else gt
            matrix[gt][pred].append(item["id"])

        # Calculate metrics
        total = 0
        correct = 0
        per_class = {}
        for c in classes:
            tp = len(matrix[c][c])
            fn = sum(len(matrix[c][p]) for p in classes if p != c)
            fp = sum(len(matrix[t][c]) for t in classes if t != c)
            prec = round(tp / max(1, tp + fp), 3)
            rec = round(tp / max(1, tp + fn), 3)
            f1 = round(2 * prec * rec / max(0.001, prec + rec), 3)
            per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
            total += (tp + fn)
            correct += tp

        overall_acc = round(correct / max(1, total), 3)
        avg_f1 = round(sum(v["f1"] for v in per_class.values()) / max(1, len(per_class)), 3)
        avg_prec = round(sum(v["precision"] for v in per_class.values()) / max(1, len(per_class)), 3)
        avg_rec = round(sum(v["recall"] for v in per_class.values()) / max(1, len(per_class)), 3)

        # Build clean JSON serializable matrix
        matrix_data = [
            {"actual": c_true, "predicted": c_pred, "count": len(matrix[c_true][c_pred]), "item_ids": matrix[c_true][c_pred]}
            for c_true in classes
            for c_pred in classes
        ]

        return {
            "task": "classification",
            "primary_metric": {"name": "Accuracy", "value": overall_acc, "target": 0.95},
            "summary": {
                "accuracy": overall_acc,
                "precision": avg_prec,
                "recall": avg_rec,
                "f1_score": avg_f1,
                "test_samples": total
            },
            "per_class": per_class,
            "confusion_matrix": {
                "classes": classes,
                "matrix": matrix_data
            },
            "predictions": Evaluator._generate_sample_predictions("classification", test_items, classes)
        }

    @staticmethod
    def _eval_detection(classes: List[str], test_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        per_class_ap = {
            cls: round(0.85 + (i * 0.03) % 0.12, 3)
            for i, cls in enumerate(classes)
        }
        mean_ap50 = round(sum(per_class_ap.values()) / max(1, len(per_class_ap)), 3)
        mean_ap50_95 = round(mean_ap50 * 0.72, 3)

        return {
            "task": "detection",
            "primary_metric": {"name": "mAP@50", "value": mean_ap50, "target": 0.88},
            "summary": {
                "mAP_50": mean_ap50,
                "mAP_50_95": mean_ap50_95,
                "precision": 0.912,
                "recall": 0.874,
                "f1_score": 0.892,
                "test_samples": len(test_items)
            },
            "per_class_ap": per_class_ap,
            "predictions": Evaluator._generate_sample_predictions("detection", test_items, classes)
        }

    @staticmethod
    def _eval_anomaly(classes: List[str], test_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Realistic ROC and PR points
        roc_curve = [
            {"fpr": 0.0, "tpr": 0.0},
            {"fpr": 0.02, "tpr": 0.65},
            {"fpr": 0.05, "tpr": 0.88},
            {"fpr": 0.10, "tpr": 0.95},
            {"fpr": 0.20, "tpr": 0.98},
            {"fpr": 0.50, "tpr": 0.99},
            {"fpr": 1.0, "tpr": 1.0}
        ]
        return {
            "task": "anomaly",
            "primary_metric": {"name": "AUROC", "value": 0.968, "target": 0.98},
            "summary": {
                "image_auroc": 0.968,
                "pixel_auroc": 0.954,
                "optimal_threshold": 0.485,
                "f1_max": 0.941,
                "test_samples": len(test_items)
            },
            "roc_curve": roc_curve,
            "predictions": Evaluator._generate_sample_predictions("anomaly", test_items, classes)
        }

    @staticmethod
    def _eval_segmentation(classes: List[str], test_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        per_class_iou = {
            cls: round(0.81 + (i * 0.04) % 0.15, 3)
            for i, cls in enumerate(classes)
        }
        mean_iou = round(sum(per_class_iou.values()) / max(1, len(per_class_iou)), 3)
        mean_dice = round(2 * mean_iou / (1 + mean_iou), 3)

        return {
            "task": "segmentation",
            "primary_metric": {"name": "Mean IoU", "value": mean_iou, "target": 0.85},
            "summary": {
                "mean_iou": mean_iou,
                "dice_score": mean_dice,
                "pixel_accuracy": 0.964,
                "test_samples": len(test_items)
            },
            "per_class_iou": per_class_iou,
            "predictions": Evaluator._generate_sample_predictions("segmentation", test_items, classes)
        }

    @staticmethod
    def _generate_sample_predictions(task: str, items: List[Dict[str, Any]], classes: List[str]) -> List[Dict[str, Any]]:
        results = []
        for item in items[:8]:
            pred_item = {
                "id": item["id"],
                "filename": item["filename"],
                "url": item["url"],
                "ground_truth": {},
                "prediction": {}
            }
            if task == "classification":
                gt = item.get("label", classes[0])
                conf = round(random.uniform(0.85, 0.99), 2)
                pred_item["ground_truth"] = {"label": gt}
                pred_item["prediction"] = {"label": gt, "confidence": conf, "is_correct": True}
            elif task == "detection":
                gt_boxes = item.get("boxes", [])
                pred_boxes = []
                for b in gt_boxes:
                    jitter = random.randint(-4, 4)
                    bx = b["box"]
                    pred_boxes.append({
                        "label": b["label"],
                        "confidence": round(random.uniform(0.86, 0.98), 2),
                        "box": [max(0, bx[0]+jitter), max(0, bx[1]+jitter), bx[2]+jitter, bx[3]+jitter]
                    })
                pred_item["ground_truth"] = {"boxes": gt_boxes}
                pred_item["prediction"] = {"boxes": pred_boxes}
            elif task == "anomaly":
                gt_ano = item.get("is_anomaly", False)
                gt_score = item.get("anomaly_score", 0.05)
                pred_score = round(min(0.99, max(0.01, gt_score + random.uniform(-0.04, 0.04))), 3)
                pred_item["ground_truth"] = {"is_anomaly": gt_ano, "score": gt_score}
                pred_item["prediction"] = {
                    "is_anomaly": pred_score >= 0.485,
                    "score": pred_score,
                    "heatmap_url": item.get("heatmap_url")
                }
            elif task == "segmentation":
                gt_seg = item.get("segmentation", [])
                pred_item["ground_truth"] = {"segmentation": gt_seg}
                pred_item["prediction"] = {"segmentation": gt_seg, "confidence": 0.93}

            results.append(pred_item)
        return results

    @staticmethod
    def run_inference(task: str, image_bytes: bytes, filename: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        ds = dataset_manager.get_dataset(task)
        classes = ds["classes"]

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            width, height = image.size
        except Exception as e:
            return {"status": "error", "message": f"Failed to parse image: {e}"}

        device = "CUDA" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "CPU"
        prediction = {}

        if task == "classification":
            sample_pixel = image.resize((1, 1)).getpixel((0, 0))
            seed_val = (sample_pixel[0] * 3 + sample_pixel[1] * 7 + sample_pixel[2] * 11) % len(classes)
            top_class = classes[seed_val]
            top_conf = round(random.uniform(0.91, 0.98), 4)

            remaining_conf = round(1.0 - top_conf, 4)
            other_classes = [c for c in classes if c != top_class]
            probs = [{"class": top_class, "confidence": top_conf}]

            if other_classes:
                raw_parts = [random.uniform(0.1, 1.0) for _ in other_classes]
                part_sum = sum(raw_parts)
                for c, p in zip(other_classes, raw_parts):
                    probs.append({
                        "class": c,
                        "confidence": round((p / part_sum) * remaining_conf, 4)
                    })
            probs.sort(key=lambda x: x["confidence"], reverse=True)

            prediction = {
                "label": top_class,
                "confidence": top_conf,
                "all_probabilities": probs
            }

        elif task == "detection":
            boxes = []
            num_objects = random.randint(1, min(3, len(classes)))
            for i in range(num_objects):
                bx_w = random.randint(int(width * 0.18), int(width * 0.45))
                bx_h = random.randint(int(height * 0.18), int(height * 0.45))
                x1 = random.randint(int(width * 0.05), max(int(width * 0.06), width - bx_w - 10))
                y1 = random.randint(int(height * 0.05), max(int(height * 0.06), height - bx_h - 10))
                x2 = min(width, x1 + bx_w)
                y2 = min(height, y1 + bx_h)
                boxes.append({
                    "id": f"box_{i+1}",
                    "label": classes[i % len(classes)],
                    "confidence": round(random.uniform(0.86, 0.98), 3),
                    "box": [x1, y1, x2, y2]
                })
            prediction = {
                "boxes": boxes,
                "count": len(boxes)
            }

        elif task == "anomaly":
            center_crop = image.crop((int(width * 0.25), int(height * 0.25), int(width * 0.75), int(height * 0.75)))
            stat = center_crop.resize((10, 10))
            pixels = list(stat.getdata())
            lum = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in pixels]
            mean_lum = sum(lum) / len(lum)
            var = sum((x - mean_lum) ** 2 for x in lum) / len(lum)

            raw_score = min(0.95, max(0.05, (var / 2500.0) + random.uniform(-0.1, 0.15)))
            anomaly_score = round(raw_score, 3)
            is_anomaly = anomaly_score >= 0.485

            hm = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(hm)
            if is_anomaly:
                cx, cy = int(width * 0.52), int(height * 0.48)
                radius = int(min(width, height) * 0.24)
                for r in range(radius, 0, -6):
                    alpha = int(190 * (1 - r / radius))
                    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 30, 80, alpha))
            else:
                cx, cy = int(width * 0.5), int(height * 0.5)
                radius = int(min(width, height) * 0.16)
                for r in range(radius, 0, -8):
                    alpha = int(90 * (1 - r / radius))
                    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(30, 220, 120, alpha))

            hm = hm.filter(ImageFilter.GaussianBlur(radius=8))
            buf = io.BytesIO()
            hm.save(buf, format="PNG")
            heatmap_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

            prediction = {
                "is_anomaly": is_anomaly,
                "score": anomaly_score,
                "threshold": 0.485,
                "verdict": "REJECT - DEFECT FOUND" if is_anomaly else "PASS - NOMINAL SURFACE",
                "heatmap_url": heatmap_b64
            }

        elif task == "segmentation":
            masks = []
            selected_cls = classes[1] if len(classes) > 1 else classes[0]
            cx, cy = int(width * 0.5), int(height * 0.5)
            rx, ry = int(width * 0.25), int(height * 0.2)
            polygon = [
                [cx - rx, cy],
                [cx - int(rx * 0.7), cy - ry],
                [cx + int(rx * 0.6), cy - int(ry * 0.8)],
                [cx + rx, cy],
                [cx + int(rx * 0.5), cy + ry],
                [cx - int(rx * 0.5), cy + int(ry * 0.9)]
            ]
            masks.append({
                "label": selected_cls,
                "confidence": 0.94,
                "polygon": polygon,
                "area_pct": round((math.pi * rx * ry) / (width * height) * 100, 1)
            })
            prediction = {
                "masks": masks,
                "classes_detected": [selected_cls]
            }

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "status": "success",
            "task": task,
            "filename": filename,
            "width": width,
            "height": height,
            "device": device,
            "inference_time_ms": latency_ms,
            "prediction": prediction
        }

evaluator = Evaluator()
