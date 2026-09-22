import asyncio
import time
import math
import random
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from ..config import CHECKPOINTS_DIR
from .dataset_manager import dataset_manager

try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE_STR = "cpu"

class Trainer:
    def __init__(self):
        self.state: str = "idle"  # idle, running, paused, stopped, completed
        self.task: str = "classification"
        self.config: Dict[str, Any] = {}
        self.current_epoch: int = 0
        self.total_epochs: int = 15
        self.history: List[Dict[str, Any]] = []
        self.subscribers: List[asyncio.Queue] = []
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._stop_requested = False
        self.training_task: Optional[asyncio.Task] = None
        self.best_metric: float = 0.0

    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue()
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        if q in self.subscribers:
            self.subscribers.remove(q)

    async def broadcast(self, message: Dict[str, Any]):
        for q in list(self.subscribers):
            try:
                await q.put(message)
            except Exception:
                pass

    def get_status(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "task": self.task,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "best_metric": round(self.best_metric, 4),
            "history": self.history,
            "config": self.config,
            "device": DEVICE_STR
        }

    async def start(self, task: str, config: Dict[str, Any]):
        if self.state in ["running", "paused"]:
            return {"status": "error", "message": "Training is already in progress"}

        self.task = task
        self.config = config
        self.total_epochs = int(config.get("epochs", 15))
        self.current_epoch = 0
        self.history = []
        self.best_metric = 0.0
        self.state = "running"
        self._stop_requested = False
        self._pause_event.set()

        # Run training loop in background
        self.training_task = asyncio.create_task(self._training_loop())
        return {"status": "success", "message": "Training started"}

    def pause(self):
        if self.state == "running":
            self.state = "paused"
            self._pause_event.clear()
            return {"status": "success", "message": "Training paused"}
        return {"status": "error", "message": "Cannot pause when not running"}

    def resume(self):
        if self.state == "paused":
            self.state = "running"
            self._pause_event.set()
            return {"status": "success", "message": "Training resumed"}
        return {"status": "error", "message": "Cannot resume when not paused"}

    def stop(self):
        if self.state in ["running", "paused"]:
            self._stop_requested = True
            self._pause_event.set()
            self.state = "stopped"
            return {"status": "success", "message": "Training stop requested"}
        return {"status": "error", "message": "No active training to stop"}

    async def _training_loop(self):
        start_time = time.time()
        metric_name = {
            "classification": "Accuracy",
            "detection": "mAP@50",
            "anomaly": "AUROC",
            "segmentation": "mIoU"
        }.get(self.task, "Score")

        base_lr = float(self.config.get("learning_rate", 0.001))
        steps_per_epoch = 10
        total_steps = self.total_epochs * steps_per_epoch

        for epoch in range(1, self.total_epochs + 1):
            if self._stop_requested:
                break

            await self._pause_event.wait()
            self.current_epoch = epoch

            # Simulate learning rate decay (cosine annealing style)
            progress = epoch / self.total_epochs
            current_lr = base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

            # Step sub-loop for smooth UI progress
            for step in range(1, steps_per_epoch + 1):
                if self._stop_requested:
                    break
                await self._pause_event.wait()

                # Step telemetry broadcast
                await asyncio.sleep(0.15)  # fast, dynamic streaming
                global_step = (epoch - 1) * steps_per_epoch + step
                elapsed = time.time() - start_time
                steps_remaining = total_steps - global_step
                sec_per_step = elapsed / max(1, global_step)
                eta = int(steps_remaining * sec_per_step)

                step_msg = {
                    "type": "step_progress",
                    "epoch": epoch,
                    "total_epochs": self.total_epochs,
                    "step": step,
                    "steps_per_epoch": steps_per_epoch,
                    "global_step": global_step,
                    "total_steps": total_steps,
                    "lr": round(current_lr, 6),
                    "eta_seconds": eta,
                    "time_elapsed": round(elapsed, 1)
                }
                await self.broadcast(step_msg)

            if self._stop_requested:
                break

            # Calculate mathematically realistic convergence curves per task
            decay_factor = math.exp(-2.2 * (epoch / self.total_epochs))
            train_loss = max(0.04, 1.85 * decay_factor + random.uniform(-0.02, 0.03))
            val_loss = max(0.08, 1.95 * decay_factor + 0.12 + random.uniform(-0.03, 0.04))

            rise_factor = 1.0 - math.exp(-2.8 * (epoch / self.total_epochs))
            train_metric = min(0.99, 0.45 + 0.52 * rise_factor + random.uniform(-0.015, 0.02))
            val_metric = min(0.97, 0.40 + 0.54 * rise_factor + random.uniform(-0.02, 0.02))

            if val_metric > self.best_metric:
                self.best_metric = val_metric

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "train_metric": round(train_metric, 4),
                "val_metric": round(val_metric, 4),
                "metric_name": metric_name,
                "learning_rate": round(current_lr, 6),
                "timestamp": round(time.time(), 2)
            }
            self.history.append(epoch_record)

            # Broadcast epoch completion
            await self.broadcast({
                "type": "epoch_complete",
                "record": epoch_record,
                "best_metric": round(self.best_metric, 4)
            })

        if not self._stop_requested:
            self.state = "completed"
            self._save_checkpoint()
        else:
            self.state = "stopped"

        await self.broadcast({
            "type": "training_finished",
            "state": self.state,
            "final_epoch": self.current_epoch,
            "best_metric": round(self.best_metric, 4)
        })

    def _save_checkpoint(self):
        ckpt_path = CHECKPOINTS_DIR / f"{self.task}_checkpoint_best.json"
        try:
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump({
                    "task": self.task,
                    "config": self.config,
                    "best_metric": self.best_metric,
                    "history": self.history,
                    "saved_at": time.time()
                }, f, indent=2)
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

trainer = Trainer()
