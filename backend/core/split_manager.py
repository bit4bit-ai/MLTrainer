import random
from typing import Dict, List, Any
from .dataset_manager import dataset_manager

class SplitManager:
    @staticmethod
    def split_dataset(task: str, train_pct: float, val_pct: float, test_pct: float, stratify: bool = True, seed: int = 42) -> Dict[str, Any]:
        random.seed(seed)
        ds = dataset_manager.get_dataset(task)
        items = ds["items"]
        
        if not items:
            return {"status": "error", "message": "Dataset is empty"}

        # Normalize percentages
        total = train_pct + val_pct + test_pct
        if total <= 0:
            train_pct, val_pct, test_pct = 70.0, 15.0, 15.0
            total = 100.0
        
        train_p = train_pct / total
        val_p = val_pct / total
        test_p = test_pct / total

        if stratify and task in ["classification", "anomaly"]:
            # Group items by class
            by_class: Dict[str, List[Dict[str, Any]]] = {}
            for item in items:
                lbl = item.get("label", "Unknown")
                by_class.setdefault(lbl, []).append(item)
            
            for cls_name, cls_items in by_class.items():
                random.shuffle(cls_items)
                n = len(cls_items)
                n_train = max(1, int(round(n * train_p))) if n >= 3 else 1
                n_val = max(1, int(round(n * val_p))) if (n - n_train) >= 2 else (1 if n >= 2 else 0)
                n_test = n - n_train - n_val
                if n_test < 0:
                    n_val = max(0, n_val + n_test)
                    n_test = 0

                for i, item in enumerate(cls_items):
                    if i < n_train:
                        item["split"] = "train"
                    elif i < n_train + n_val:
                        item["split"] = "val"
                    else:
                        item["split"] = "test"
        else:
            # Random split across all items
            shuffled = list(items)
            random.shuffle(shuffled)
            n = len(shuffled)
            n_train = int(round(n * train_p))
            n_val = int(round(n * val_p))
            
            for i, item in enumerate(shuffled):
                if i < n_train:
                    item["split"] = "train"
                elif i < n_train + n_val:
                    item["split"] = "val"
                else:
                    item["split"] = "test"

        dataset_manager.save_metadata()
        return SplitManager.get_split_distribution(task)

    @staticmethod
    def get_split_distribution(task: str) -> Dict[str, Any]:
        ds = dataset_manager.get_dataset(task)
        items = ds["items"]
        classes = ds["classes"]

        counts = {"train": 0, "val": 0, "test": 0}
        class_distribution = {cls: {"train": 0, "val": 0, "test": 0} for cls in classes}

        for item in items:
            split = item.get("split", "train")
            if split not in counts:
                split = "train"
            counts[split] += 1

            lbl = item.get("label")
            if lbl in class_distribution:
                class_distribution[lbl][split] += 1
            elif task == "detection":
                # For detection, count unique labels present in bounding boxes
                boxes = item.get("boxes", [])
                seen = set()
                for b in boxes:
                    blbl = b.get("label")
                    if blbl in class_distribution and blbl not in seen:
                        class_distribution[blbl][split] += 1
                        seen.add(blbl)

        total_items = len(items)
        percentages = {
            k: round((v / total_items * 100), 1) if total_items > 0 else 0
            for k, v in counts.items()
        }

        return {
            "total_items": total_items,
            "counts": counts,
            "percentages": percentages,
            "class_distribution": class_distribution
        }

    @staticmethod
    def reassign_item(task: str, item_id: str, new_split: str) -> bool:
        ds = dataset_manager.get_dataset(task)
        for item in ds["items"]:
            if item["id"] == item_id:
                item["split"] = new_split
                dataset_manager.save_metadata()
                return True
        return False
