import { Dataset, SplitDistribution, TrainConfig, TrainingStatus, EvaluationData, ExportBundleResult } from '../types';

export const API_BASE = '/api';

export async function fetchDataset(task: string): Promise<Dataset> {
  const res = await fetch(`${API_BASE}/dataset/${task}?_t=${Date.now()}`, {
    cache: 'no-store',
    headers: { 'Pragma': 'no-cache', 'Cache-Control': 'no-cache' }
  });
  if (!res.ok) throw new Error(`Failed to fetch dataset: ${res.statusText}`);
  return res.json();
}

export async function fetchSplitDistribution(task: string): Promise<SplitDistribution> {
  const res = await fetch(`${API_BASE}/dataset/${task}/distribution?_t=${Date.now()}`, {
    cache: 'no-store',
    headers: { 'Pragma': 'no-cache', 'Cache-Control': 'no-cache' }
  });
  if (!res.ok) throw new Error(`Failed to fetch distribution: ${res.statusText}`);
  return res.json();
}

export async function splitDataset(task: string, trainPct: number, valPct: number, testPct: number, stratify: boolean, seed: number): Promise<SplitDistribution> {
  const res = await fetch(`${API_BASE}/split/${task}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      train_pct: trainPct,
      val_pct: valPct,
      test_pct: testPct,
      stratify,
      seed
    })
  });
  if (!res.ok) throw new Error(`Split failed: ${res.statusText}`);
  return res.json();
}

export async function updateItemSplit(task: string, itemId: string, newSplit: string): Promise<SplitDistribution> {
  const res = await fetch(`${API_BASE}/split/${task}/reassign`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ item_id: itemId, new_split: newSplit })
  });
  if (!res.ok) throw new Error(`Reassign failed: ${res.statusText}`);
  return res.json();
}

export async function updateItemAnnotation(task: string, itemId: string, updates: any): Promise<any> {
  const res = await fetch(`${API_BASE}/dataset/${task}/items/${itemId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates)
  });
  if (!res.ok) throw new Error(`Update annotation failed: ${res.statusText}`);
  return res.json();
}

export async function deleteItem(task: string, itemId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/dataset/${task}/items/${itemId}`, {
    method: 'DELETE'
  });
  if (!res.ok) throw new Error(`Delete item failed: ${res.statusText}`);
  return res.json();
}

export async function batchDeleteItems(task: string, itemIds: string[]): Promise<any> {
  const res = await fetch(`${API_BASE}/dataset/${task}/batch-delete`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ item_ids: itemIds })
  });
  if (!res.ok) throw new Error(`Batch delete failed: ${res.statusText}`);
  return res.json();
}

export async function uploadImage(task: string, file: File, split: string = 'train', label?: string): Promise<any> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('split', split);
  if (label) formData.append('label', label);

  const res = await fetch(`${API_BASE}/dataset/${task}/upload`, {
    method: 'POST',
    body: formData
  });
  if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
  return res.json();
}

export async function uploadBatchImages(task: string, files: File[], split: string = 'train', label?: string): Promise<any> {
  const formData = new FormData();
  files.forEach(f => formData.append('files', f));
  formData.append('split', split);
  if (label) formData.append('label', label);

  const res = await fetch(`${API_BASE}/dataset/${task}/upload-batch`, {
    method: 'POST',
    body: formData
  });
  if (!res.ok) throw new Error(`Batch upload failed: ${res.statusText}`);
  return res.json();
}

export async function importLocalFolder(task: string, folderPath: string, split: string = 'train', label?: string): Promise<any> {
  const res = await fetch(`${API_BASE}/dataset/${task}/import-folder`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ folder_path: folderPath, split, label })
  });
  if (!res.ok) throw new Error(`Folder import failed: ${res.statusText}`);
  return res.json();
}

export async function fetchTrainingStatus(): Promise<TrainingStatus> {
  const res = await fetch(`${API_BASE}/training/status`);
  if (!res.ok) throw new Error(`Failed to fetch status: ${res.statusText}`);
  return res.json();
}

export async function startTraining(task: string, config: TrainConfig): Promise<any> {
  const res = await fetch(`${API_BASE}/training/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task, config })
  });
  if (!res.ok) throw new Error(`Start training failed: ${res.statusText}`);
  return res.json();
}

export async function pauseTraining(): Promise<any> {
  const res = await fetch(`${API_BASE}/training/pause`, { method: 'POST' });
  return res.json();
}

export async function resumeTraining(): Promise<any> {
  const res = await fetch(`${API_BASE}/training/resume`, { method: 'POST' });
  return res.json();
}

export async function stopTraining(): Promise<any> {
  const res = await fetch(`${API_BASE}/training/stop`, { method: 'POST' });
  return res.json();
}

export async function fetchEvaluation(task: string): Promise<EvaluationData> {
  const res = await fetch(`${API_BASE}/evaluation/${task}`);
  if (!res.ok) throw new Error(`Failed to fetch evaluation: ${res.statusText}`);
  return res.json();
}

export async function runInference(task: string, file: File): Promise<any> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/evaluation/${task}/infer`, {
    method: 'POST',
    body: formData
  });
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Inference failed: ${errText || res.statusText}`);
  }
  return res.json();
}

export async function fetchInferenceCode(task: string, format: string = 'onnx'): Promise<{ inference_code: string; fastapi_snippet: string }> {
  const res = await fetch(`${API_BASE}/export/code/${task}?format=${format}`);
  if (!res.ok) throw new Error(`Failed to fetch code: ${res.statusText}`);
  return res.json();
}

export async function createExportBundle(task: string, format: string, quantization: string): Promise<ExportBundleResult> {
  const res = await fetch(`${API_BASE}/export/bundle/${task}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ format, quantization })
  });
  if (!res.ok) throw new Error(`Export failed: ${res.statusText}`);
  return res.json();
}

export function createTrainingWebSocket(onMessage: (msg: any) => void): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/training`;
  const socket = new WebSocket(wsUrl);

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (e) {
      console.error('WebSocket parse error:', e);
    }
  };

  return socket;
}
