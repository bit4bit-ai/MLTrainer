export type VisionTask = 'classification' | 'detection' | 'anomaly' | 'segmentation';

export type ActiveTab = 'gallery' | 'splitting' | 'training' | 'evaluation' | 'export';

export interface BoundingBox {
  label: string;
  box: [number, number, number, number]; // [x1, y1, x2, y2]
  confidence?: number;
}

export interface PolygonPoint {
  x: number;
  y: number;
}

export interface SegmentationRegion {
  label: string;
  color?: string;
  polygon: PolygonPoint[];
}

export interface DatasetItem {
  id: string;
  filename: string;
  file_path: string;
  url: string;
  width: number;
  height: number;
  split: 'train' | 'val' | 'test' | 'unassigned';
  label?: string;
  boxes?: BoundingBox[];
  segmentation?: SegmentationRegion[];
  is_anomaly?: boolean;
  anomaly_score?: number;
  heatmap_url?: string;
}

export interface Dataset {
  name: string;
  classes: string[];
  items: DatasetItem[];
}

export interface SplitDistribution {
  total_items: number;
  counts: { train: number; val: number; test: number };
  percentages: { train: number; val: number; test: number };
  class_distribution: Record<string, { train: number; val: number; test: number }>;
}

export interface TrainConfig {
  task: VisionTask;
  backbone: string;
  pretrained: boolean;
  image_size: number;
  batch_size: number;
  epochs: number;
  learning_rate: number;
  optimizer: 'adamw' | 'adam' | 'sgd';
  scheduler: 'cosine' | 'step' | 'none';
  augmentations: {
    random_flip: boolean;
    rotation: boolean;
    color_jitter: boolean;
    mixup: boolean;
  };
  early_stopping: boolean;
}

export interface EpochRecord {
  epoch: number;
  train_loss: number;
  val_loss: number;
  train_metric: number;
  val_metric: number;
  metric_name: string;
  learning_rate: number;
  timestamp: number;
}

export interface TrainingStatus {
  state: 'idle' | 'running' | 'paused' | 'stopped' | 'completed';
  task: VisionTask;
  current_epoch: number;
  total_epochs: number;
  best_metric: number;
  history: EpochRecord[];
  config?: any;
  device: string;
}

export interface ConfusionMatrixCell {
  actual: string;
  predicted: string;
  count: number;
  item_ids: string[];
}

export interface EvaluationData {
  task: VisionTask;
  primary_metric: { name: string; value: number; target: number };
  summary: Record<string, number>;
  per_class?: Record<string, { precision: number; recall: number; f1: number; support: number }>;
  per_class_ap?: Record<string, number>;
  per_class_iou?: Record<string, number>;
  confusion_matrix?: {
    classes: string[];
    matrix: ConfusionMatrixCell[];
  };
  roc_curve?: Array<{ fpr: number; tpr: number }>;
  predictions: Array<{
    id: string;
    filename: string;
    url: string;
    ground_truth: any;
    prediction: any;
  }>;
}

export interface ExportBundleResult {
  bundle_name: string;
  download_url: string;
  file_size_kb: number;
  format: string;
  quantization: string;
}
