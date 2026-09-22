import React, { useState } from 'react';
import { VisionTask, TrainConfig, TrainingStatus, EpochRecord } from '../../types';
import { Play, Pause, Square, Cpu, Zap, Activity, CheckCircle, TrendingDown, TrendingUp, Sliders, Settings } from 'lucide-react';
import { startTraining, pauseTraining, resumeTraining, stopTraining } from '../../services/api';

interface TrainingStudioProps {
  task: VisionTask;
  status: TrainingStatus;
  onRefreshStatus: () => Promise<void>;
}

export const TrainingStudio: React.FC<TrainingStudioProps> = ({
  task,
  status,
  onRefreshStatus
}) => {
  // Default configurations based on task
  const [backbone, setBackbone] = useState<string>(() => {
    if (task === 'classification') return 'resnet18';
    if (task === 'detection') return 'tiny_yolo';
    if (task === 'anomaly') return 'autoencoder';
    return 'unet';
  });

  const [pretrained, setPretrained] = useState(true);
  const [imageSize, setImageSize] = useState(224);
  const [batchSize, setBatchSize] = useState(16);
  const [epochs, setEpochs] = useState(15);
  const [lr, setLr] = useState(0.001);
  const [optimizer, setOptimizer] = useState<'adamw' | 'adam' | 'sgd'>('adamw');
  const [scheduler, setScheduler] = useState<'cosine' | 'step' | 'none'>('cosine');
  const [augmentations, setAugmentations] = useState({
    random_flip: true,
    rotation: true,
    color_jitter: false,
    mixup: false
  });
  const [earlyStopping, setEarlyStopping] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleStart = async () => {
    setIsSubmitting(true);
    try {
      const config: TrainConfig = {
        task,
        backbone,
        pretrained,
        image_size: imageSize,
        batch_size: batchSize,
        epochs,
        learning_rate: lr,
        optimizer,
        scheduler,
        augmentations,
        early_stopping: earlyStopping
      };
      await startTraining(task, config);
      await onRefreshStatus();
    } catch (e) {
      console.error(e);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handlePause = async () => {
    await pauseTraining();
    await onRefreshStatus();
  };

  const handleResume = async () => {
    await resumeTraining();
    await onRefreshStatus();
  };

  const handleStop = async () => {
    await stopTraining();
    await onRefreshStatus();
  };

  const isRunning = status.state === 'running';
  const isPaused = status.state === 'paused';
  const history = status.history || [];
  const latestEpoch = history.length > 0 ? history[history.length - 1] : null;

  // SVG Chart rendering helper
  const renderSvgLossChart = () => {
    if (history.length < 2) {
      return (
        <div style={{ height: '220px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b', fontSize: '0.85rem' }}>
          Loss curves will stream live as training progresses...
        </div>
      );
    }

    const width = 480;
    const height = 200;
    const padding = 30;

    const maxLoss = Math.max(...history.map(h => Math.max(h.train_loss, h.val_loss)), 1.0);
    const minLoss = 0.0;

    const getX = (idx: number) => padding + (idx / (history.length - 1)) * (width - 2 * padding);
    const getY = (loss: number) => height - padding - ((loss - minLoss) / (maxLoss - minLoss)) * (height - 2 * padding);

    const trainPoints = history.map((h, i) => `${getX(i)},${getY(h.train_loss)}`).join(' ');
    const valPoints = history.map((h, i) => `${getX(i)},${getY(h.val_loss)}`).join(' ');

    return (
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: '220px', overflow: 'visible' }}>
        {/* Horizontal gridlines */}
        {[0.25, 0.5, 0.75, 1.0].map((frac, idx) => {
          const y = height - padding - frac * (height - 2 * padding);
          return (
            <line key={idx} x1={padding} y1={y} x2={width - padding} y2={y} stroke="#1e293b" strokeDasharray="3 3" />
          );
        })}

        {/* Train Loss Line (Cyan) */}
        <polyline points={trainPoints} fill="none" stroke="#06b6d4" strokeWidth="2.5" />
        
        {/* Val Loss Line (Amber) */}
        <polyline points={valPoints} fill="none" stroke="#f59e0b" strokeWidth="2.5" />

        {/* Dots on latest point */}
        {latestEpoch && (
          <>
            <circle cx={getX(history.length - 1)} cy={getY(latestEpoch.train_loss)} r="4" fill="#06b6d4" />
            <circle cx={getX(history.length - 1)} cy={getY(latestEpoch.val_loss)} r="4" fill="#f59e0b" />
          </>
        )}
      </svg>
    );
  };

  const renderSvgMetricChart = () => {
    if (history.length < 2) {
      return (
        <div style={{ height: '220px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b', fontSize: '0.85rem' }}>
          Metric curve will stream live as epochs complete...
        </div>
      );
    }

    const width = 480;
    const height = 200;
    const padding = 30;

    const getX = (idx: number) => padding + (idx / (history.length - 1)) * (width - 2 * padding);
    const getY = (val: number) => height - padding - val * (height - 2 * padding);

    const valMetricPoints = history.map((h, i) => `${getX(i)},${getY(h.val_metric)}`).join(' ');
    const trainMetricPoints = history.map((h, i) => `${getX(i)},${getY(h.train_metric)}`).join(' ');

    return (
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: '220px', overflow: 'visible' }}>
        {[0.25, 0.5, 0.75, 1.0].map((frac, idx) => {
          const y = height - padding - frac * (height - 2 * padding);
          return (
            <line key={idx} x1={padding} y1={y} x2={width - padding} y2={y} stroke="#1e293b" strokeDasharray="3 3" />
          );
        })}

        {/* Train Metric (Indigo) */}
        <polyline points={trainMetricPoints} fill="none" stroke="#6366f1" strokeWidth="2" strokeDasharray="4 2" />
        
        {/* Val Metric (Emerald) */}
        <polyline points={valMetricPoints} fill="none" stroke="#10b981" strokeWidth="3" />

        {latestEpoch && (
          <circle cx={getX(history.length - 1)} cy={getY(latestEpoch.val_metric)} r="5" fill="#10b981" />
        )}
      </svg>
    );
  };

  return (
    <div style={{ padding: '0 20px 24px', maxWidth: '1280px', margin: '0 auto' }}>
      
      {/* Top Banner & Training Control Bar */}
      <div className="glass-panel" style={{ padding: '20px 24px', marginBottom: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <div style={{
              width: '46px',
              height: '46px',
              borderRadius: '12px',
              background: 'linear-gradient(135deg, #06b6d4, #3b82f6)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 0 16px rgba(6,182,212,0.4)'
            }}>
              <Activity size={24} color="#ffffff" />
            </div>
            <div>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#f8fafc' }}>
                {task.toUpperCase()} Training Studio
              </h2>
              <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
                Status: <strong style={{ color: isRunning ? '#34d399' : '#f8fafc', textTransform: 'uppercase' }}>{status.state}</strong> • 
                Device: <strong style={{ color: '#38bdf8' }}>{status.device.toUpperCase()}</strong> • 
                Best Metric: <strong style={{ color: '#6ee7b7' }}>{(status.best_metric * 100).toFixed(1)}%</strong>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            {!isRunning && !isPaused && (
              <button
                onClick={handleStart}
                disabled={isSubmitting}
                className="btn-primary"
                style={{ padding: '10px 22px', fontSize: '0.95rem' }}
              >
                <Play size={18} />
                {isSubmitting ? 'Starting...' : 'Start Model Training'}
              </button>
            )}

            {isRunning && (
              <button onClick={handlePause} className="btn-secondary" style={{ padding: '10px 18px' }}>
                <Pause size={18} color="#f59e0b" />
                Pause
              </button>
            )}

            {isPaused && (
              <button onClick={handleResume} className="btn-primary" style={{ padding: '10px 18px' }}>
                <Play size={18} />
                Resume Training
              </button>
            )}

            {(isRunning || isPaused) && (
              <button onClick={handleStop} className="btn-danger" style={{ padding: '10px 18px' }}>
                <Square size={18} />
                Abort / Stop
              </button>
            )}
          </div>

        </div>

        {/* Progress Bar (when active) */}
        {(isRunning || isPaused || status.state === 'completed') && (
          <div style={{ marginTop: '20px', paddingTop: '16px', borderTop: '1px solid #1e293b' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#94a3b8', marginBottom: '6px' }}>
              <span>Epoch {status.current_epoch} of {status.total_epochs}</span>
              <span>{Math.round((status.current_epoch / Math.max(1, status.total_epochs)) * 100)}% Completed</span>
            </div>
            <div style={{ height: '8px', background: '#0b1120', borderRadius: '4px', overflow: 'hidden' }}>
              <div style={{
                height: '100%',
                width: `${(status.current_epoch / Math.max(1, status.total_epochs)) * 100}%`,
                background: 'linear-gradient(90deg, #06b6d4, #10b981)',
                transition: 'width 0.3s ease'
              }} />
            </div>
          </div>
        )}
      </div>

      {/* Main Grid: Config (Left) & Real-time Telemetry (Right) */}
      <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: '24px' }}>
        
        {/* Left: Hyperparameters Config Panel */}
        <div className="glass-panel" style={{ padding: '20px', height: 'fit-content' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#f8fafc', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Settings size={18} color="#06b6d4" />
            Hyperparameter Config
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            
            {/* Backbone Selection */}
            <div>
              <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', marginBottom: '6px' }}>
                MODEL BACKBONE / ARCHITECTURE
              </label>
              <select
                value={backbone}
                onChange={(e) => setBackbone(e.target.value)}
                disabled={isRunning || isPaused}
                style={{ width: '100%' }}
              >
                {task === 'classification' && (
                  <>
                    <option value="resnet18">ResNet-18 (Fast, Balanced)</option>
                    <option value="mobilenet_v3_small">MobileNet-V3 Small (Edge / Mobile)</option>
                    <option value="efficientnet_b0">EfficientNet-B0 (High Accuracy)</option>
                    <option value="custom_cnn">Custom Lightweight CNN</option>
                  </>
                )}
                {task === 'detection' && (
                  <>
                    <option value="tiny_yolo">TinyYOLO Grid Detector</option>
                    <option value="resnet_detector">ResNet-18 Feature Pyramid Head</option>
                  </>
                )}
                {task === 'anomaly' && (
                  <>
                    <option value="autoencoder">Deep Conv Autoencoder</option>
                    <option value="patchcore">PatchCore Feature-Distance Memory</option>
                  </>
                )}
                {task === 'segmentation' && (
                  <>
                    <option value="unet">U-Net ResNet Encoder-Decoder</option>
                    <option value="deeplabv3">DeepLabV3+ Atrous Spatial Pooling</option>
                  </>
                )}
              </select>
            </div>

            {/* Transfer learning toggle */}
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '0.825rem', color: '#e2e8f0' }}>
              <input
                type="checkbox"
                checked={pretrained}
                onChange={(e) => setPretrained(e.target.checked)}
                disabled={isRunning || isPaused}
                style={{ accentColor: '#06b6d4' }}
              />
              <span>Transfer Learning (Pretrained ImageNet)</span>
            </label>

            {/* Epochs slider */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8' }}>EPOCHS</span>
                <strong style={{ color: '#06b6d4' }}>{epochs}</strong>
              </div>
              <input
                type="range"
                min="5"
                max="50"
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value))}
                disabled={isRunning || isPaused}
              />
            </div>

            {/* Batch Size & Resolution */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', marginBottom: '6px' }}>
                  BATCH SIZE
                </label>
                <select
                  value={batchSize}
                  onChange={(e) => setBatchSize(parseInt(e.target.value))}
                  disabled={isRunning || isPaused}
                  style={{ width: '100%' }}
                >
                  <option value={4}>4</option>
                  <option value={8}>8</option>
                  <option value={16}>16</option>
                  <option value={32}>32</option>
                </select>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', marginBottom: '6px' }}>
                  RESOLUTION
                </label>
                <select
                  value={imageSize}
                  onChange={(e) => setImageSize(parseInt(e.target.value))}
                  disabled={isRunning || isPaused}
                  style={{ width: '100%' }}
                >
                  <option value={224}>224 × 224</option>
                  <option value={384}>384 × 384</option>
                  <option value={512}>512 × 512</option>
                </select>
              </div>
            </div>

            {/* Optimizer & LR */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', marginBottom: '6px' }}>
                  OPTIMIZER
                </label>
                <select
                  value={optimizer}
                  onChange={(e) => setOptimizer(e.target.value as any)}
                  disabled={isRunning || isPaused}
                  style={{ width: '100%' }}
                >
                  <option value="adamw">AdamW</option>
                  <option value="adam">Adam</option>
                  <option value="sgd">SGD (Mom)</option>
                </select>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', marginBottom: '6px' }}>
                  LEARNING RATE
                </label>
                <select
                  value={lr}
                  onChange={(e) => setLr(parseFloat(e.target.value))}
                  disabled={isRunning || isPaused}
                  style={{ width: '100%' }}
                >
                  <option value={0.01}>0.01 (High)</option>
                  <option value={0.001}>0.001 (Default)</option>
                  <option value={0.0003}>0.0003 (Fine)</option>
                  <option value={0.0001}>0.0001 (Micro)</option>
                </select>
              </div>
            </div>

            {/* Augmentations */}
            <div>
              <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', marginBottom: '8px' }}>
                DATA AUGMENTATIONS
              </label>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '0.75rem', color: '#cbd5e1' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={augmentations.random_flip}
                    onChange={(e) => setAugmentations({ ...augmentations, random_flip: e.target.checked })}
                    disabled={isRunning || isPaused}
                  />
                  <span>Random Flip</span>
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={augmentations.rotation}
                    onChange={(e) => setAugmentations({ ...augmentations, rotation: e.target.checked })}
                    disabled={isRunning || isPaused}
                  />
                  <span>Rotation (±15°)</span>
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={augmentations.color_jitter}
                    onChange={(e) => setAugmentations({ ...augmentations, color_jitter: e.target.checked })}
                    disabled={isRunning || isPaused}
                  />
                  <span>Color Jitter</span>
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={augmentations.mixup}
                    onChange={(e) => setAugmentations({ ...augmentations, mixup: e.target.checked })}
                    disabled={isRunning || isPaused}
                  />
                  <span>MixUp / CutMix</span>
                </label>
              </div>
            </div>

          </div>
        </div>

        {/* Right: Real-time Telemetry, Live Charts, and Logs */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Key Metric Scorecards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '16px' }}>
            
            <div className="glass-card" style={{ padding: '14px 18px' }}>
              <div style={{ fontSize: '0.72rem', color: '#94a3b8', fontWeight: 600, textTransform: 'uppercase' }}>Current Train Loss</div>
              <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#38bdf8', marginTop: '4px' }}>
                {latestEpoch ? latestEpoch.train_loss.toFixed(4) : '--'}
              </div>
            </div>

            <div className="glass-card" style={{ padding: '14px 18px' }}>
              <div style={{ fontSize: '0.72rem', color: '#94a3b8', fontWeight: 600, textTransform: 'uppercase' }}>Validation Loss</div>
              <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#f59e0b', marginTop: '4px' }}>
                {latestEpoch ? latestEpoch.val_loss.toFixed(4) : '--'}
              </div>
            </div>

            <div className="glass-card" style={{ padding: '14px 18px' }}>
              <div style={{ fontSize: '0.72rem', color: '#94a3b8', fontWeight: 600, textTransform: 'uppercase' }}>
                Validation {latestEpoch?.metric_name || 'Metric'}
              </div>
              <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#10b981', marginTop: '4px' }}>
                {latestEpoch ? `${(latestEpoch.val_metric * 100).toFixed(1)}%` : '--'}
              </div>
            </div>

            <div className="glass-card" style={{ padding: '14px 18px' }}>
              <div style={{ fontSize: '0.72rem', color: '#94a3b8', fontWeight: 600, textTransform: 'uppercase' }}>Learning Rate</div>
              <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#a78bfa', marginTop: '4px' }}>
                {latestEpoch ? latestEpoch.learning_rate.toExponential(2) : lr.toExponential(2)}
              </div>
            </div>

          </div>

          {/* Charts Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
            
            {/* Loss Curve */}
            <div className="glass-panel" style={{ padding: '18px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <div style={{ fontSize: '0.875rem', fontWeight: 700, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <TrendingDown size={16} color="#06b6d4" />
                  Convergence Loss Curves
                </div>
                <div style={{ display: 'flex', gap: '12px', fontSize: '0.75rem' }}>
                  <span style={{ color: '#06b6d4', display: 'flex', alignItems: 'center', gap: '4px' }}>● Train Loss</span>
                  <span style={{ color: '#f59e0b', display: 'flex', alignItems: 'center', gap: '4px' }}>● Val Loss</span>
                </div>
              </div>
              {renderSvgLossChart()}
            </div>

            {/* Validation Metric Curve */}
            <div className="glass-panel" style={{ padding: '18px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <div style={{ fontSize: '0.875rem', fontWeight: 700, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <TrendingUp size={16} color="#10b981" />
                  {latestEpoch?.metric_name || 'Evaluation'} Metric Progression
                </div>
                <div style={{ display: 'flex', gap: '12px', fontSize: '0.75rem' }}>
                  <span style={{ color: '#6366f1', display: 'flex', alignItems: 'center', gap: '4px' }}>- - Train</span>
                  <span style={{ color: '#10b981', display: 'flex', alignItems: 'center', gap: '4px' }}>● Validation</span>
                </div>
              </div>
              {renderSvgMetricChart()}
            </div>

          </div>

          {/* Epoch History Table / Console */}
          <div className="glass-panel" style={{ padding: '18px' }}>
            <h4 style={{ fontSize: '0.875rem', fontWeight: 700, color: '#f8fafc', marginBottom: '12px' }}>
              Live Training Log ({history.length} epochs recorded)
            </h4>
            <div style={{ maxHeight: '180px', overflowY: 'auto' }}>
              <table style={{ width: '100%', fontSize: '0.8rem', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ color: '#64748b', textAlign: 'left', borderBottom: '1px solid #1e293b' }}>
                    <th style={{ padding: '6px 10px' }}>Epoch</th>
                    <th style={{ padding: '6px 10px' }}>Train Loss</th>
                    <th style={{ padding: '6px 10px' }}>Val Loss</th>
                    <th style={{ padding: '6px 10px' }}>Val {latestEpoch?.metric_name || 'Score'}</th>
                    <th style={{ padding: '6px 10px' }}>Learning Rate</th>
                  </tr>
                </thead>
                <tbody>
                  {history.slice().reverse().map((h) => (
                    <tr key={h.epoch} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                      <td style={{ padding: '6px 10px', color: '#f8fafc', fontWeight: 600 }}>#{h.epoch}</td>
                      <td style={{ padding: '6px 10px', color: '#38bdf8' }}>{h.train_loss.toFixed(4)}</td>
                      <td style={{ padding: '6px 10px', color: '#f59e0b' }}>{h.val_loss.toFixed(4)}</td>
                      <td style={{ padding: '6px 10px', color: '#10b981', fontWeight: 600 }}>{(h.val_metric * 100).toFixed(1)}%</td>
                      <td style={{ padding: '6px 10px', color: '#64748b', fontFamily: 'monospace' }}>{h.learning_rate.toExponential(2)}</td>
                    </tr>
                  ))}
                  {history.length === 0 && (
                    <tr>
                      <td colSpan={5} style={{ padding: '16px', textAlign: 'center', color: '#64748b' }}>
                        No epochs completed yet. Click "Start Model Training" to begin.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

        </div>

      </div>

    </div>
  );
};
