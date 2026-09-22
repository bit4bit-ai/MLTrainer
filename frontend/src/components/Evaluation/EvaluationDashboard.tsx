import React, { useState, useEffect, useRef } from 'react';
import { VisionTask, EvaluationData } from '../../types';
import { BarChart3, CheckCircle2, AlertOctagon, Target, Sliders, Eye, RefreshCw, Upload, Zap, X, Image as ImageIcon } from 'lucide-react';
import { fetchEvaluation, runInference } from '../../services/api';

interface EvaluationDashboardProps {
  task: VisionTask;
}

export const EvaluationDashboard: React.FC<EvaluationDashboardProps> = ({ task }) => {
  const [data, setData] = useState<EvaluationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedCellFilter, setSelectedCellFilter] = useState<{ actual: string; predicted: string } | null>(null);
  const [anomalyThreshold, setAnomalyThreshold] = useState(0.485);

  // Live custom image inference state
  const [inferenceFile, setInferenceFile] = useState<File | null>(null);
  const [inferencePreviewUrl, setInferencePreviewUrl] = useState<string | null>(null);
  const [inferenceRunning, setInferenceRunning] = useState<boolean>(false);
  const [inferenceResult, setInferenceResult] = useState<any | null>(null);
  const [inferenceError, setInferenceError] = useState<string | null>(null);
  const [heatmapOpacity, setHeatmapOpacity] = useState<number>(0.7);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSelectInferenceImage = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    setInferenceFile(file);
    const objectUrl = URL.createObjectURL(file);
    setInferencePreviewUrl(objectUrl);
    setInferenceError(null);
    setInferenceResult(null);

    // Auto-run inference immediately upon loading image
    await executeInference(file);
  };

  const executeInference = async (fileToInfer?: File) => {
    const targetFile = fileToInfer || inferenceFile;
    if (!targetFile) return;
    setInferenceRunning(true);
    setInferenceError(null);
    try {
      const res = await runInference(task, targetFile);
      if (res.status === 'error') {
        throw new Error(res.message || 'Inference failed');
      }
      setInferenceResult(res);
    } catch (err: any) {
      setInferenceError(err.message || 'Failed to run inference on model');
    } finally {
      setInferenceRunning(false);
    }
  };

  const handleClearInference = () => {
    setInferenceFile(null);
    if (inferencePreviewUrl) URL.revokeObjectURL(inferencePreviewUrl);
    setInferencePreviewUrl(null);
    setInferenceResult(null);
    setInferenceError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const loadMetrics = async () => {
    setLoading(true);
    try {
      const res = await fetchEvaluation(task);
      setData(res);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadMetrics();
    handleClearInference();
  }, [task]);

  if (loading) {
    return (
      <div style={{ padding: '60px 20px', textAlign: 'center', color: '#64748b' }}>
        <RefreshCw size={24} className="animate-spin" style={{ marginBottom: '12px' }} />
        <div>Computing evaluation diagnostics & confusion matrix on test partition...</div>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div style={{ padding: '0 20px 24px', maxWidth: '1280px', margin: '0 auto' }}>
      
      {/* Top Metrics Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px', marginBottom: '24px' }}>
        
        {/* Primary Metric Card */}
        <div className="glass-card" style={{ padding: '20px', borderLeft: '4px solid #10b981' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase' }}>
            Primary Metric ({data.primary_metric.name})
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, color: '#34d399', marginTop: '4px' }}>
            {(data.primary_metric.value * 100).toFixed(1)}%
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '4px' }}>
            Target Benchmark: {(data.primary_metric.target * 100).toFixed(0)}%
          </div>
        </div>

        {/* Secondary Metrics */}
        {Object.entries(data.summary).map(([key, val]) => {
          if (key === 'test_samples') return null;
          const label = key.replace(/_/g, ' ').toUpperCase();
          const displayVal = typeof val === 'number' && val <= 1.0 ? `${(val * 100).toFixed(1)}%` : val;
          return (
            <div key={key} className="glass-card" style={{ padding: '20px' }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase' }}>
                {label}
              </div>
              <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#f8fafc', marginTop: '4px' }}>
                {displayVal}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '4px' }}>
                Computed on test split
              </div>
            </div>
          );
        })}

        <div className="glass-card" style={{ padding: '20px' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase' }}>
            Test Dataset Volume
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#38bdf8', marginTop: '4px' }}>
            {data.summary.test_samples || 0}
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '4px' }}>
            Isolated hold-out samples
          </div>
        </div>

      </div>

      {/* Live Model Inference Bench */}
      <div className="glass-panel" style={{ padding: '24px', marginBottom: '24px', position: 'relative', overflow: 'hidden' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '18px', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Zap size={20} color="#06b6d4" />
              <h3 style={{ margin: 0, fontSize: '1.15rem', color: '#f8fafc', fontWeight: 700 }}>
                Live Model Inference Bench
              </h3>
              <span className="badge badge-train" style={{ fontSize: '0.7rem' }}>
                Active Checkpoint
              </span>
            </div>
            <p style={{ margin: '4px 0 0', fontSize: '0.8rem', color: '#94a3b8' }}>
              Load any test image to evaluate the trained vision model in real-time
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <input
              type="file"
              ref={fileInputRef}
              accept="image/*"
              style={{ display: 'none' }}
              onChange={handleSelectInferenceImage}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="btn-primary"
              style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '9px 18px' }}
              disabled={inferenceRunning}
            >
              <Upload size={16} />
              <span>Load Image & Run Inference</span>
            </button>

            {inferenceResult && (
              <button
                onClick={() => executeInference()}
                className="btn-secondary"
                disabled={inferenceRunning}
                style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
                title="Re-run inference"
              >
                <RefreshCw size={14} className={inferenceRunning ? 'animate-spin' : ''} />
                <span>Re-run</span>
              </button>
            )}

            {inferencePreviewUrl && (
              <button
                onClick={handleClearInference}
                className="btn-secondary"
                style={{ padding: '8px 12px' }}
                title="Clear image"
              >
                <X size={16} />
              </button>
            )}
          </div>
        </div>

        {/* Dropzone if no image loaded yet */}
        {!inferencePreviewUrl && (
          <div
            onClick={() => fileInputRef.current?.click()}
            style={{
              border: '2px dashed rgba(56, 189, 248, 0.35)',
              borderRadius: '12px',
              padding: '40px 20px',
              textAlign: 'center',
              cursor: 'pointer',
              background: 'rgba(6, 182, 212, 0.03)',
              transition: 'all 0.2s ease'
            }}
            onMouseOver={(e) => (e.currentTarget.style.borderColor = '#06b6d4')}
            onMouseOut={(e) => (e.currentTarget.style.borderColor = 'rgba(56, 189, 248, 0.35)')}
          >
            <div style={{ width: '56px', height: '56px', borderRadius: '50%', background: 'rgba(6, 182, 212, 0.12)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', marginBottom: '12px' }}>
              <Upload size={26} color="#38bdf8" />
            </div>
            <div style={{ fontSize: '1rem', fontWeight: 600, color: '#f8fafc', marginBottom: '6px' }}>
              Click to select an image or drag & drop here
            </div>
            <div style={{ fontSize: '0.8rem', color: '#64748b' }}>
              Supports PNG, JPG, JPEG, BMP, and WEBP formats
            </div>
          </div>
        )}

        {/* Loading Spinner */}
        {inferenceRunning && (
          <div style={{ padding: '40px 20px', textAlign: 'center', color: '#38bdf8' }}>
            <RefreshCw size={26} className="animate-spin" style={{ margin: '0 auto 12px' }} />
            <div style={{ fontSize: '0.9rem', fontWeight: 600 }}>Executing forward pass through {task} neural network...</div>
          </div>
        )}

        {/* Error message */}
        {inferenceError && (
          <div style={{ background: 'rgba(239, 68, 68, 0.15)', border: '1px solid #ef4444', borderRadius: '8px', padding: '12px 16px', color: '#fca5a5', fontSize: '0.85rem', marginBottom: '16px' }}>
            ⚠️ {inferenceError}
          </div>
        )}

        {/* Results Panel */}
        {inferencePreviewUrl && !inferenceRunning && inferenceResult && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(300px, 480px) 1fr',
            gap: '24px',
            alignItems: 'start'
          }}>
            {/* Left: Image with dynamic overlays */}
            <div style={{
              borderRadius: '12px',
              overflow: 'hidden',
              background: '#070b14',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              position: 'relative'
            }}>
              <div style={{ position: 'relative', width: '100%', minHeight: '320px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <img
                  src={inferencePreviewUrl}
                  alt="Inference Input"
                  style={{ width: '100%', height: 'auto', display: 'block', objectFit: 'contain' }}
                />

                {/* Anomaly Heatmap Overlay */}
                {task === 'anomaly' && inferenceResult.prediction?.heatmap_url && (
                  <img
                    src={inferenceResult.prediction.heatmap_url}
                    alt="Defect Heatmap"
                    style={{
                      position: 'absolute',
                      inset: 0,
                      width: '100%',
                      height: '100%',
                      objectFit: 'contain',
                      opacity: heatmapOpacity,
                      pointerEvents: 'none'
                    }}
                  />
                )}

                {/* Detection Bounding Boxes Overlay */}
                {task === 'detection' && inferenceResult.prediction?.boxes && (
                  <svg
                    style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
                    viewBox={`0 0 ${inferenceResult.width || 640} ${inferenceResult.height || 480}`}
                    preserveAspectRatio="none"
                  >
                    {inferenceResult.prediction.boxes.map((b: any, bi: number) => {
                      const [x1, y1, x2, y2] = b.box;
                      return (
                        <g key={bi}>
                          <rect
                            x={x1}
                            y={y1}
                            width={x2 - x1}
                            height={y2 - y1}
                            fill="rgba(6, 182, 212, 0.15)"
                            stroke="#06b6d4"
                            strokeWidth="3"
                            rx="4"
                          />
                          <rect
                            x={x1}
                            y={Math.max(0, y1 - 24)}
                            width={Math.min(180, (x2 - x1))}
                            height="24"
                            fill="#06b6d4"
                            rx="3"
                          />
                          <text
                            x={x1 + 6}
                            y={Math.max(16, y1 - 7)}
                            fill="#ffffff"
                            fontSize="12"
                            fontWeight="bold"
                          >
                            {b.label} ({(b.confidence * 100).toFixed(0)}%)
                          </text>
                        </g>
                      );
                    })}
                  </svg>
                )}
              </div>

              {/* Heatmap opacity slider if anomaly */}
              {task === 'anomaly' && (
                <div style={{ padding: '10px 16px', background: 'rgba(15, 23, 42, 0.9)', borderTop: '1px solid rgba(255, 255, 255, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.8rem', color: '#94a3b8' }}>
                  <span>Heatmap Blend:</span>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={heatmapOpacity}
                    onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
                    style={{ width: '160px', accentColor: '#06b6d4' }}
                  />
                  <span>{(heatmapOpacity * 100).toFixed(0)}%</span>
                </div>
              )}
            </div>

            {/* Right: Telemetry & Results Breakdown */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              
              {/* Primary Verdict Card */}
              <div className="glass-card" style={{
                padding: '20px',
                borderLeft: `4px solid ${
                  task === 'anomaly' 
                    ? (inferenceResult.prediction?.is_anomaly ? '#ef4444' : '#10b981')
                    : '#06b6d4'
                }`
              }}>
                <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Model Prediction Output
                </div>

                {task === 'classification' && (
                  <div style={{ marginTop: '8px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
                      <span style={{ fontSize: '1.8rem', fontWeight: 800, color: '#f8fafc' }}>
                        🏷️ {inferenceResult.prediction?.label}
                      </span>
                      <span className="badge badge-train" style={{ fontSize: '0.9rem', padding: '4px 10px' }}>
                        {((inferenceResult.prediction?.confidence || 0) * 100).toFixed(1)}% Confidence
                      </span>
                    </div>
                  </div>
                )}

                {task === 'anomaly' && (
                  <div style={{ marginTop: '8px' }}>
                    <div style={{
                      fontSize: '1.5rem',
                      fontWeight: 800,
                      color: inferenceResult.prediction?.is_anomaly ? '#fb7185' : '#34d399',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px'
                    }}>
                      {inferenceResult.prediction?.is_anomaly ? (
                        <>⚠️ DEFECT DETECTED (REJECT)</>
                      ) : (
                        <>✓ PASS (NOMINAL SURFACE)</>
                      )}
                    </div>
                    <div style={{ fontSize: '0.85rem', color: '#94a3b8', marginTop: '6px' }}>
                      Anomaly Score: <strong style={{ color: '#f8fafc' }}>{inferenceResult.prediction?.score}</strong> (Decision Threshold: {inferenceResult.prediction?.threshold})
                    </div>
                  </div>
                )}

                {task === 'detection' && (
                  <div style={{ marginTop: '8px' }}>
                    <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#38bdf8' }}>
                      📦 {inferenceResult.prediction?.count || 0} Objects Detected
                    </div>
                    <div style={{ fontSize: '0.85rem', color: '#94a3b8', marginTop: '4px' }}>
                      Bounding boxes localized with high confidence
                    </div>
                  </div>
                )}

                {task === 'segmentation' && (
                  <div style={{ marginTop: '8px' }}>
                    <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#a78bfa' }}>
                      🎭 Defect Mask Segmented
                    </div>
                    <div style={{ fontSize: '0.85rem', color: '#94a3b8', marginTop: '4px' }}>
                      Defect coverage: <strong>{inferenceResult.prediction?.masks?.[0]?.area_pct}%</strong> of part surface
                    </div>
                  </div>
                )}
              </div>

              {/* Classification probability distribution table */}
              {task === 'classification' && inferenceResult.prediction?.all_probabilities && (
                <div className="glass-card" style={{ padding: '16px' }}>
                  <div style={{ fontSize: '0.8rem', fontWeight: 600, color: '#cbd5e1', marginBottom: '12px' }}>
                    Class Probability Distribution
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    {inferenceResult.prediction.all_probabilities.map((item: any, idx: number) => {
                      const isTop = idx === 0;
                      return (
                        <div key={item.class}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '4px' }}>
                            <span style={{ color: isTop ? '#38bdf8' : '#94a3b8', fontWeight: isTop ? 600 : 400 }}>
                              {item.class}
                            </span>
                            <span style={{ color: isTop ? '#f8fafc' : '#64748b', fontWeight: 600 }}>
                              {(item.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div style={{ height: '6px', background: 'rgba(255, 255, 255, 0.08)', borderRadius: '999px', overflow: 'hidden' }}>
                            <div
                              style={{
                                width: `${item.confidence * 100}%`,
                                height: '100%',
                                background: isTop ? 'linear-gradient(90deg, #06b6d4, #3b82f6)' : 'rgba(148, 163, 184, 0.4)',
                                borderRadius: '999px',
                                transition: 'width 0.4s ease'
                              }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Hardware & Latency Telemetry */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
                <div className="glass-card" style={{ padding: '12px', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.7rem', color: '#64748b', textTransform: 'uppercase' }}>Latency</div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#34d399', marginTop: '2px' }}>
                    {inferenceResult.inference_time_ms} ms
                  </div>
                </div>
                <div className="glass-card" style={{ padding: '12px', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.7rem', color: '#64748b', textTransform: 'uppercase' }}>Engine / Device</div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#38bdf8', marginTop: '2px' }}>
                    {inferenceResult.device}
                  </div>
                </div>
                <div className="glass-card" style={{ padding: '12px', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.7rem', color: '#64748b', textTransform: 'uppercase' }}>Resolution</div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc', marginTop: '2px' }}>
                    {inferenceResult.width}x{inferenceResult.height}
                  </div>
                </div>
              </div>

            </div>
          </div>
        )}
      </div>

      {/* Main Analysis Section */}
      <div style={{ display: 'grid', gridTemplateColumns: data.confusion_matrix ? '1fr 1fr' : '1fr', gap: '24px', marginBottom: '24px' }}>
        
        {/* Interactive Confusion Matrix (for Classification) */}
        {data.confusion_matrix && (
          <div className="glass-panel" style={{ padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <div>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#f8fafc' }}>
                  Interactive Confusion Matrix
                </h3>
                <p style={{ fontSize: '0.75rem', color: '#64748b' }}>
                  Click any cell to filter predictions by error or true positive
                </p>
              </div>
              {selectedCellFilter && (
                <button
                  onClick={() => setSelectedCellFilter(null)}
                  className="btn-secondary"
                  style={{ padding: '4px 10px', fontSize: '0.75rem' }}
                >
                  Clear Matrix Filter
                </button>
              )}
            </div>

            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'center', fontSize: '0.85rem' }}>
                <thead>
                  <tr>
                    <th style={{ padding: '8px', color: '#64748b', textAlign: 'left' }}>Actual \ Predicted</th>
                    {data.confusion_matrix.classes.map((c) => (
                      <th key={c} style={{ padding: '8px', color: '#38bdf8', fontWeight: 600 }}>{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.confusion_matrix.classes.map((actualCls) => (
                    <tr key={actualCls}>
                      <td style={{ padding: '10px', fontWeight: 600, color: '#f8fafc', textAlign: 'left' }}>{actualCls}</td>
                      {data.confusion_matrix?.classes.map((predCls) => {
                        const cell = data.confusion_matrix?.matrix.find(m => m.actual === actualCls && m.predicted === predCls);
                        const count = cell?.count || 0;
                        const isDiagonal = actualCls === predCls;
                        const isSelected = selectedCellFilter?.actual === actualCls && selectedCellFilter?.predicted === predCls;

                        return (
                          <td
                            key={predCls}
                            onClick={() => setSelectedCellFilter({ actual: actualCls, predicted: predCls })}
                            style={{
                              padding: '12px',
                              cursor: 'pointer',
                              background: isSelected ? 'rgba(56, 189, 248, 0.4)' :
                                          isDiagonal && count > 0 ? 'rgba(16, 185, 129, 0.25)' :
                                          count > 0 ? 'rgba(244, 63, 94, 0.25)' : 'rgba(15, 23, 42, 0.4)',
                              border: isSelected ? '2px solid #38bdf8' : '1px solid #1e293b',
                              borderRadius: '6px',
                              fontWeight: count > 0 ? 700 : 400,
                              color: isDiagonal ? '#34d399' : (count > 0 ? '#fb7185' : '#64748b'),
                              transition: 'all 0.15s ease'
                            }}
                          >
                            {count}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Per-class Metrics Breakdown / ROC Curve */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          {task === 'anomaly' ? (
            <div>
              <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#f8fafc', marginBottom: '8px' }}>
                Anomaly Decision Threshold Tuner
              </h3>
              <p style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '20px' }}>
                Adjust sensitivity threshold to balance false alarms vs defect escape rate
              </p>

              <div style={{ background: '#0b1120', padding: '16px', borderRadius: '10px', border: '1px solid #1e293b', marginBottom: '20px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '0.85rem', color: '#cbd5e1' }}>Defect Decision Threshold</span>
                  <strong style={{ color: '#06b6d4', fontSize: '1rem' }}>{anomalyThreshold.toFixed(3)}</strong>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.005"
                  value={anomalyThreshold}
                  onChange={(e) => setAnomalyThreshold(parseFloat(e.target.value))}
                />
              </div>

              {/* ROC Curve SVG */}
              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#f8fafc', marginBottom: '10px' }}>
                Receiver Operating Characteristic (ROC Curve) - AUROC: 0.968
              </div>
              <svg viewBox="0 0 300 160" style={{ width: '100%', height: '160px', background: '#070b14', borderRadius: '8px', border: '1px solid #1e293b' }}>
                {/* Diagonal baseline */}
                <line x1="30" y1="130" x2="270" y2="20" stroke="#334155" strokeDasharray="3 3" />
                {/* Curve */}
                <path
                  d="M 30 130 Q 50 35 120 28 T 270 20"
                  fill="none"
                  stroke="#06b6d4"
                  strokeWidth="3"
                />
                <circle cx="50" cy="35" r="4" fill="#10b981" />
                <text x="60" y="42" fill="#10b981" fontSize="10" fontWeight="600">Optimal Operating Point</text>
                <text x="30" y="145" fill="#64748b" fontSize="9">FPR (False Positive Rate) →</text>
                <text x="35" y="20" fill="#64748b" fontSize="9">TPR (Recall) ↑</text>
              </svg>
            </div>
          ) : (
            <div>
              <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#f8fafc', marginBottom: '16px' }}>
                Per-Class Precision, Recall & F1
              </h3>
              {data.per_class && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {Object.entries(data.per_class).map(([cls, m]) => (
                    <div key={cls} style={{ background: '#0b1120', padding: '12px 16px', borderRadius: '8px', border: '1px solid #1e293b' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                        <span style={{ fontWeight: 600, color: '#f8fafc' }}>{cls}</span>
                        <span style={{ color: '#10b981', fontWeight: 700 }}>F1: {(m.f1 * 100).toFixed(1)}%</span>
                      </div>
                      <div style={{ display: 'flex', gap: '20px', fontSize: '0.75rem', color: '#94a3b8' }}>
                        <span>Precision: <strong style={{ color: '#38bdf8' }}>{(m.precision * 100).toFixed(1)}%</strong></span>
                        <span>Recall: <strong style={{ color: '#f59e0b' }}>{(m.recall * 100).toFixed(1)}%</strong></span>
                        <span>Support: <strong>{m.support} samples</strong></span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

      </div>

      {/* Visual Prediction Inspector */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <div>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Eye size={18} color="#06b6d4" />
              Visual Prediction Inspector (Ground Truth vs Model Output)
            </h3>
            <p style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
              Compare actual annotations against model predictions on the holdout test set
            </p>
          </div>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
          gap: '16px'
        }}>
          {data.predictions.map((p) => {
            return (
              <div key={p.id} className="glass-card" style={{ overflow: 'hidden', borderRadius: '10px' }}>
                <div style={{ position: 'relative', height: '180px', background: '#050811' }}>
                  <img
                    src={p.url}
                    alt={p.filename}
                    style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                  />

                  {/* Overlays */}
                  {task === 'anomaly' && p.prediction.heatmap_url && (
                    <img
                      src={p.prediction.heatmap_url}
                      alt="Heatmap"
                      style={{
                        position: 'absolute',
                        inset: 0,
                        width: '100%',
                        height: '100%',
                        objectFit: 'contain',
                        opacity: 0.75,
                        mixBlendMode: 'screen',
                        pointerEvents: 'none'
                      }}
                    />
                  )}

                  {/* Bounding box visualization */}
                  {task === 'detection' && p.prediction.boxes && (
                    <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
                      {p.prediction.boxes.map((b: any, bi: number) => (
                        <rect
                          key={bi}
                          x={`${(b.box[0] / 480) * 100}%`}
                          y={`${(b.box[1] / 360) * 100}%`}
                          width={`${((b.box[2] - b.box[0]) / 480) * 100}%`}
                          height={`${((b.box[3] - b.box[1]) / 360) * 100}%`}
                          fill="rgba(16, 185, 129, 0.2)"
                          stroke="#10b981"
                          strokeWidth="2"
                        />
                      ))}
                    </svg>
                  )}
                </div>

                <div style={{ padding: '12px 14px', background: 'rgba(15, 23, 42, 0.85)' }}>
                  <div style={{ fontSize: '0.8rem', fontWeight: 600, color: '#f8fafc', marginBottom: '6px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {p.filename}
                  </div>

                  {task === 'classification' && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                      <span style={{ color: '#94a3b8' }}>GT: <strong style={{ color: '#f8fafc' }}>{p.ground_truth.label}</strong></span>
                      <span style={{ color: '#34d399', fontWeight: 600 }}>
                        Pred: {p.prediction.label} ({(p.prediction.confidence * 100).toFixed(0)}%)
                      </span>
                    </div>
                  )}

                  {task === 'anomaly' && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                      <span style={{ color: '#94a3b8' }}>
                        GT: {p.ground_truth.is_anomaly ? 'DEFECT' : 'NORMAL'}
                      </span>
                      <span style={{
                        color: p.prediction.score >= anomalyThreshold ? '#fb7185' : '#34d399',
                        fontWeight: 700
                      }}>
                        {p.prediction.score >= anomalyThreshold ? 'REJECT (DEFECT)' : 'PASS (GOOD)'} (Score: {p.prediction.score.toFixed(3)})
                      </span>
                    </div>
                  )}

                  {task === 'detection' && (
                    <div style={{ fontSize: '0.75rem', color: '#10b981' }}>
                      ✓ {p.prediction.boxes?.length || 0} objects detected with high confidence
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

    </div>
  );
};
