import React, { useState } from 'react';
import { DatasetItem, VisionTask } from '../../types';
import { X, Tag, Box, AlertTriangle, Scan, Check, Trash2 } from 'lucide-react';

interface ImageModalProps {
  item: DatasetItem;
  task: VisionTask;
  classes: string[];
  onClose: () => void;
  onUpdate: (itemId: string, updates: any) => Promise<void>;
  onDelete: (itemId: string) => Promise<void>;
}

export const ImageModal: React.FC<ImageModalProps> = ({
  item,
  task,
  classes,
  onClose,
  onUpdate,
  onDelete
}) => {
  const [currentLabel, setCurrentLabel] = useState(item.label || classes[0]);
  const [currentSplit, setCurrentSplit] = useState(item.split || 'train');
  const [showOverlays, setShowOverlays] = useState(true);
  const [isAnomaly, setIsAnomaly] = useState(item.is_anomaly || false);
  const [anomalyScore, setAnomalyScore] = useState(item.anomaly_score || 0.0);
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    try {
      await onUpdate(item.id, {
        label: currentLabel,
        split: currentSplit,
        is_anomaly: isAnomaly,
        anomaly_score: anomalyScore
      });
      onClose();
    } catch (e) {
      console.error(e);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (confirm('Are you sure you want to delete this sample from dataset?')) {
      await onDelete(item.id);
      onClose();
    }
  };

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(0, 0, 0, 0.85)',
      backdropFilter: 'blur(8px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      padding: '24px'
    }}>
      <div className="glass-panel" style={{
        maxWidth: '920px',
        width: '100%',
        maxHeight: '90vh',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        border: '1px solid rgba(255,255,255,0.15)'
      }}>
        {/* Header */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '16px 20px',
          borderBottom: '1px solid rgba(255, 255, 255, 0.08)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span style={{ fontWeight: 600, fontSize: '1rem', color: '#f8fafc' }}>{item.filename}</span>
            <span className={`badge badge-${item.split}`}>{item.split}</span>
            <span style={{ fontSize: '0.75rem', color: '#64748b' }}>{item.width} × {item.height} px</span>
          </div>
          <button onClick={onClose} style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: '#94a3b8' }}>
            <X size={20} />
          </button>
        </div>

        {/* Content Body */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: '20px', padding: '20px', overflowY: 'auto' }}>
          
          {/* Main Visual Image & Annotation Canvas Overlay */}
          <div style={{
            position: 'relative',
            background: '#040711',
            borderRadius: '10px',
            border: '1px solid #1e293b',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            overflow: 'hidden',
            minHeight: '360px'
          }}>
            <img
              src={item.url}
              alt={item.filename}
              style={{
                maxWidth: '100%',
                maxHeight: '440px',
                objectFit: 'contain',
                display: 'block'
              }}
            />

            {/* Heatmap overlay for anomaly */}
            {task === 'anomaly' && item.heatmap_url && showOverlays && (
              <img
                src={item.heatmap_url}
                alt="Anomaly Heatmap"
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

            {/* Bounding box overlays for detection */}
            {task === 'detection' && item.boxes && showOverlays && (
              <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
                {item.boxes.map((b, i) => {
                  const [x1, y1, x2, y2] = b.box;
                  // normalize coordinates to viewBox ratio
                  const scaleX = 100 / item.width;
                  const scaleY = 100 / item.height;
                  return (
                    <g key={i}>
                      <rect
                        x={`${x1 * scaleX}%`}
                        y={`${y1 * scaleY}%`}
                        width={`${(x2 - x1) * scaleX}%`}
                        height={`${(y2 - y1) * scaleY}%`}
                        fill="rgba(6, 182, 212, 0.15)"
                        stroke="#06b6d4"
                        strokeWidth="2"
                        strokeDasharray="4 2"
                      />
                      <text
                        x={`${x1 * scaleX}%`}
                        y={`${Math.max(12, y1 * scaleY - 4)}%`}
                        fill="#06b6d4"
                        fontSize="11"
                        fontWeight="600"
                        fontFamily="monospace"
                      >
                        {b.label} {b.confidence ? `(${(b.confidence * 100).toFixed(0)}%)` : ''}
                      </text>
                    </g>
                  );
                })}
              </svg>
            )}

            {/* Segmentation polygon overlay */}
            {task === 'segmentation' && item.segmentation && showOverlays && (
              <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
                {item.segmentation.map((s, i) => {
                  const pointsStr = s.polygon.map(p => `${(p.x / item.width) * 100}%,${(p.y / item.height) * 100}%`).join(' ');
                  return (
                    <polygon
                      key={i}
                      points={pointsStr}
                      fill={s.color ? `${s.color}40` : 'rgba(99, 102, 241, 0.25)'}
                      stroke={s.color || '#6366f1'}
                      strokeWidth="2"
                    />
                  );
                })}
              </svg>
            )}
          </div>

          {/* Sidebar / Inspector Panel */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            
            {/* Overlay toggle */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', background: '#0b1120', borderRadius: '8px', border: '1px solid #1e293b' }}>
              <span style={{ fontSize: '0.825rem', color: '#94a3b8' }}>Visual Overlays</span>
              <button
                onClick={() => setShowOverlays(!showOverlays)}
                className="btn-secondary"
                style={{ padding: '4px 10px', fontSize: '0.75rem' }}
              >
                {showOverlays ? 'Hide' : 'Show'}
              </button>
            </div>

            {/* Split Assignment */}
            <div>
              <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', marginBottom: '6px', textTransform: 'uppercase' }}>
                Dataset Partition (Split)
              </label>
              <select
                value={currentSplit}
                onChange={(e) => setCurrentSplit(e.target.value as any)}
                style={{ width: '100%' }}
              >
                <option value="train">Train Partition</option>
                <option value="val">Validation Partition</option>
                <option value="test">Test Partition</option>
              </select>
            </div>

            {/* Categorization Tag */}
            {task === 'classification' && (
              <div>
                <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', marginBottom: '6px', textTransform: 'uppercase' }}>
                  Target Class Label
                </label>
                <select
                  value={currentLabel}
                  onChange={(e) => setCurrentLabel(e.target.value)}
                  style={{ width: '100%' }}
                >
                  {classes.map((c) => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </div>
            )}

            {/* Anomaly Inspection Controls */}
            {task === 'anomaly' && (
              <div style={{ background: '#0b1120', padding: '14px', borderRadius: '8px', border: '1px solid #1e293b' }}>
                <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', marginBottom: '8px', textTransform: 'uppercase' }}>
                  Defect Ground Truth
                </label>
                <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
                  <button
                    onClick={() => { setIsAnomaly(false); setAnomalyScore(0.05); }}
                    style={{
                      flex: 1,
                      padding: '8px',
                      borderRadius: '6px',
                      border: '1px solid',
                      fontSize: '0.8rem',
                      fontWeight: 600,
                      cursor: 'pointer',
                      background: !isAnomaly ? 'rgba(16, 185, 129, 0.2)' : 'transparent',
                      color: !isAnomaly ? '#34d399' : '#64748b',
                      borderColor: !isAnomaly ? '#10b981' : '#1e293b'
                    }}
                  >
                    Normal
                  </button>
                  <button
                    onClick={() => { setIsAnomaly(true); setAnomalyScore(0.85); }}
                    style={{
                      flex: 1,
                      padding: '8px',
                      borderRadius: '6px',
                      border: '1px solid',
                      fontSize: '0.8rem',
                      fontWeight: 600,
                      cursor: 'pointer',
                      background: isAnomaly ? 'rgba(244, 63, 94, 0.2)' : 'transparent',
                      color: isAnomaly ? '#fb7185' : '#64748b',
                      borderColor: isAnomaly ? '#f43f5e' : '#1e293b'
                    }}
                  >
                    Anomaly
                  </button>
                </div>
                <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                  Score: <strong style={{ color: isAnomaly ? '#fb7185' : '#34d399' }}>{anomalyScore.toFixed(3)}</strong>
                </div>
              </div>
            )}

            {/* Object Detection Summary */}
            {task === 'detection' && item.boxes && (
              <div>
                <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', marginBottom: '6px', textTransform: 'uppercase' }}>
                  Annotated Objects ({item.boxes.length})
                </label>
                <div style={{ maxHeight: '120px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  {item.boxes.map((b, i) => (
                    <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 10px', background: '#0b1120', borderRadius: '6px', fontSize: '0.75rem', border: '1px solid #1e293b' }}>
                      <span style={{ color: '#38bdf8', fontWeight: 600 }}>{b.label}</span>
                      <span style={{ color: '#64748b' }}>[{b.box.join(', ')}]</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Actions */}
            <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <button onClick={handleSave} disabled={saving} className="btn-primary" style={{ justifyContent: 'center' }}>
                <Check size={16} />
                {saving ? 'Saving...' : 'Apply Changes'}
              </button>
              <button onClick={handleDelete} className="btn-danger" style={{ justifyContent: 'center' }}>
                <Trash2 size={16} />
                Delete Sample
              </button>
            </div>

          </div>

        </div>
      </div>
    </div>
  );
};
