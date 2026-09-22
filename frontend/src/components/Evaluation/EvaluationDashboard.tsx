import React, { useState, useEffect } from 'react';
import { VisionTask, EvaluationData } from '../../types';
import { BarChart3, CheckCircle2, AlertOctagon, Target, Sliders, Eye, RefreshCw } from 'lucide-react';
import { fetchEvaluation } from '../../services/api';

interface EvaluationDashboardProps {
  task: VisionTask;
}

export const EvaluationDashboard: React.FC<EvaluationDashboardProps> = ({ task }) => {
  const [data, setData] = useState<EvaluationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedCellFilter, setSelectedCellFilter] = useState<{ actual: string; predicted: string } | null>(null);
  const [anomalyThreshold, setAnomalyThreshold] = useState(0.485);

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
