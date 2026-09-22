import React, { useState } from 'react';
import { VisionTask, SplitDistribution } from '../../types';
import { Sliders, RefreshCw, CheckCircle2, ShieldCheck, Sparkles, BarChart2 } from 'lucide-react';
import { splitDataset } from '../../services/api';

interface SplitEditorProps {
  task: VisionTask;
  distribution: SplitDistribution | null;
  onSplitCompleted: () => Promise<void>;
}

export const SplitEditor: React.FC<SplitEditorProps> = ({
  task,
  distribution,
  onSplitCompleted
}) => {
  const [trainPct, setTrainPct] = useState(70);
  const [valPct, setValPct] = useState(15);
  const [testPct, setTestPct] = useState(15);
  const [stratify, setStratify] = useState(true);
  const [seed, setSeed] = useState(42);
  const [splitting, setSplitting] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  const applyPreset = (train: number, val: number, test: number) => {
    setTrainPct(train);
    setValPct(val);
    setTestPct(test);
  };

  const handleExecuteSplit = async () => {
    setSplitting(true);
    setStatusMsg(null);
    try {
      await splitDataset(task, trainPct, valPct, testPct, stratify, seed);
      setStatusMsg('Dataset successfully partitioned!');
      await onSplitCompleted();
      setTimeout(() => setStatusMsg(null), 3000);
    } catch (e: any) {
      console.error(e);
      setStatusMsg('Error partitioning dataset');
    } finally {
      setSplitting(false);
    }
  };

  const total = trainPct + valPct + testPct;
  const trainNorm = total > 0 ? (trainPct / total) * 100 : 70;
  const valNorm = total > 0 ? (valPct / total) * 100 : 15;
  const testNorm = total > 0 ? (testPct / total) * 100 : 15;

  return (
    <div style={{ padding: '0 20px 24px', maxWidth: '1100px', margin: '0 auto' }}>
      
      {/* Overview Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px', marginBottom: '20px' }}>
        
        <div className="glass-card" style={{ padding: '16px 20px', borderLeft: '4px solid #3b82f6' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase' }}>Train Partition</div>
          <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#93c5fd', marginTop: '4px' }}>
            {distribution?.counts.train ?? 0} <span style={{ fontSize: '0.9rem', color: '#64748b' }}>samples</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '4px' }}>
            {distribution?.percentages.train ?? 0}% of total dataset
          </div>
        </div>

        <div className="glass-card" style={{ padding: '16px 20px', borderLeft: '4px solid #f59e0b' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase' }}>Validation Partition</div>
          <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#fcd34d', marginTop: '4px' }}>
            {distribution?.counts.val ?? 0} <span style={{ fontSize: '0.9rem', color: '#64748b' }}>samples</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '4px' }}>
            {distribution?.percentages.val ?? 0}% of total dataset
          </div>
        </div>

        <div className="glass-card" style={{ padding: '16px 20px', borderLeft: '4px solid #10b981' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase' }}>Test Partition</div>
          <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#6ee7b7', marginTop: '4px' }}>
            {distribution?.counts.test ?? 0} <span style={{ fontSize: '0.9rem', color: '#64748b' }}>samples</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '4px' }}>
            {distribution?.percentages.test ?? 0}% of total dataset
          </div>
        </div>

      </div>

      {/* Main Split Configuration Card */}
      <div className="glass-panel" style={{ padding: '24px', marginBottom: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Sliders size={20} color="#06b6d4" />
              Dataset Partition Ratio Configurator
            </h3>
            <p style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
              Define train, validation, and test subsets with optional stratified balancing
            </p>
          </div>

          {/* Quick Presets */}
          <div style={{ display: 'flex', gap: '8px' }}>
            <button onClick={() => applyPreset(70, 15, 15)} className="btn-secondary" style={{ padding: '4px 10px', fontSize: '0.75rem' }}>
              70 / 15 / 15 (Standard)
            </button>
            <button onClick={() => applyPreset(80, 10, 10)} className="btn-secondary" style={{ padding: '4px 10px', fontSize: '0.75rem' }}>
              80 / 10 / 10 (Deep Train)
            </button>
            <button onClick={() => applyPreset(60, 20, 20)} className="btn-secondary" style={{ padding: '4px 10px', fontSize: '0.75rem' }}>
              60 / 20 / 20 (Balanced)
            </button>
          </div>
        </div>

        {/* Visual Multi-Segment Bar */}
        <div style={{
          height: '18px',
          borderRadius: '999px',
          overflow: 'hidden',
          display: 'flex',
          marginBottom: '28px',
          background: '#0b1120',
          border: '1px solid #1e293b'
        }}>
          <div style={{ width: `${trainNorm}%`, background: '#3b82f6', transition: 'width 0.2s ease' }} title={`Train: ${trainNorm.toFixed(1)}%`} />
          <div style={{ width: `${valNorm}%`, background: '#f59e0b', transition: 'width 0.2s ease' }} title={`Val: ${valNorm.toFixed(1)}%`} />
          <div style={{ width: `${testNorm}%`, background: '#10b981', transition: 'width 0.2s ease' }} title={`Test: ${testNorm.toFixed(1)}%`} />
        </div>

        {/* Sliders Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '24px', marginBottom: '24px' }}>
          
          {/* Train Slider */}
          <div style={{ background: '#0b1120', padding: '16px', borderRadius: '10px', border: '1px solid #1e293b' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.825rem', fontWeight: 600, color: '#93c5fd' }}>Train Ratio</span>
              <strong style={{ color: '#3b82f6' }}>{trainPct}%</strong>
            </div>
            <input
              type="range"
              min="30"
              max="90"
              value={trainPct}
              onChange={(e) => setTrainPct(parseInt(e.target.value))}
            />
          </div>

          {/* Val Slider */}
          <div style={{ background: '#0b1120', padding: '16px', borderRadius: '10px', border: '1px solid #1e293b' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.825rem', fontWeight: 600, color: '#fcd34d' }}>Validation Ratio</span>
              <strong style={{ color: '#f59e0b' }}>{valPct}%</strong>
            </div>
            <input
              type="range"
              min="5"
              max="40"
              value={valPct}
              onChange={(e) => setValPct(parseInt(e.target.value))}
            />
          </div>

          {/* Test Slider */}
          <div style={{ background: '#0b1120', padding: '16px', borderRadius: '10px', border: '1px solid #1e293b' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.825rem', fontWeight: 600, color: '#6ee7b7' }}>Test Ratio</span>
              <strong style={{ color: '#10b981' }}>{testPct}%</strong>
            </div>
            <input
              type="range"
              min="5"
              max="40"
              value={testPct}
              onChange={(e) => setTestPct(parseInt(e.target.value))}
            />
          </div>

        </div>

        {/* Options Bar */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px', paddingTop: '16px', borderTop: '1px solid #1e293b' }}>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '20px', flexWrap: 'wrap' }}>
            {/* Stratification Toggle */}
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '0.85rem', color: '#e2e8f0' }}>
              <input
                type="checkbox"
                checked={stratify}
                onChange={(e) => setStratify(e.target.checked)}
                style={{ width: '16px', height: '16px', accentColor: '#06b6d4' }}
              />
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <ShieldCheck size={16} color="#06b6d4" />
                Stratified Sampling (Preserve class balance)
              </span>
            </label>

            {/* Seed input */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Random Seed:</span>
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(parseInt(e.target.value) || 0)}
                style={{ width: '80px', padding: '4px 8px' }}
              />
            </div>
          </div>

          {/* Action Button */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            {statusMsg && (
              <span style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8rem', color: '#34d399' }}>
                <CheckCircle2 size={16} />
                {statusMsg}
              </span>
            )}
            <button onClick={handleExecuteSplit} disabled={splitting} className="btn-primary">
              <RefreshCw size={16} className={splitting ? 'animate-spin' : ''} />
              {splitting ? 'Splitting Dataset...' : 'Execute Partitioning'}
            </button>
          </div>

        </div>

      </div>

      {/* Class Balance per Split Table */}
      {distribution?.class_distribution && (
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#f8fafc', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <BarChart2 size={18} color="#6366f1" />
            Class Distribution per Partition
          </h4>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #1e293b', color: '#94a3b8', textAlign: 'left' }}>
                  <th style={{ padding: '10px 14px' }}>Class / Defect Category</th>
                  <th style={{ padding: '10px 14px', color: '#93c5fd' }}>Train Samples</th>
                  <th style={{ padding: '10px 14px', color: '#fcd34d' }}>Val Samples</th>
                  <th style={{ padding: '10px 14px', color: '#6ee7b7' }}>Test Samples</th>
                  <th style={{ padding: '10px 14px' }}>Partition Balance Preview</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(distribution.class_distribution).map(([clsName, dist]) => {
                  const clsTotal = dist.train + dist.val + dist.test;
                  const tP = clsTotal > 0 ? (dist.train / clsTotal) * 100 : 0;
                  const vP = clsTotal > 0 ? (dist.val / clsTotal) * 100 : 0;
                  const teP = clsTotal > 0 ? (dist.test / clsTotal) * 100 : 0;

                  return (
                    <tr key={clsName} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                      <td style={{ padding: '12px 14px', fontWeight: 600, color: '#f8fafc' }}>{clsName}</td>
                      <td style={{ padding: '12px 14px', color: '#38bdf8' }}>{dist.train}</td>
                      <td style={{ padding: '12px 14px', color: '#f59e0b' }}>{dist.val}</td>
                      <td style={{ padding: '12px 14px', color: '#10b981' }}>{dist.test}</td>
                      <td style={{ padding: '12px 14px', minWidth: '180px' }}>
                        <div style={{ height: '8px', borderRadius: '4px', overflow: 'hidden', display: 'flex', background: '#0b1120' }}>
                          <div style={{ width: `${tP}%`, background: '#3b82f6' }} />
                          <div style={{ width: `${vP}%`, background: '#f59e0b' }} />
                          <div style={{ width: `${teP}%`, background: '#10b981' }} />
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

    </div>
  );
};
