import React from 'react';
import { VisionTask, ActiveTab } from '../types';
import { Layers, Box, AlertTriangle, Scan, Image as ImageIcon, Sliders, Activity, BarChart3, Download, Cpu } from 'lucide-react';

interface NavbarProps {
  currentTask: VisionTask;
  onSelectTask: (task: VisionTask) => void;
  activeTab: ActiveTab;
  onSelectTab: (tab: ActiveTab) => void;
  trainingState: string;
  device: string;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentTask,
  onSelectTask,
  activeTab,
  onSelectTab,
  trainingState,
  device
}) => {
  const tasks: { id: VisionTask; label: string; icon: React.ReactNode }[] = [
    { id: 'classification', label: 'Categorization', icon: <Layers size={16} /> },
    { id: 'detection', label: 'Object Detection', icon: <Box size={16} /> },
    { id: 'anomaly', label: 'Anomaly Detection', icon: <AlertTriangle size={16} /> },
    { id: 'segmentation', label: 'Segmentation', icon: <Scan size={16} /> }
  ];

  const tabs: { id: ActiveTab; label: string; icon: React.ReactNode }[] = [
    { id: 'gallery', label: 'Image Gallery', icon: <ImageIcon size={16} /> },
    { id: 'splitting', label: 'Dataset Splitting', icon: <Sliders size={16} /> },
    { id: 'training', label: 'Training Studio', icon: <Activity size={16} /> },
    { id: 'evaluation', label: 'Evaluation & Test', icon: <BarChart3 size={16} /> },
    { id: 'export', label: 'Export & Inference', icon: <Download size={16} /> }
  ];

  return (
    <header className="glass-panel" style={{ margin: '16px 20px', padding: '12px 20px', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
        
        {/* Logo & Task Selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{
              width: '38px',
              height: '38px',
              borderRadius: '10px',
              background: 'linear-gradient(135deg, #06b6d4, #6366f1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 0 15px rgba(6,182,212,0.4)'
            }}>
              <Scan size={22} color="#ffffff" />
            </div>
            <div>
              <div style={{ fontWeight: 800, fontSize: '1.25rem', letterSpacing: '-0.02em', background: 'linear-gradient(to right, #ffffff, #94a3b8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                MLTrainer
              </div>
              <div style={{ fontSize: '0.72rem', color: '#64748b', fontWeight: 500, letterSpacing: '0.04em' }}>
                MACHINE VISION STUDIO
              </div>
            </div>
          </div>

          {/* Task selector pill bar */}
          <div style={{
            display: 'flex',
            background: 'rgba(15, 23, 42, 0.9)',
            padding: '4px',
            borderRadius: '10px',
            border: '1px solid #1e293b'
          }}>
            {tasks.map((t) => {
              const active = currentTask === t.id;
              return (
                <button
                  key={t.id}
                  onClick={() => onSelectTask(t.id)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '6px 14px',
                    borderRadius: '8px',
                    fontSize: '0.825rem',
                    fontWeight: active ? 600 : 500,
                    border: 'none',
                    cursor: 'pointer',
                    background: active ? 'linear-gradient(135deg, rgba(6,182,212,0.2), rgba(99,102,241,0.2))' : 'transparent',
                    color: active ? '#38bdf8' : '#94a3b8',
                    borderBottom: active ? '2px solid #38bdf8' : '2px solid transparent',
                    transition: 'all 0.15s ease'
                  }}
                >
                  {t.icon}
                  {t.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Telemetry Status Badges */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            background: 'rgba(30, 41, 59, 0.6)',
            padding: '5px 10px',
            borderRadius: '6px',
            border: '1px solid rgba(255,255,255,0.06)',
            fontSize: '0.75rem',
            color: '#94a3b8'
          }}>
            <Cpu size={14} color="#38bdf8" />
            <span>ACCEL: <strong style={{ color: '#f8fafc' }}>{device.toUpperCase()}</strong></span>
          </div>

          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            background: 'rgba(30, 41, 59, 0.6)',
            padding: '5px 12px',
            borderRadius: '6px',
            border: '1px solid rgba(255,255,255,0.06)',
            fontSize: '0.75rem',
            fontWeight: 600
          }}>
            {trainingState === 'running' && <span className="live-pulse" />}
            {trainingState === 'idle' && <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#64748b' }} />}
            {trainingState === 'paused' && <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#f59e0b' }} />}
            {trainingState === 'completed' && <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#10b981' }} />}
            <span style={{
              color: trainingState === 'running' ? '#34d399' :
                     trainingState === 'paused' ? '#fbbf24' :
                     trainingState === 'completed' ? '#6ee7b7' : '#94a3b8',
              textTransform: 'uppercase'
            }}>
              {trainingState}
            </span>
          </div>
        </div>
      </div>

      {/* Navigation Sub-Tabs */}
      <nav style={{
        display: 'flex',
        gap: '8px',
        marginTop: '16px',
        paddingTop: '12px',
        borderTop: '1px solid rgba(255, 255, 255, 0.05)',
        overflowX: 'auto'
      }}>
        {tabs.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => onSelectTab(tab.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '8px 16px',
                borderRadius: '8px',
                fontSize: '0.875rem',
                fontWeight: isActive ? 600 : 500,
                border: 'none',
                cursor: 'pointer',
                background: isActive ? '#1e293b' : 'transparent',
                color: isActive ? '#f8fafc' : '#94a3b8',
                boxShadow: isActive ? 'inset 0 0 0 1px rgba(56, 189, 248, 0.4)' : 'none',
                transition: 'all 0.15s ease'
              }}
            >
              {React.cloneElement(tab.icon as React.ReactElement<any>, {
                color: isActive ? '#38bdf8' : '#64748b'
              })}
              {tab.label}
            </button>
          );
        })}
      </nav>
    </header>
  );
};
