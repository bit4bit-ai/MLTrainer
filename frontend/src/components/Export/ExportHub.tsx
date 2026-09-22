import React, { useState, useEffect } from 'react';
import { VisionTask, ExportBundleResult } from '../../types';
import { Download, Copy, Check, Terminal, FileCode, Server, ShieldCheck, Sparkles } from 'lucide-react';
import { fetchInferenceCode, createExportBundle } from '../../services/api';

interface ExportHubProps {
  task: VisionTask;
}

export const ExportHub: React.FC<ExportHubProps> = ({ task }) => {
  const [format, setFormat] = useState<string>('onnx');
  const [quantization, setQuantization] = useState<string>('fp32');
  const [codeSnippets, setCodeSnippets] = useState<{ inference_code: string; fastapi_snippet: string } | null>(null);
  const [activeCodeTab, setActiveCodeTab] = useState<'inference' | 'fastapi'>('inference');
  const [copied, setCopied] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exportResult, setExportResult] = useState<ExportBundleResult | null>(null);

  useEffect(() => {
    loadCode();
  }, [task, format]);

  const loadCode = async () => {
    try {
      const res = await fetchInferenceCode(task, format);
      setCodeSnippets(res);
    } catch (e) {
      console.error(e);
    }
  };

  const handleCopy = () => {
    const textToCopy = activeCodeTab === 'inference'
      ? codeSnippets?.inference_code
      : codeSnippets?.fastapi_snippet;
    if (textToCopy) {
      navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleExportBundle = async () => {
    setExporting(true);
    try {
      const res = await createExportBundle(task, format, quantization);
      setExportResult(res);
      // Auto-trigger download
      const link = document.createElement('a');
      link.href = res.download_url;
      link.download = res.bundle_name;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (e) {
      console.error(e);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div style={{ padding: '0 20px 24px', maxWidth: '1180px', margin: '0 auto' }}>
      
      {/* Top Banner */}
      <div className="glass-panel" style={{ padding: '24px', marginBottom: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Download size={22} color="#06b6d4" />
              Model Export & Production Deployment Hub
            </h2>
            <p style={{ fontSize: '0.825rem', color: '#94a3b8' }}>
              Export serialized weights, quantized runtimes, inference scripts, and container deployment packages
            </p>
          </div>

          <button
            onClick={handleExportBundle}
            disabled={exporting}
            className="btn-primary"
            style={{ padding: '10px 24px', fontSize: '0.95rem' }}
          >
            <Download size={18} />
            {exporting ? 'Packaging ZIP Archive...' : 'Download Export Bundle (.ZIP)'}
          </button>
        </div>

        {exportResult && (
          <div style={{ marginTop: '16px', padding: '12px 16px', background: 'rgba(16, 185, 129, 0.15)', border: '1px solid rgba(16, 185, 129, 0.3)', borderRadius: '8px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.85rem', color: '#6ee7b7' }}>
              ✓ Export bundle ready: <strong>{exportResult.bundle_name}</strong> ({exportResult.file_size_kb} KB)
            </span>
            <a href={exportResult.download_url} download className="btn-secondary" style={{ padding: '4px 12px', fontSize: '0.75rem' }}>
              Re-download
            </a>
          </div>
        )}
      </div>

      {/* Grid: Export Formats & Options (Left) and Generated Code (Right) */}
      <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: '24px' }}>
        
        {/* Left: Configuration */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Format Selection Card */}
          <div className="glass-panel" style={{ padding: '20px' }}>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#f8fafc', marginBottom: '14px' }}>
              Target Serialized Format
            </h3>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {[
                { id: 'onnx', name: 'ONNX (.onnx)', desc: 'Standard cross-platform format for TensorRT, OpenVINO, and ONNX Runtime' },
                { id: 'torchscript', name: 'TorchScript (.pt)', desc: 'Optimized JIT compiled graph for high-performance C++ and Python deployment' },
                { id: 'pytorch', name: 'PyTorch State Dict (.pth)', desc: 'Raw model weights for continued fine-tuning in PyTorch' }
              ].map((f) => (
                <div
                  key={f.id}
                  onClick={() => setFormat(f.id)}
                  style={{
                    padding: '12px 14px',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    background: format === f.id ? 'rgba(6, 182, 212, 0.15)' : 'rgba(15, 23, 42, 0.5)',
                    border: format === f.id ? '1px solid #06b6d4' : '1px solid #1e293b',
                    transition: 'all 0.15s ease'
                  }}
                >
                  <div style={{ fontWeight: 600, fontSize: '0.875rem', color: format === f.id ? '#38bdf8' : '#f8fafc' }}>
                    {f.name}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '2px' }}>
                    {f.desc}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Quantization Card */}
          <div className="glass-panel" style={{ padding: '20px' }}>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#f8fafc', marginBottom: '14px' }}>
              Precision & Quantization
            </h3>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {[
                { id: 'fp32', name: 'FP32 (Standard Single Precision)', desc: 'Highest accuracy, default deployment precision' },
                { id: 'fp16', name: 'FP16 (Half Precision)', desc: '2x faster throughput on modern NVIDIA GPUs' },
                { id: 'int8', name: 'INT8 (Dynamic Integer Quantization)', desc: '4x smaller file size, optimized for CPU & edge inference' }
              ].map((q) => (
                <div
                  key={q.id}
                  onClick={() => setQuantization(q.id)}
                  style={{
                    padding: '12px 14px',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    background: quantization === q.id ? 'rgba(16, 185, 129, 0.15)' : 'rgba(15, 23, 42, 0.5)',
                    border: quantization === q.id ? '1px solid #10b981' : '1px solid #1e293b',
                    transition: 'all 0.15s ease'
                  }}
                >
                  <div style={{ fontWeight: 600, fontSize: '0.875rem', color: quantization === q.id ? '#34d399' : '#f8fafc' }}>
                    {q.name}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '2px' }}>
                    {q.desc}
                  </div>
                </div>
              ))}
            </div>
          </div>

        </div>

        {/* Right: Code Generator */}
        <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column' }}>
          
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            {/* Tabs */}
            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                onClick={() => setActiveCodeTab('inference')}
                className="btn-secondary"
                style={{
                  background: activeCodeTab === 'inference' ? '#1e293b' : 'transparent',
                  color: activeCodeTab === 'inference' ? '#38bdf8' : '#94a3b8',
                  borderColor: activeCodeTab === 'inference' ? '#38bdf8' : 'transparent'
                }}
              >
                <Terminal size={15} />
                inference.py (CLI Runner)
              </button>
              <button
                onClick={() => setActiveCodeTab('fastapi')}
                className="btn-secondary"
                style={{
                  background: activeCodeTab === 'fastapi' ? '#1e293b' : 'transparent',
                  color: activeCodeTab === 'fastapi' ? '#38bdf8' : '#94a3b8',
                  borderColor: activeCodeTab === 'fastapi' ? '#38bdf8' : 'transparent'
                }}
              >
                <Server size={15} />
                serve.py (FastAPI Microservice)
              </button>
            </div>

            {/* Copy Button */}
            <button onClick={handleCopy} className="btn-secondary" style={{ padding: '6px 14px' }}>
              {copied ? <Check size={16} color="#10b981" /> : <Copy size={16} />}
              {copied ? 'Copied!' : 'Copy Code'}
            </button>
          </div>

          {/* Code Viewer Block */}
          <div style={{
            background: '#040711',
            borderRadius: '8px',
            border: '1px solid #1e293b',
            padding: '16px',
            flex: 1,
            overflowX: 'auto',
            maxHeight: '520px',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.825rem',
            color: '#e2e8f0',
            whiteSpace: 'pre',
            lineHeight: 1.6
          }}>
            {activeCodeTab === 'inference'
              ? codeSnippets?.inference_code
              : codeSnippets?.fastapi_snippet}
          </div>

        </div>

      </div>

    </div>
  );
};
