import React, { useState, useRef } from 'react';
import { VisionTask } from '../../types';
import { FolderUp, Folder, HardDrive, CheckCircle2, AlertCircle, X, Image as ImageIcon } from 'lucide-react';
import { uploadBatchImages, importLocalFolder } from '../../services/api';

interface DatasetUploaderProps {
  task: VisionTask;
  classes: string[];
  isOpen: boolean;
  onClose: () => void;
  onUploaded: () => void;
}

export const DatasetUploader: React.FC<DatasetUploaderProps> = ({
  task,
  classes,
  isOpen,
  onClose,
  onUploaded
}) => {
  const [mode, setMode] = useState<'folder_picker' | 'local_path'>('folder_picker');
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [folderName, setFolderName] = useState<string>('');
  const [localPath, setLocalPath] = useState<string>('');
  const [split, setSplit] = useState<string>('train');
  const [label, setLabel] = useState<string>(classes[0] || 'Default');
  const [uploading, setUploading] = useState(false);
  const [progressText, setProgressText] = useState<string>('');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  
  const folderInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  // Handle browser folder selection
  const handleFolderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const allFiles = Array.from(e.target.files);
    
    // Filter valid image formats
    const validExts = ['.png', '.jpg', '.jpeg', '.webp', '.bmp'];
    const imageFiles = allFiles.filter(f => validExts.some(ext => f.name.toLowerCase().endsWith(ext)));

    // Derive folder name from webkitRelativePath
    const firstPath = allFiles[0].webkitRelativePath || '';
    const extractedFolderName = firstPath ? firstPath.split('/')[0] : 'Selected Folder';

    setFolderName(extractedFolderName);
    setSelectedFiles(imageFiles);
    setErrorMsg(null);
    setSuccessMsg(null);
  };

  const handleUploadFolder = async () => {
    if (mode === 'folder_picker') {
      if (selectedFiles.length === 0) {
        setErrorMsg('Please select a folder containing images.');
        return;
      }
      setUploading(true);
      setErrorMsg(null);
      setSuccessMsg(null);
      const CHUNK_SIZE = 50;
      let totalImported = 0;
      const totalBatches = Math.ceil(selectedFiles.length / CHUNK_SIZE);

      try {
        for (let i = 0; i < selectedFiles.length; i += CHUNK_SIZE) {
          const chunk = selectedFiles.slice(i, i + CHUNK_SIZE);
          const currentBatch = Math.floor(i / CHUNK_SIZE) + 1;
          setProgressText(`Uploading batch ${currentBatch} of ${totalBatches} (${totalImported} / ${selectedFiles.length} images)...`);
          
          const res = await uploadBatchImages(task, chunk, split, label);
          totalImported += res.count || chunk.length;
        }

        setSuccessMsg(`Successfully imported ${totalImported} images from folder "${folderName}"!`);
        setSelectedFiles([]);
        setFolderName('');
        setTimeout(() => {
          onUploaded();
          onClose();
        }, 1200);
      } catch (err: any) {
        setErrorMsg(err.message || 'Folder upload failed');
      } finally {
        setUploading(false);
        setProgressText('');
      }
    } else {
      // Local path mode
      if (!localPath.trim()) {
        setErrorMsg('Please enter a local directory path.');
        return;
      }
      setUploading(true);
      setErrorMsg(null);
      setSuccessMsg(null);
      setProgressText('Scanning and importing local directory...');

      try {
        const res = await importLocalFolder(task, localPath.trim(), split, label);
        setSuccessMsg(`Successfully imported ${res.count} images from "${localPath}"!`);
        setLocalPath('');
        setTimeout(() => {
          onUploaded();
          onClose();
        }, 1200);
      } catch (err: any) {
        setErrorMsg(err.message || 'Local folder import failed');
      } finally {
        setUploading(false);
        setProgressText('');
      }
    }
  };

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(0,0,0,0.85)',
      backdropFilter: 'blur(8px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1100,
      padding: '20px'
    }}>
      <div className="glass-panel" style={{
        maxWidth: '580px',
        width: '100%',
        padding: '24px',
        border: '1px solid rgba(255,255,255,0.12)',
        boxShadow: '0 20px 40px rgba(0,0,0,0.6)'
      }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <div>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <FolderUp size={22} color="#06b6d4" />
              Import Image Folder to {task.toUpperCase()}
            </h3>
            <p style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
              Select a local folder on your computer to import all contained images
            </p>
          </div>
          <button onClick={onClose} style={{ background: 'transparent', border: 'none', color: '#94a3b8', cursor: 'pointer' }}>
            <X size={20} />
          </button>
        </div>

        {/* Mode Selector Tabs */}
        <div style={{
          display: 'flex',
          background: 'rgba(15, 23, 42, 0.8)',
          borderRadius: '8px',
          padding: '4px',
          marginBottom: '20px',
          border: '1px solid #1e293b'
        }}>
          <button
            onClick={() => setMode('folder_picker')}
            style={{
              flex: 1,
              padding: '8px',
              borderRadius: '6px',
              fontSize: '0.825rem',
              fontWeight: 600,
              border: 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '6px',
              background: mode === 'folder_picker' ? 'rgba(6, 182, 212, 0.2)' : 'transparent',
              color: mode === 'folder_picker' ? '#38bdf8' : '#94a3b8'
            }}
          >
            <Folder size={16} />
            Browse & Select Folder
          </button>
          <button
            onClick={() => setMode('local_path')}
            style={{
              flex: 1,
              padding: '8px',
              borderRadius: '6px',
              fontSize: '0.825rem',
              fontWeight: 600,
              border: 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '6px',
              background: mode === 'local_path' ? 'rgba(6, 182, 212, 0.2)' : 'transparent',
              color: mode === 'local_path' ? '#38bdf8' : '#94a3b8'
            }}
          >
            <HardDrive size={16} />
            Direct Folder Path
          </button>
        </div>

        {/* Mode 1: Browser Folder Picker */}
        {mode === 'folder_picker' && (
          <div>
            <input
              ref={folderInputRef}
              type="file"
              // webkitdirectory and directory enable the OS folder selector
              // @ts-ignore
              webkitdirectory=""
              // @ts-ignore
              directory=""
              multiple
              style={{ display: 'none' }}
              onChange={handleFolderChange}
            />

            <div
              onClick={() => folderInputRef.current?.click()}
              style={{
                border: selectedFiles.length > 0 ? '2px solid #06b6d4' : '2px dashed #334155',
                borderRadius: '12px',
                padding: '36px 20px',
                textAlign: 'center',
                cursor: 'pointer',
                background: selectedFiles.length > 0 ? 'rgba(6, 182, 212, 0.08)' : 'rgba(15, 23, 42, 0.5)',
                marginBottom: '20px',
                transition: 'all 0.2s ease'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '12px' }}>
                <Folder size={46} color={selectedFiles.length > 0 ? '#38bdf8' : '#06b6d4'} />
              </div>

              {selectedFiles.length > 0 ? (
                <div>
                  <div style={{ fontWeight: 700, color: '#f8fafc', fontSize: '1rem', marginBottom: '4px' }}>
                    📁 Folder: {folderName}
                  </div>
                  <div style={{ fontSize: '0.85rem', color: '#34d399', fontWeight: 600 }}>
                    ✓ Found {selectedFiles.length} image files ready to import
                  </div>
                  {selectedFiles.length > 500 && (
                    <div style={{ fontSize: '0.75rem', color: '#38bdf8', marginTop: '6px', background: 'rgba(6, 182, 212, 0.1)', padding: '6px 10px', borderRadius: '6px' }}>
                      💡 Tip: For large folders (e.g. 10,000 - 50,000 images), the <strong>"Direct Folder Path"</strong> tab above imports them in 2 seconds directly from disk!
                    </div>
                  )}
                  <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '6px' }}>
                    Click again to choose a different folder
                  </div>
                </div>
              ) : (
                <div>
                  <div style={{ fontSize: '0.95rem', fontWeight: 600, color: '#f8fafc', marginBottom: '6px' }}>
                    Click to Open Folder Selector Dialog
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                    Opens the Windows folder dialog to choose an entire directory of images
                  </div>
                  <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '8px' }}>
                    Supported formats inside folder: PNG, JPG, JPEG, WEBP, BMP
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Mode 2: Local Path on Disk */}
        {mode === 'local_path' && (
          <div style={{ marginBottom: '20px' }}>
            <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', marginBottom: '6px', textTransform: 'uppercase' }}>
              LOCAL DIRECTORY PATH ON DISK
            </label>
            <input
              type="text"
              placeholder="e.g. C:\Users\YourName\Pictures\DefectDataset"
              value={localPath}
              onChange={(e) => setLocalPath(e.target.value)}
              style={{ width: '100%', marginBottom: '8px' }}
            />
            <div style={{ fontSize: '0.72rem', color: '#64748b' }}>
              Enter any absolute directory path on your computer. MLTrainer will directly scan and import all images found inside.
            </div>
          </div>
        )}

        {/* Partition & Label Options */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
          <div>
            <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', marginBottom: '6px', textTransform: 'uppercase' }}>
              ASSIGN TO PARTITION
            </label>
            <select value={split} onChange={(e) => setSplit(e.target.value)} style={{ width: '100%' }}>
              <option value="train">Train Partition</option>
              <option value="val">Validation Partition</option>
              <option value="test">Test Partition</option>
            </select>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', marginBottom: '6px', textTransform: 'uppercase' }}>
              INITIAL CLASS LABEL
            </label>
            <select value={label} onChange={(e) => setLabel(e.target.value)} style={{ width: '100%' }}>
              {classes.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Feedback Messages */}
        {progressText && (
          <div style={{ padding: '10px 14px', background: 'rgba(6,182,212,0.15)', color: '#38bdf8', borderRadius: '8px', fontSize: '0.825rem', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span className="live-pulse" />
            <span>{progressText}</span>
          </div>
        )}

        {errorMsg && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 14px', background: 'rgba(244,63,94,0.15)', color: '#fda4af', borderRadius: '8px', fontSize: '0.825rem', marginBottom: '16px' }}>
            <AlertCircle size={16} />
            <span>{errorMsg}</span>
          </div>
        )}
        
        {successMsg && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 14px', background: 'rgba(16,185,129,0.15)', color: '#6ee7b7', borderRadius: '8px', fontSize: '0.825rem', marginBottom: '16px' }}>
            <CheckCircle2 size={16} />
            <span>{successMsg}</span>
          </div>
        )}

        {/* Action Buttons */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
          <button onClick={onClose} className="btn-secondary">Cancel</button>
          <button
            onClick={handleUploadFolder}
            disabled={uploading || (mode === 'folder_picker' && selectedFiles.length === 0) || (mode === 'local_path' && !localPath.trim())}
            className="btn-primary"
            style={{ padding: '10px 20px' }}
          >
            {uploading ? 'Importing Folder...' : 'Import Folder'}
          </button>
        </div>

      </div>
    </div>
  );
};
