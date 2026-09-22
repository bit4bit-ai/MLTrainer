import React, { useState, useEffect } from 'react';
import { Dataset, DatasetItem, VisionTask } from '../../types';
import { Search, Filter, Plus, Box, Trash2, CheckSquare, Square, Check, RefreshCw, FolderUp, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react';
import { ImageModal } from './ImageModal';
import { DatasetUploader } from './DatasetUploader';
import { batchDeleteItems } from '../../services/api';

interface ImageGalleryProps {
  dataset: Dataset;
  task: VisionTask;
  onRefresh: () => Promise<void>;
  onUpdateAnnotation: (itemId: string, updates: any) => Promise<void>;
  onDeleteItem: (itemId: string) => Promise<void>;
}

export const ImageGallery: React.FC<ImageGalleryProps> = ({
  dataset,
  task,
  onRefresh,
  onUpdateAnnotation,
  onDeleteItem
}) => {
  const [selectedSplit, setSelectedSplit] = useState<string>('all');
  const [selectedClass, setSelectedClass] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [activeItem, setActiveItem] = useState<DatasetItem | null>(null);
  const [isUploaderOpen, setIsUploaderOpen] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [isDeleting, setIsDeleting] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Pagination state for high-volume datasets (e.g. 50,000 images)
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [pageSize, setPageSize] = useState<number>(60);

  // Filter items
  const filteredItems = dataset.items.filter((item) => {
    if (selectedSplit !== 'all' && item.split !== selectedSplit) return false;
    if (selectedClass !== 'all') {
      if (task === 'classification' && item.label !== selectedClass) return false;
      if (task === 'anomaly') {
        const isAno = selectedClass === 'Anomaly';
        if (item.is_anomaly !== isAno) return false;
      }
      if (task === 'detection') {
        const hasClass = item.boxes?.some(b => b.label === selectedClass);
        if (!hasClass) return false;
      }
    }
    if (searchQuery && !item.filename.toLowerCase().includes(searchQuery.toLowerCase())) {
      return false;
    }
    return true;
  });

  // Reset page when filter criteria change
  useEffect(() => {
    setCurrentPage(1);
  }, [selectedSplit, selectedClass, searchQuery, task]);

  const totalPages = Math.max(1, Math.ceil(filteredItems.length / pageSize));
  const validCurrentPage = Math.min(currentPage, totalPages);
  const startIndex = (validCurrentPage - 1) * pageSize;
  const paginatedItems = filteredItems.slice(startIndex, startIndex + pageSize);

  const allFilteredSelected = filteredItems.length > 0 && filteredItems.every(i => selectedIds.has(i.id));

  const handleToggleSelectAll = () => {
    if (allFilteredSelected) {
      // Deselect all filtered items
      const next = new Set(selectedIds);
      filteredItems.forEach(i => next.delete(i.id));
      setSelectedIds(next);
    } else {
      // Select all filtered items
      const next = new Set(selectedIds);
      filteredItems.forEach(i => next.add(i.id));
      setSelectedIds(next);
    }
  };

  const handleToggleItem = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const next = new Set(selectedIds);
    if (next.has(id)) {
      next.delete(id);
    } else {
      next.add(id);
    }
    setSelectedIds(next);
  };

  const handleBatchDelete = async () => {
    if (selectedIds.size === 0) return;
    const count = selectedIds.size;
    if (confirm(`Are you sure you want to delete ${count} selected image${count > 1 ? 's' : ''} from the dataset?`)) {
      setIsDeleting(true);
      try {
        await batchDeleteItems(task, Array.from(selectedIds));
        setSelectedIds(new Set());
        await onRefresh();
      } catch (err) {
        console.error('Batch delete error:', err);
        alert('Failed to delete selected items.');
      } finally {
        setIsDeleting(false);
      }
    }
  };

  return (
    <div style={{ padding: '0 20px 24px' }}>
      
      {/* Top Action Bar */}
      <div className="glass-panel" style={{ padding: '16px 20px', marginBottom: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
          
          {/* Search & Filters */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
            
            {/* Search */}
            <div style={{ position: 'relative', width: '220px' }}>
              <Search size={16} color="#64748b" style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)' }} />
              <input
                type="text"
                placeholder="Search images..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                style={{ paddingLeft: '34px', width: '100%' }}
              />
            </div>

            {/* Split Filter */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Filter size={15} color="#94a3b8" />
              <select value={selectedSplit} onChange={(e) => setSelectedSplit(e.target.value)}>
                <option value="all">All Partitions ({dataset.items.length})</option>
                <option value="train">Train Set</option>
                <option value="val">Validation Set</option>
                <option value="test">Test Set</option>
              </select>
            </div>

            {/* Class Filter */}
            <div>
              <select value={selectedClass} onChange={(e) => setSelectedClass(e.target.value)}>
                <option value="all">All Classes / Tags</option>
                {dataset.classes.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>

          </div>

          {/* Right Action Buttons */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
            
            {/* Select All / Deselect All Button */}
            <button
              onClick={handleToggleSelectAll}
              className="btn-secondary"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                background: allFilteredSelected ? 'rgba(6, 182, 212, 0.2)' : 'rgba(30, 41, 59, 0.8)',
                borderColor: allFilteredSelected ? '#06b6d4' : 'rgba(255, 255, 255, 0.1)',
                color: allFilteredSelected ? '#38bdf8' : '#f8fafc'
              }}
            >
              {allFilteredSelected ? <CheckSquare size={16} color="#06b6d4" /> : <Square size={16} />}
              {allFilteredSelected ? 'Deselect All' : `Select All (${filteredItems.length})`}
            </button>

            {/* Delete Selected Button */}
            {selectedIds.size > 0 && (
              <button
                onClick={handleBatchDelete}
                disabled={isDeleting}
                className="btn-danger"
                style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
              >
                <Trash2 size={16} />
                {isDeleting ? 'Deleting...' : `Delete Selected (${selectedIds.size})`}
              </button>
            )}

            {/* Refresh Button */}
            <button
              onClick={async () => {
                setIsRefreshing(true);
                try {
                  await onRefresh();
                } finally {
                  setTimeout(() => setIsRefreshing(false), 400);
                }
              }}
              disabled={isRefreshing}
              className="btn-secondary"
              title="Reload images from server"
              style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              <RefreshCw size={15} className={isRefreshing ? 'animate-spin' : ''} />
              {isRefreshing ? 'Reloading...' : 'Reload'}
            </button>

            <button onClick={() => setIsUploaderOpen(true)} className="btn-primary">
              <FolderUp size={16} />
              Import Folder
            </button>
          </div>

        </div>

        {/* Selected count info banner */}
        {selectedIds.size > 0 && (
          <div style={{
            marginTop: '12px',
            paddingTop: '10px',
            borderTop: '1px solid rgba(255, 255, 255, 0.08)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            fontSize: '0.825rem'
          }}>
            <span style={{ color: '#38bdf8' }}>
              <strong>{selectedIds.size}</strong> image{selectedIds.size > 1 ? 's' : ''} selected
            </span>
            <button
              onClick={() => setSelectedIds(new Set())}
              style={{ background: 'transparent', border: 'none', color: '#94a3b8', cursor: 'pointer', textDecoration: 'underline', fontSize: '0.75rem' }}
            >
              Clear selection
            </button>
          </div>
        )}
      </div>

      {/* Dataset Image Grid */}
      {filteredItems.length === 0 ? (
        <div className="glass-panel" style={{ padding: '60px 20px', textAlign: 'center' }}>
          <div style={{ color: '#64748b', marginBottom: '8px' }}>No images match your current filter criteria.</div>
          <button onClick={() => { setSelectedSplit('all'); setSelectedClass('all'); setSearchQuery(''); }} className="btn-secondary">
            Reset Filters
          </button>
        </div>
      ) : (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
          gap: '16px'
        }}>
          {paginatedItems.map((item) => {
            const isSelected = selectedIds.has(item.id);

            return (
              <div
                key={item.id}
                className="glass-card"
                onClick={() => setActiveItem(item)}
                style={{
                  cursor: 'pointer',
                  borderRadius: '10px',
                  overflow: 'hidden',
                  position: 'relative',
                  display: 'flex',
                  flexDirection: 'column',
                  border: isSelected ? '2px solid #06b6d4' : undefined,
                  boxShadow: isSelected ? '0 0 16px rgba(6, 182, 212, 0.35)' : undefined
                }}
              >
                {/* Thumbnail Container */}
                <div style={{ position: 'relative', height: '170px', background: '#070b14', overflow: 'hidden' }}>
                  <img
                    src={item.url}
                    alt={item.filename}
                    loading="lazy"
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'cover',
                      transition: 'transform 0.3s ease'
                    }}
                    onMouseOver={(e) => (e.currentTarget.style.transform = 'scale(1.05)')}
                    onMouseOut={(e) => (e.currentTarget.style.transform = 'scale(1.0)')}
                  />

                  {/* Heatmap overlay for anomaly preview */}
                  {task === 'anomaly' && item.is_anomaly && item.heatmap_url && (
                    <img
                      src={item.heatmap_url}
                      alt="Anomaly Heatmap"
                      style={{
                        position: 'absolute',
                        inset: 0,
                        width: '100%',
                        height: '100%',
                        objectFit: 'cover',
                        opacity: 0.65,
                        pointerEvents: 'none'
                      }}
                    />
                  )}

                  {/* Multi-select checkbox overlay */}
                  <div
                    onClick={(e) => {
                      e.stopPropagation();
                      const next = new Set(selectedIds);
                      if (next.has(item.id)) {
                        next.delete(item.id);
                      } else {
                        next.add(item.id);
                      }
                      setSelectedIds(next);
                    }}
                    style={{
                      position: 'absolute',
                      top: '8px',
                      left: '8px',
                      width: '26px',
                      height: '26px',
                      borderRadius: '6px',
                      background: isSelected ? '#06b6d4' : 'rgba(15, 23, 42, 0.75)',
                      backdropFilter: 'blur(4px)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      border: isSelected ? '1px solid #38bdf8' : '1px solid rgba(255, 255, 255, 0.25)',
                      cursor: 'pointer',
                      zIndex: 10,
                      transition: 'all 0.15s ease'
                    }}
                    title={isSelected ? 'Deselect image' : 'Select image'}
                  >
                    {isSelected && <Check size={16} color="#ffffff" strokeWidth={3} />}
                  </div>

                  {/* Split tag badge */}
                  <div style={{
                    position: 'absolute',
                    top: '8px',
                    right: '8px',
                    background: 'rgba(15, 23, 42, 0.75)',
                    backdropFilter: 'blur(4px)',
                    padding: '2px 8px',
                    borderRadius: '4px',
                    fontSize: '0.65rem',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                    fontWeight: 600,
                    color: item.split === 'train' ? '#38bdf8' : item.split === 'val' ? '#a78bfa' : '#34d399',
                    border: `1px solid ${item.split === 'train' ? 'rgba(56, 189, 248, 0.3)' : item.split === 'val' ? 'rgba(167, 139, 250, 0.3)' : 'rgba(52, 211, 153, 0.3)'}`
                  }}>
                    {item.split}
                  </div>

                  {/* Task specific indicator overlay */}
                  {task === 'anomaly' && (
                    <div style={{
                      position: 'absolute',
                      bottom: '8px',
                      left: '8px',
                      background: item.is_anomaly ? 'rgba(239, 68, 68, 0.85)' : 'rgba(16, 185, 129, 0.85)',
                      padding: '2px 6px',
                      borderRadius: '4px',
                      fontSize: '0.65rem',
                      fontWeight: 600,
                      color: 'white'
                    }}>
                      {item.is_anomaly ? 'Anomaly' : 'Normal'}
                    </div>
                  )}

                  {task === 'detection' && item.boxes && item.boxes.length > 0 && (
                    <div style={{
                      position: 'absolute',
                      bottom: '8px',
                      left: '8px',
                      background: 'rgba(59, 130, 246, 0.85)',
                      padding: '2px 6px',
                      borderRadius: '4px',
                      fontSize: '0.65rem',
                      fontWeight: 600,
                      color: 'white',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px'
                    }}>
                      <Box size={10} />
                      {item.boxes.length} {item.boxes.length === 1 ? 'box' : 'boxes'}
                    </div>
                  )}
                </div>

                {/* Metadata details */}
                <div style={{ padding: '12px', background: 'rgba(15, 23, 42, 0.4)' }}>
                  <div style={{
                    fontSize: '0.82rem',
                    fontWeight: 500,
                    color: '#f8fafc',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis'
                  }} title={item.filename}>
                    {item.filename}
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '4px' }}>
                    {task === 'classification' && (
                      <span style={{ fontSize: '0.72rem', color: '#38bdf8', fontWeight: 500 }}>
                        🏷️ {item.label}
                      </span>
                    )}
                    {task === 'anomaly' && (
                      <span style={{ fontSize: '0.72rem', color: item.is_anomaly ? '#fb7185' : '#34d399' }}>
                        Score: {item.anomaly_score?.toFixed(2)}
                      </span>
                    )}
                    {task === 'detection' && (
                      <span style={{ fontSize: '0.72rem', color: '#94a3b8' }}>
                        {item.boxes?.map(b => b.label).filter((v, i, a) => a.indexOf(v) === i).join(', ')}
                      </span>
                    )}
                    {task === 'segmentation' && (
                      <span style={{ fontSize: '0.72rem', color: '#a78bfa' }}>
                        {item.segmentation?.length || 0} masks
                      </span>
                    )}
                    <span style={{ fontSize: '0.7rem', color: '#64748b' }}>
                      {item.width}x{item.height}
                    </span>
                  </div>
                </div>

              </div>
            );
          })}
        </div>
      )}

      {/* Pagination Controls Bar */}
      {filteredItems.length > 0 && (
        <div className="glass-panel" style={{
          marginTop: '20px',
          padding: '12px 20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '12px'
        }}>
          {/* Showing count */}
          <div style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
            Showing <strong style={{ color: '#f1f5f9' }}>{filteredItems.length === 0 ? 0 : startIndex + 1}</strong> - <strong style={{ color: '#f1f5f9' }}>{Math.min(startIndex + pageSize, filteredItems.length)}</strong> of <strong style={{ color: '#38bdf8' }}>{filteredItems.length.toLocaleString()}</strong> images
          </div>

          {/* Page controls */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <button
              onClick={() => setCurrentPage(1)}
              disabled={validCurrentPage <= 1}
              className="btn-secondary"
              style={{ padding: '6px 10px', opacity: validCurrentPage <= 1 ? 0.4 : 1, cursor: validCurrentPage <= 1 ? 'not-allowed' : 'pointer' }}
              title="First Page"
            >
              <ChevronsLeft size={16} />
            </button>
            <button
              onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
              disabled={validCurrentPage <= 1}
              className="btn-secondary"
              style={{ padding: '6px 10px', opacity: validCurrentPage <= 1 ? 0.4 : 1, cursor: validCurrentPage <= 1 ? 'not-allowed' : 'pointer' }}
              title="Previous Page"
            >
              <ChevronLeft size={16} />
            </button>

            <span style={{ fontSize: '0.85rem', color: '#cbd5e1', padding: '0 8px' }}>
              Page <strong style={{ color: '#f8fafc' }}>{validCurrentPage}</strong> of <strong style={{ color: '#f8fafc' }}>{totalPages}</strong>
            </span>

            <button
              onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
              disabled={validCurrentPage >= totalPages}
              className="btn-secondary"
              style={{ padding: '6px 10px', opacity: validCurrentPage >= totalPages ? 0.4 : 1, cursor: validCurrentPage >= totalPages ? 'not-allowed' : 'pointer' }}
              title="Next Page"
            >
              <ChevronRight size={16} />
            </button>
            <button
              onClick={() => setCurrentPage(totalPages)}
              disabled={validCurrentPage >= totalPages}
              className="btn-secondary"
              style={{ padding: '6px 10px', opacity: validCurrentPage >= totalPages ? 0.4 : 1, cursor: validCurrentPage >= totalPages ? 'not-allowed' : 'pointer' }}
              title="Last Page"
            >
              <ChevronsRight size={16} />
            </button>
          </div>

          {/* Items per page selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85rem', color: '#94a3b8' }}>
            <span>Per page:</span>
            <select
              value={pageSize}
              onChange={(e) => {
                setPageSize(Number(e.target.value));
                setCurrentPage(1);
              }}
              className="filter-select"
              style={{ padding: '4px 10px', fontSize: '0.85rem' }}
            >
              <option value={30}>30</option>
              <option value={60}>60</option>
              <option value={120}>120</option>
              <option value={240}>240</option>
            </select>
          </div>
        </div>
      )}

      {/* Inspection Modal */}
      {activeItem && (
        <ImageModal
          item={activeItem}
          task={task}
          classes={dataset.classes}
          onClose={() => setActiveItem(null)}
          onUpdate={async (id, updates) => {
            await onUpdateAnnotation(id, updates);
            await onRefresh();
          }}
          onDelete={async (id) => {
            await onDeleteItem(id);
            await onRefresh();
          }}
        />
      )}

      {/* Upload Modal */}
      <DatasetUploader
        task={task}
        classes={dataset.classes}
        isOpen={isUploaderOpen}
        onClose={() => setIsUploaderOpen(false)}
        onUploaded={onRefresh}
      />

    </div>
  );
};
