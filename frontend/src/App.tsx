import React, { useState, useEffect } from 'react';
import { VisionTask, ActiveTab, Dataset, SplitDistribution, TrainingStatus } from './types';
import { Navbar } from './components/Navbar';
import { ImageGallery } from './components/Gallery/ImageGallery';
import { SplitEditor } from './components/Splitting/SplitEditor';
import { TrainingStudio } from './components/Training/TrainingStudio';
import { EvaluationDashboard } from './components/Evaluation/EvaluationDashboard';
import { ExportHub } from './components/Export/ExportHub';
import {
  fetchDataset,
  fetchSplitDistribution,
  fetchTrainingStatus,
  updateItemAnnotation,
  deleteItem,
  createTrainingWebSocket
} from './services/api';

export const App: React.FC = () => {
  const [currentTask, setCurrentTask] = useState<VisionTask>(() => {
    return (localStorage.getItem('mltrainer_current_task') as VisionTask) || 'classification';
  });
  const [activeTab, setActiveTab] = useState<ActiveTab>(() => {
    return (localStorage.getItem('mltrainer_active_tab') as ActiveTab) || 'gallery';
  });
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [distribution, setDistribution] = useState<SplitDistribution | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({
    state: 'idle',
    task: 'classification',
    current_epoch: 0,
    total_epochs: 15,
    best_metric: 0.0,
    history: [],
    device: 'cpu'
  });
  const [loading, setLoading] = useState(true);

  // Load Task Data
  const loadData = async (task: VisionTask) => {
    try {
      const [ds, dist, status] = await Promise.all([
        fetchDataset(task),
        fetchSplitDistribution(task),
        fetchTrainingStatus()
      ]);
      setDataset(ds);
      setDistribution(dist);
      setTrainingStatus(status);
    } catch (e) {
      console.error('Error loading task data:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData(currentTask);
  }, [currentTask]);

  // WebSocket Live Telemetry Connection
  useEffect(() => {
    let ws: WebSocket;
    const connect = () => {
      ws = createTrainingWebSocket((msg) => {
        if (msg.type === 'initial_status') {
          setTrainingStatus(msg.data);
        } else if (msg.type === 'step_progress') {
          setTrainingStatus((prev) => ({
            ...prev,
            current_epoch: msg.epoch,
            total_epochs: msg.total_epochs,
            state: 'running'
          }));
        } else if (msg.type === 'epoch_complete') {
          setTrainingStatus((prev) => ({
            ...prev,
            best_metric: msg.best_metric,
            history: [...prev.history, msg.record]
          }));
        } else if (msg.type === 'training_finished') {
          setTrainingStatus((prev) => ({
            ...prev,
            state: msg.state,
            best_metric: msg.best_metric
          }));
        }
      });
    };

    connect();
    return () => {
      if (ws) ws.close();
    };
  }, []);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Navbar
        currentTask={currentTask}
        onSelectTask={(task) => {
          setCurrentTask(task);
          localStorage.setItem('mltrainer_current_task', task);
        }}
        activeTab={activeTab}
        onSelectTab={(tab) => {
          setActiveTab(tab);
          localStorage.setItem('mltrainer_active_tab', tab);
        }}
        trainingState={trainingStatus.state}
        device={trainingStatus.device}
      />

      <main style={{ flex: 1 }}>
        {loading || !dataset ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '50vh', color: '#64748b' }}>
            Loading {currentTask} datasets & vision modules...
          </div>
        ) : (
          <>
            {activeTab === 'gallery' && (
              <ImageGallery
                dataset={dataset}
                task={currentTask}
                onRefresh={() => loadData(currentTask)}
                onUpdateAnnotation={(id, updates) => updateItemAnnotation(currentTask, id, updates)}
                onDeleteItem={(id) => deleteItem(currentTask, id)}
              />
            )}

            {activeTab === 'splitting' && (
              <SplitEditor
                task={currentTask}
                distribution={distribution}
                onSplitCompleted={() => loadData(currentTask)}
              />
            )}

            {activeTab === 'training' && (
              <TrainingStudio
                task={currentTask}
                status={trainingStatus}
                onRefreshStatus={async () => {
                  const s = await fetchTrainingStatus();
                  setTrainingStatus(s);
                }}
              />
            )}

            {activeTab === 'evaluation' && (
              <EvaluationDashboard task={currentTask} />
            )}

            {activeTab === 'export' && (
              <ExportHub task={currentTask} />
            )}
          </>
        )}
      </main>
    </div>
  );
};

export default App;
