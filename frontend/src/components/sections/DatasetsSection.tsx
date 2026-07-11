import React, { useState } from 'react';
import { Upload, Eye, Trash2, Plus, X, Zap } from 'lucide-react';
import { useDatasets, useDeleteDataset, useDataset, useDatasetItems, useDatasetStats } from '@/api/hooks';
import { formatDate } from '@/utils/format';
import StatusBadge from '@/components/common/StatusBadge';
import DatasetUploadModal from './DatasetUploadModal';
import { useAppStore } from '@/stores/useAppStore';

export default function DatasetsSection() {
  const { data: datasets = [] } = useDatasets();
  const deleteDataset = useDeleteDataset();
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [activeTab, setActiveTab] = useState<'list' | 'details'>('list');
  const setActiveSection = useAppStore(s => s.setActiveSection);
  const setPreselectedDatasetId = useAppStore(s => s.setPreselectedDatasetId);

  const selectedDataset = useDataset(selectedDatasetId);
  const datasetItems = useDatasetItems(selectedDatasetId, 0, 20);
  const datasetStats = useDatasetStats(selectedDatasetId);
  const selectedItem = useAppStore(s => s.selectedItem);
  const clearSelectedItem = useAppStore(s => s.setSelectedItem);

  React.useEffect(() => {
    if (selectedItem?.type === 'dataset') {
      setSelectedDatasetId(selectedItem.id);
      setActiveTab('details');
      clearSelectedItem(null);
    }
  }, [selectedItem, clearSelectedItem]);

  const handleViewDetails = (id: string) => {
    setSelectedDatasetId(id);
    setActiveTab('details');
  };

  const handleBackToList = () => {
    setSelectedDatasetId(null);
    setActiveTab('list');
  };

  const handleCreateTask = (datasetId: string) => {
    setPreselectedDatasetId(datasetId);
    setActiveSection('tasks');
  };

  return (
    <div className="p-6 space-y-6 animate-slideIn">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-white mb-1">Datasets</h1>
          <p className="text-sm text-gray-400">Manage your evaluation datasets</p>
        </div>
        {activeTab === 'list' && (
          <button
            onClick={() => setShowUploadModal(true)}
            className="px-4 py-2 bg-violet-600 hover:bg-violet-700 rounded-lg text-sm text-white font-medium flex items-center gap-2 transition-colors"
          >
            <Upload size={16} />
            Upload Dataset
          </button>
        )}
        {activeTab !== 'list' && (
          <button
            onClick={handleBackToList}
            className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm text-white font-medium transition-colors"
          >
            ← Back to List
          </button>
        )}
      </div>

      {/* Upload Modal */}
      {showUploadModal && (
        <DatasetUploadModal
          onClose={() => setShowUploadModal(false)}
          onSuccess={() => {
            setShowUploadModal(false);
            setActiveTab('list');
          }}
        />
      )}

      {/* List View */}
      {activeTab === 'list' && (
        <div className="grid gap-4">
          {datasets.map(dataset => (
            <div key={dataset.id} className="bg-gray-900 border border-gray-800 rounded-xl p-6 hover:border-gray-700 transition-all group">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <h3 className="text-base font-semibold text-white mb-2">{dataset.name}</h3>
                  <p className="text-sm text-gray-400 mb-4">{dataset.description}</p>
                  <div className="flex items-center gap-6 text-xs text-gray-500">
                    <span>{dataset.size.toLocaleString()} items</span>
                    <span>{dataset.task_type}</span>
                    <span>{dataset.format.toUpperCase()}</span>
                    <span>{formatDate(dataset.created_at)}</span>
                  </div>
                </div>
                <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={() => handleViewDetails(dataset.id)}
                    className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-gray-400 hover:text-white transition-colors"
                    title="View details"
                  >
                    <Eye size={16} />
                  </button>
                  <button
                    onClick={() => {
                      if (confirm(`Delete dataset "${dataset.name}"?`)) {
                        deleteDataset.mutate(dataset.id);
                      }
                    }}
                    className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-gray-400 hover:text-red-400 transition-colors"
                    title="Delete dataset"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
            </div>
          ))}
          {datasets.length === 0 && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-12 text-center">
              <Upload size={48} className="mx-auto text-gray-700 mb-4" />
              <h3 className="text-lg font-medium text-white mb-2">No datasets yet</h3>
              <p className="text-sm text-gray-500 mb-6">Upload your first dataset to get started</p>
              <button
                onClick={() => setShowUploadModal(true)}
                className="px-4 py-2 bg-violet-600 hover:bg-violet-700 rounded-lg text-white text-sm font-medium transition-colors"
              >
                Upload Dataset
              </button>
            </div>
          )}
        </div>
      )}

      {/* Details View */}
      {activeTab === 'details' && selectedDataset.data && (
        <div className="space-y-6">
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <h2 className="text-xl font-semibold text-white mb-2">{selectedDataset.data.name}</h2>
                <p className="text-gray-400 mb-6">{selectedDataset.data.description || 'No description'}</p>
              </div>
              <button
                onClick={() => handleCreateTask(selectedDataset.data!.id)}
                className="px-4 py-2 bg-violet-600 hover:bg-violet-700 rounded-lg text-sm text-white font-medium flex items-center gap-2 transition-colors ml-4"
              >
                <Zap size={16} />
                Create Task
              </button>
            </div>

            {selectedDataset.data.tags && selectedDataset.data.tags.length > 0 && (
              <div className="mb-6">
                <div className="text-xs text-gray-500 mb-2">Tags</div>
                <div className="flex flex-wrap gap-2">
                  {selectedDataset.data.tags.map(tag => (
                    <span key={tag} className="px-2 py-1 bg-violet-500/10 border border-violet-500/20 rounded text-xs text-violet-400">
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div className="grid grid-cols-2 gap-4 mb-6">
              <div className="bg-black/40 rounded-lg p-4">
                <div className="text-xs text-gray-500 mb-1">Total Items</div>
                <div className="text-2xl font-semibold text-white">{datasetStats.data?.total_items || 0}</div>
              </div>
              <div className="bg-black/40 rounded-lg p-4">
                <div className="text-xs text-gray-500 mb-1">Created</div>
                <div className="text-sm text-white">{formatDate(selectedDataset.data.created_at)}</div>
              </div>
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Sample Items</h3>
            <div className="space-y-3">
              {datasetItems.data?.map((item, i) => (
                <details key={item.id} className="bg-black/40 border border-gray-800 rounded-lg">
                  <summary className="p-3 cursor-pointer hover:bg-gray-800/50 transition-colors">
                    <span className="text-sm text-white font-medium">Item #{i + 1}: </span>
                    <span className="text-sm text-gray-400">{item.prompt.slice(0, 70)}...</span>
                  </summary>
                  <div className="p-4 border-t border-gray-800 space-y-3">
                    <div>
                      <div className="text-xs text-gray-500 mb-1">Prompt:</div>
                      <div className="bg-black/50 p-2 rounded text-xs text-gray-300 font-mono">{item.prompt}</div>
                    </div>
                    {item.target && (
                      <div>
                        <div className="text-xs text-gray-500 mb-1">Target:</div>
                        <div className="bg-black/50 p-2 rounded text-xs text-gray-300 font-mono">{item.target}</div>
                      </div>
                    )}
                    {item.metadata && Object.keys(item.metadata).length > 0 && (
                      <div>
                        <div className="text-xs text-gray-500 mb-1">Metadata:</div>
                        <div className="bg-black/50 p-2 rounded text-xs text-gray-300 font-mono whitespace-pre-wrap">
                          {JSON.stringify(item.metadata, null, 2)}
                        </div>
                      </div>
                    )}
                  </div>
                </details>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}