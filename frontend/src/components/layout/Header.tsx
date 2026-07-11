import React, { useState, useMemo } from 'react';
import { Search, Bell, User, Database, Cpu, Zap, X } from 'lucide-react';
import { useAppStore } from '@/stores/useAppStore';
import { useDatasets, useModels, useTasks } from '@/api/hooks';

type SearchResult = {
  type: 'dataset' | 'model' | 'task';
  id: string;
  name: string;
};

export default function Header() {
  const { searchQuery, setSearchQuery, setActiveSection, setSelectedTaskId, setSelectedItem } = useAppStore();
  const { data: datasets = [] } = useDatasets();
  const { data: models = [] } = useModels();
  const { data: tasks = [] } = useTasks();
  const [showDropdown, setShowDropdown] = useState(false);

  const searchResults = useMemo<SearchResult[]>(() => {
    if (!searchQuery.trim()) return [];

    const query = searchQuery.toLowerCase();
    const results: SearchResult[] = [];

    datasets.forEach(ds => {
      if (ds.name.toLowerCase().includes(query) || ds.description?.toLowerCase().includes(query)) {
        results.push({ type: 'dataset', id: ds.id, name: ds.name });
      }
    });

    models.forEach(m => {
      if (m.name.toLowerCase().includes(query) || m.model_name.toLowerCase().includes(query)) {
        results.push({ type: 'model', id: m.id, name: m.name });
      }
    });

    tasks.forEach(t => {
      if (t.name.toLowerCase().includes(query)) {
        results.push({ type: 'task', id: t.id, name: t.name });
      }
    });

    return results.slice(0, 10);
  }, [searchQuery, datasets, models, tasks]);

  const handleSelect = (result: SearchResult) => {
    setSearchQuery('');
    setShowDropdown(false);

    switch (result.type) {
      case 'model':
        setSelectedItem({ type: 'model', id: result.id, name: result.name });
        setActiveSection('models');
        break;
      case 'task':
        setSelectedTaskId(result.id);
        setActiveSection('results');
        break;
      case 'dataset':
        setSelectedItem({ type: 'dataset', id: result.id, name: result.name });
        setActiveSection('datasets');
        break;
    }
  };

  const getIcon = (type: string) => {
    switch (type) {
      case 'dataset': return <Database size={14} className="text-violet-400" />;
      case 'model': return <Cpu size={14} className="text-cyan-400" />;
      case 'task': return <Zap size={14} className="text-amber-400" />;
      default: return null;
    }
  };

  return (
    <div className="h-16 bg-[#0A0A0A] border-b border-gray-800 flex items-center justify-between px-6">
      {/* Search */}
      <div className="flex-1 max-w-2xl relative">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={18} />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setShowDropdown(true);
            }}
            onFocus={() => setShowDropdown(true)}
            placeholder="Search datasets, models, tasks..."
            className="w-full bg-gray-900 border border-gray-800 rounded-lg pl-10 pr-4 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-violet-600/50 focus:ring-1 focus:ring-violet-600/50 transition-all"
          />
          {searchQuery && (
            <button
              onClick={() => {
                setSearchQuery('');
                setShowDropdown(false);
              }}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-white"
            >
              <X size={14} />
            </button>
          )}
        </div>

        {/* Search Results Dropdown */}
        {showDropdown && searchResults.length > 0 && (
          <div className="absolute top-full left-0 right-0 mt-2 bg-gray-900 border border-gray-800 rounded-lg shadow-xl z-50 max-h-80 overflow-y-auto">
            {searchResults.map((result) => (
              <button
                key={`${result.type}-${result.id}`}
                onClick={() => handleSelect(result)}
                className="w-full px-4 py-3 flex items-center gap-3 hover:bg-gray-800 transition-colors text-left border-b border-gray-800 last:border-b-0"
              >
                {getIcon(result.type)}
                <div className="flex-1 min-w-0">
                  <div className="text-sm text-white truncate">{result.name}</div>
                  <div className="text-xs text-gray-500 capitalize">{result.type}</div>
                </div>
              </button>
            ))}
          </div>
        )}

        {showDropdown && searchQuery.trim() && searchResults.length === 0 && (
          <div className="absolute top-full left-0 right-0 mt-2 bg-gray-900 border border-gray-800 rounded-lg shadow-xl z-50 p-4">
            <div className="text-sm text-gray-500 text-center">No results found</div>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2 ml-4">
        <button className="p-2 text-gray-400 hover:text-white hover:bg-gray-900 rounded-lg transition-all">
          <Bell size={18} />
        </button>
        <button className="p-2 text-gray-400 hover:text-white hover:bg-gray-900 rounded-lg transition-all">
          <User size={18} />
        </button>
      </div>
    </div>
  );
}