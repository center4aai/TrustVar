import React from 'react';
import { Database, Cpu, Zap, BarChart3, Settings, Activity } from 'lucide-react';
import { useAppStore } from '@/stores/useAppStore';
import { useDatasets, useModels, useTasks } from '@/api/hooks';

export default function Sidebar() {
  const { activeSection, setActiveSection } = useAppStore();
  const { data: datasets = [] } = useDatasets();
  const { data: models = [] } = useModels();
  const { data: tasks = [] } = useTasks();

  const navItems = [
    {
      id: 'dashboard',
      icon: Activity,
      label: 'Dashboard',
      badge: null
    },
    {
      id: 'datasets',
      icon: Database,
      label: 'Datasets',
      badge: datasets.length
    },
    {
      id: 'models',
      icon: Cpu,
      label: 'Models',
      badge: models.filter(m => m.status === 'registered').length
    },
    {
      id: 'tasks',
      icon: Zap,
      label: 'Tasks',
      badge: tasks.filter(t => t.status === 'running').length
    },
    {
      id: 'results',
      icon: BarChart3,
      label: 'Results',
      badge: null
    },
  ];

  return (
    <div className="w-64 bg-[#0A0A0A] border-r border-gray-800 flex flex-col">
      {/* Logo */}
      <div className="h-16 flex items-center px-6 border-b border-gray-800">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-gradient-to-br from-violet-500 to-purple-600 rounded-lg flex items-center justify-center shadow-lg shadow-violet-500/20">
            <span className="text-white font-bold text-sm">TV</span>
          </div>
          <span className="text-white font-semibold text-lg">TrustVar</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeSection === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveSection(item.id)}
              className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg transition-all ${
                isActive
                  ? 'bg-violet-600/10 text-violet-400 border border-violet-600/20'
                  : 'text-gray-400 hover:text-white hover:bg-gray-900'
              }`}
            >
              <div className="flex items-center gap-3">
                <Icon size={18} />
                <span className="text-sm font-medium">{item.label}</span>
              </div>
              {item.badge !== null && item.badge > 0 && (
                <span className={`px-2 py-0.5 ${
                  isActive ? 'bg-violet-600/20 text-violet-300' : 'bg-gray-800 text-gray-400'
                } text-xs rounded-full font-medium`}>
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Settings */}
      <div className="p-3 border-t border-gray-800">
        <button className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-gray-400 hover:text-white hover:bg-gray-900 transition-all">
          <Settings size={18} />
          <span className="text-sm font-medium">Settings</span>
        </button>
      </div>
    </div>
  );
}