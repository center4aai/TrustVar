import React, { useState } from 'react';
import { Database, Cpu, Zap, BarChart3, Settings, Home } from 'lucide-react';
import { useAppStore } from './stores/useAppStore';
import Sidebar from './components/layout/Sidebar';
import Header from './components/layout/Header';
import DashboardSection from './components/sections/DashboardSection';
import DatasetsSection from './components/sections/DatasetsSection';
import ModelsSection from './components/sections/ModelsSection';
import TasksSection from './components/sections/TasksSection';
import ResultsSection from './components/sections/ResultsSection';

const SECTIONS = {
  dashboard: DashboardSection,
  datasets: DatasetsSection,
  models: ModelsSection,
  tasks: TasksSection,
  results: ResultsSection,
};

export default function App() {
  const { activeSection } = useAppStore();

  const ActiveComponent = activeSection ? SECTIONS[activeSection as keyof typeof SECTIONS] : null;

  return (
    <div className="h-screen bg-black flex overflow-hidden">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <div className="flex-1 overflow-y-auto">
          {!activeSection ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center max-w-4xl mx-auto px-6">
                <div className="w-20 h-20 bg-gradient-to-br from-violet-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-2xl shadow-violet-500/20">
                  <span className="text-white font-bold text-3xl">TV</span>
                </div>
                <h1 className="text-4xl font-bold text-white mb-4">
                  Welcome to TrustVar
                </h1>
                <p className="text-lg text-gray-400 mb-12">
                  A framework for measuring the robustness of LLM benchmarks — and of the models that run on them
                </p>

                <div className="grid grid-cols-4 gap-4 mb-8">
                  {[
                    { icon: Database, title: 'Prepare Data', desc: 'Upload your test datasets' },
                    { icon: Cpu, title: 'Add Models', desc: 'Register models to evaluate' },
                    { icon: Zap, title: 'Run Tests', desc: 'Create evaluation tasks' },
                    { icon: BarChart3, title: 'Analyze', desc: 'Review results and metrics' },
                  ].map((step, i) => {
                    const Icon = step.icon;
                    return (
                      <div key={i} className="bg-gray-900/50 border border-gray-800 rounded-xl p-6 hover:border-gray-700 transition-all">
                        <div className="w-12 h-12 bg-violet-500/10 rounded-lg flex items-center justify-center mb-4 mx-auto">
                          <Icon className="text-violet-400" size={24} />
                        </div>
                        <h3 className="text-white font-semibold mb-2">{i + 1}. {step.title}</h3>
                        <p className="text-sm text-gray-500">{step.desc}</p>
                      </div>
                    );
                  })}
                </div>

                <p className="text-sm text-gray-500">
                  Select a section from the sidebar to get started
                </p>
              </div>
            </div>
          ) : ActiveComponent ? (
            <ActiveComponent />
          ) : (
            <div className="h-full flex items-center justify-center">
              <p className="text-gray-500">Section not found</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}