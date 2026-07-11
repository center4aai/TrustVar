import { create } from 'zustand';

const VALID_SECTIONS = ['dashboard', 'datasets', 'models', 'tasks', 'results'];

interface SelectedItem {
  type: 'model' | 'task' | 'dataset';
  id: string;
  name: string;
}

interface AppState {
  activeSection: string | null;
  setActiveSection: (section: string | null) => void;
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  selectedItem: SelectedItem | null;
  setSelectedItem: (item: SelectedItem | null) => void;
  selectedTaskId: string | null;
  setSelectedTaskId: (taskId: string | null) => void;
  preselectedDatasetId: string | null;
  setPreselectedDatasetId: (datasetId: string | null) => void;
}

export const useAppStore = create<AppState>((set) => ({
  activeSection: null,
  setActiveSection: (section) => {
    if (section === null || section === '' || section === undefined) {
      set({ activeSection: null });
      return;
    }
    if (typeof section !== 'string') {
      console.warn('Invalid section type:', section);
      set({ activeSection: null });
      return;
    }
    if (!VALID_SECTIONS.includes(section)) {
      console.warn('Invalid section:', section);
      set({ activeSection: null });
      return;
    }
    set({ activeSection: section });
  },
  searchQuery: '',
  setSearchQuery: (query) => set({ searchQuery: query }),
  selectedItem: null,
  setSelectedItem: (item) => set({ selectedItem: item }),
  selectedTaskId: null,
  setSelectedTaskId: (taskId) => set({ selectedTaskId: taskId }),
  preselectedDatasetId: null,
  setPreselectedDatasetId: (datasetId) => set({ preselectedDatasetId: datasetId }),
}));
