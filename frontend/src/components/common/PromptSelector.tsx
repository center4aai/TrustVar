import React, { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { Plus, Trash2, ChevronDown, Pencil, X } from 'lucide-react';
import { usePrompts, useCreatePrompt, useDeletePrompt } from '@/api/hooks';
import type { Prompt } from '@/api/types';

interface PromptSelectorProps {
  selectedPromptId: string | null;
  onSelect: (prompt: Prompt | null) => void;
  onContentChange: (content: string) => void;
  label: string;
  error?: string;
  placeholder?: string;
}

export default function PromptSelector({
  selectedPromptId,
  onSelect,
  onContentChange,
  label,
  error,
  placeholder,
}: PromptSelectorProps) {
  const { data: prompts = [] } = usePrompts();
  const createPrompt = useCreatePrompt();
  const deletePrompt = useDeletePrompt();
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const [newPromptName, setNewPromptName] = useState('');
  const [newPromptContent, setNewPromptContent] = useState('');
  const [newPromptDescription, setNewPromptDescription] = useState('');
  const [expandedPromptId, setExpandedPromptId] = useState<string | null>(null);

  const selectedPrompt = prompts.find((p) => p.id === selectedPromptId);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const [dropdownPos, setDropdownPos] = useState<{ top: number; left: number; width: number } | null>(null);

  useEffect(() => {
    if (showDropdown && triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect();
      setDropdownPos({
        top: rect.bottom + window.scrollY,
        left: rect.left + window.scrollX,
        width: rect.width,
      });
    } else {
      setDropdownPos(null);
    }
  }, [showDropdown]);

  useEffect(() => {
    if (!showDropdown) return;
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as Node;
      if (
        (triggerRef.current && triggerRef.current.contains(target)) ||
        (dropdownRef.current && dropdownRef.current.contains(target))
      ) {
        return;
      }
      setShowDropdown(false);
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showDropdown]);

  const handleCreatePrompt = async () => {
    if (!newPromptName || !newPromptContent) return;

    try {
      const created = await createPrompt.mutateAsync({
        name: newPromptName,
        content: newPromptContent,
        prompt_type: 'judge',
        description: newPromptDescription || undefined,
      });
      onSelect(created);
      onContentChange(created.content);
      setNewPromptName('');
      setNewPromptContent('');
      setNewPromptDescription('');
      setShowCreateForm(false);
      setShowDropdown(false);
    } catch (err) {
      console.error('Failed to create prompt:', err);
    }
  };

  const handleSelectPrompt = (prompt: Prompt) => {
    onSelect(prompt);
    onContentChange(prompt.content);
    setShowDropdown(false);
  };

  const handleDeletePrompt = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm('Delete this prompt?')) {
      await deletePrompt.mutateAsync(id);
      if (selectedPromptId === id) {
        onSelect(null);
        onContentChange('');
      }
    }
  };

  return (
    <div className="space-y-2">
      <label className="block text-sm text-gray-400">{label}</label>

      {/* Selected prompt OR Dropdown */}
      {selectedPrompt ? (
        <div className={`bg-gray-800 border rounded-lg p-3 ${error ? 'border-red-500/50' : 'border-gray-700'}`}>
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm font-medium text-white">
              {selectedPrompt.name}
            </span>
            <button
              onClick={() => {
                onSelect(null);
                onContentChange('');
              }}
              className="text-xs text-gray-400 hover:text-white flex items-center gap-1"
            >
              <X size={12} />
              Clear
            </button>
          </div>
          {selectedPrompt.description && (
            <p className="text-xs text-gray-500">{selectedPrompt.description}</p>
          )}
        </div>
      ) : (
        <div>
          <button
            ref={triggerRef}
            onClick={() => setShowDropdown(!showDropdown)}
            className={`w-full p-2 bg-gray-800 border rounded-lg text-sm text-left flex items-center justify-between hover:border-gray-600 ${
              error ? 'border-red-500/50' : 'border-gray-700'
            }`}
          >
            <span className="text-gray-500">Select a prompt...</span>
            <ChevronDown
              size={16}
              className={`text-gray-400 transition-transform ${showDropdown ? 'rotate-180' : ''}`}
            />
          </button>

        {showDropdown && dropdownPos && createPortal(
          <div
            ref={dropdownRef}
            className="fixed z-[9999] bg-gray-800 border border-gray-700 rounded-lg shadow-lg max-h-72 overflow-y-auto"
            style={{ top: dropdownPos.top, left: dropdownPos.left, width: dropdownPos.width }}
          >
               {prompts.map((prompt) => (
                 <div key={prompt.id}>
                   <div
                     className="p-2 hover:bg-gray-700 cursor-pointer"
                     onClick={() => handleSelectPrompt(prompt)}
                   >
                     <div className="flex items-center justify-between">
                       <div className="flex-1 min-w-0">
                         <div className="text-sm text-white truncate">{prompt.name}</div>
                         {prompt.description && (
                           <div className="text-xs text-gray-400 truncate">
                             {prompt.description}
                           </div>
                         )}
                       </div>
                       <div className="flex items-center gap-1 ml-2">
                         <button
                           onClick={(e) => {
                             e.stopPropagation();
                             setExpandedPromptId(
                               expandedPromptId === prompt.id ? null : prompt.id
                             );
                           }}
                           className="p-1 hover:bg-gray-600 rounded text-gray-400"
                           title="View"
                         >
                           <Pencil size={12} />
                         </button>
                         <button
                           onClick={(e) => handleDeletePrompt(prompt.id, e)}
                           className="p-1 hover:bg-red-500/20 rounded text-gray-400 hover:text-red-400"
                           title="Delete"
                         >
                           <Trash2 size={12} />
                         </button>
                       </div>
                     </div>
                   </div>
                   {expandedPromptId === prompt.id && (
                     <div className="px-2 pb-2">
                       <div className="bg-gray-900 rounded p-2">
                         <pre className="text-xs text-gray-300 whitespace-pre-wrap overflow-x-auto max-h-24">
                           {prompt.content}
                         </pre>
                       </div>
                     </div>
                   )}
                 </div>
               ))}
               {prompts.length === 0 && (
                 <div className="p-3 text-sm text-gray-500 text-center">
                   No saved prompts yet
                 </div>
               )}
             </div>,
          document.body
        )}
        </div>
      )}

      {/* Error message */}
      {error && (
        <p className="text-xs text-red-400">{error}</p>
      )}

      {/* Placeholder */}
      {placeholder && !selectedPrompt && prompts.length === 0 && !showDropdown && (
        <p className="text-xs text-gray-500">{placeholder}</p>
      )}

      {/* Add new prompt button */}
      {!selectedPrompt && (
        <button
          onClick={() => {
            setShowCreateForm(!showCreateForm);
            setShowDropdown(false);
          }}
          className="w-full p-2 border border-dashed border-gray-600 rounded-lg text-sm text-gray-400 hover:text-white hover:border-gray-500 flex items-center justify-center gap-2"
        >
          <Plus size={14} />
          Create New Prompt
        </button>
      )}

      {/* Create prompt form */}
      {showCreateForm && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-3">
          <h4 className="text-sm font-medium text-white">Create New Prompt</h4>
          <div>
            <input
              type="text"
              value={newPromptName}
              onChange={(e) => setNewPromptName(e.target.value)}
              placeholder="Prompt name"
              className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm text-white"
            />
          </div>
          <div>
            <textarea
              value={newPromptContent}
              onChange={(e) => setNewPromptContent(e.target.value)}
              placeholder="Prompt content..."
              rows={4}
              className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm text-white resize-none"
            />
          </div>
          <div>
            <input
              type="text"
              value={newPromptDescription}
              onChange={(e) => setNewPromptDescription(e.target.value)}
              placeholder="Description (optional)"
              className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm text-white"
            />
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleCreatePrompt}
              disabled={!newPromptName || !newPromptContent}
              className="px-3 py-1.5 bg-violet-600 hover:bg-violet-700 disabled:bg-gray-700 rounded text-sm text-white"
            >
              Save
            </button>
            <button
              onClick={() => setShowCreateForm(false)}
              className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-sm text-white"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
