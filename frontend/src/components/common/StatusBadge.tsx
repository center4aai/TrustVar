import React from 'react';
import { Clock, Activity, Pause, CheckCircle, AlertCircle, XCircle, Download } from 'lucide-react';

interface StatusBadgeProps {
  status: string;
  size?: 'sm' | 'md';
}

export default function StatusBadge({ status, size = 'sm' }: StatusBadgeProps) {
  const configs: Record<string, any> = {
    pending: {
      bg: 'bg-amber-500/10',
      text: 'text-amber-400',
      border: 'border-amber-500/20',
      icon: Clock
    },
    running: {
      bg: 'bg-blue-500/10',
      text: 'text-blue-400',
      border: 'border-blue-500/20',
      icon: Activity
    },
    paused: {
      bg: 'bg-orange-500/10',
      text: 'text-orange-400',
      border: 'border-orange-500/20',
      icon: Pause
    },
    completed: {
      bg: 'bg-emerald-500/10',
      text: 'text-emerald-400',
      border: 'border-emerald-500/20',
      icon: CheckCircle
    },
    failed: {
      bg: 'bg-red-500/10',
      text: 'text-red-400',
      border: 'border-red-500/20',
      icon: AlertCircle
    },
    cancelled: {
      bg: 'bg-gray-500/10',
      text: 'text-gray-400',
      border: 'border-gray-500/20',
      icon: XCircle
    },
    registered: {
      bg: 'bg-emerald-500/10',
      text: 'text-emerald-400',
      border: 'border-emerald-500/20',
      icon: CheckCircle
    },
    downloading: {
      bg: 'bg-blue-500/10',
      text: 'text-blue-400',
      border: 'border-blue-500/20',
      icon: Download
    },
  };

  const config = configs[status] || configs.pending;
  const Icon = config.icon;
  const iconSize = size === 'sm' ? 12 : 14;
  const textSize = size === 'sm' ? 'text-xs' : 'text-sm';

  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md ${textSize} font-medium border ${config.bg} ${config.text} ${config.border}`}>
      <Icon size={iconSize} />
      {status.toUpperCase()}
    </span>
  );
}