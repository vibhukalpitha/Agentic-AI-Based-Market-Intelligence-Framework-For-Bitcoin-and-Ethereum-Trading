import React from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string;
  change?: string;
  isPositive?: boolean;
  status?: string;
  subtext?: string;
  icon?: React.ReactNode;
  accentColor?: 'blue' | 'purple' | 'green' | 'amber';
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  change,
  isPositive,
  status,
  subtext,
  icon,
  accentColor = 'blue'
}) => {
  const getAccentBorder = () => {
    switch (accentColor) {
      case 'green': return 'hover:border-emerald-500/40';
      case 'purple': return 'hover:border-purple-500/40';
      case 'amber': return 'hover:border-amber-500/40';
      default: return 'hover:border-blue-500/40';
    }
  };

  return (
    <div className={`glass-panel p-4 rounded-xl border border-[#1e2330] glass-panel-hover ${getAccentBorder()} flex flex-col justify-between`}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{title}</span>
        {icon && <div className="p-1.5 rounded-lg bg-[#1a1d29] text-gray-400">{icon}</div>}
      </div>

      <div className="mt-3 flex items-baseline justify-between">
        <span className="text-2xl font-extrabold text-white tracking-tight">{value}</span>
        {change && (
          <span
            className={`inline-flex items-center gap-1 text-xs font-bold px-2 py-0.5 rounded-md ${
              isPositive
                ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
            }`}
          >
            {isPositive ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
            {change}
          </span>
        )}
      </div>

      {(status || subtext) && (
        <div className="mt-2 pt-2 border-t border-[#1a1d29] flex items-center justify-between text-xs text-gray-400">
          {status && (
            <span
              className={`font-semibold px-2 py-0.5 rounded text-[11px] ${
                status.toLowerCase().includes('bullish')
                  ? 'text-emerald-400 bg-emerald-500/10 border border-emerald-500/20'
                  : status.toLowerCase().includes('bearish')
                  ? 'text-rose-400 bg-rose-500/10 border border-rose-500/20'
                  : 'text-amber-400 bg-amber-500/10 border border-amber-500/20'
              }`}
            >
              {status}
            </span>
          )}
          {subtext && <span className="text-[11px] text-gray-400">{subtext}</span>}
        </div>
      )}
    </div>
  );
};
