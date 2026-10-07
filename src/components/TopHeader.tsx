import React from 'react';
import type { AssetType, ForecastHorizon, TimeRange } from '../types';
import { Clock, Sparkles } from 'lucide-react';

interface TopHeaderProps {
  title: string;
  subtitle?: string;
  selectedAsset: AssetType;
  setSelectedAsset: (asset: AssetType) => void;
  selectedTimeRange: TimeRange;
  setSelectedTimeRange: (range: TimeRange) => void;
  forecastHorizon: ForecastHorizon;
  setForecastHorizon: (horizon: ForecastHorizon) => void;
}

export const TopHeader: React.FC<TopHeaderProps> = ({
  title,
  subtitle,
  selectedAsset,
  setSelectedAsset,
  selectedTimeRange,
  setSelectedTimeRange,
  forecastHorizon,
  setForecastHorizon,
}) => {
  return (
    <header className="sticky top-0 z-20 bg-[#0b0c10]/90 backdrop-blur-md border-b border-[#1e2330] px-6 py-4 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
      <div>
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-bold text-white tracking-tight">{title}</h1>
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 text-[10px] font-bold uppercase tracking-wider">
            <Sparkles className="w-3 h-3" />
            PROTOTYPE • MOCK DATA
          </span>
        </div>
        {subtitle && <p className="text-xs text-gray-400 mt-0.5">{subtitle}</p>}
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center bg-[#12141c] border border-[#1f2330] rounded-lg p-1 text-xs">
          {(['ALL', 'BTC', 'ETH'] as AssetType[]).map((asset) => (
            <button
              key={asset}
              onClick={() => setSelectedAsset(asset)}
              className={`px-3 py-1 rounded-md font-semibold transition-all ${
                selectedAsset === asset
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              {asset === 'ALL' ? 'Multi-Asset' : asset}
            </button>
          ))}
        </div>

        <div className="flex items-center bg-[#12141c] border border-[#1f2330] rounded-lg p-1 text-xs">
          <span className="px-2 text-gray-400 font-medium text-[11px] flex items-center gap-1">
            <Clock className="w-3 h-3 text-gray-400" />
            Interval:
          </span>
          {(['1H', '4H', '1D', '1W'] as TimeRange[]).map((range) => (
            <button
              key={range}
              onClick={() => setSelectedTimeRange(range)}
              className={`px-2.5 py-1 rounded-md font-semibold transition-all ${
                selectedTimeRange === range
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              {range}
            </button>
          ))}
        </div>

        <div className="flex items-center bg-[#12141c] border border-[#1f2330] rounded-lg p-1 text-xs">
          <span className="px-2 text-gray-400 font-medium text-[11px]">Forecast:</span>
          {(['1H', '2H', '3H', '4H'] as ForecastHorizon[]).map((horizon) => (
            <button
              key={horizon}
              onClick={() => setForecastHorizon(horizon)}
              className={`px-2 py-1 rounded-md font-semibold transition-all ${
                forecastHorizon === horizon
                  ? 'bg-purple-600 text-white shadow-sm'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              +{horizon}
            </button>
          ))}
        </div>
      </div>
    </header>
  );
};
