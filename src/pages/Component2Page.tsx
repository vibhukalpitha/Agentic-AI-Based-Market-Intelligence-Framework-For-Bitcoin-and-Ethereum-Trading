import React, { useState } from 'react';
import type { AssetType, Component2SubTab, ForecastHorizon, TimeRange } from '../types';
import { OverviewPage } from './OverviewPage';
import { BtcIntelligencePage } from './BtcIntelligencePage';
import { EthIntelligencePage } from './EthIntelligencePage';
import { CrossMarketPage } from './CrossMarketPage';
import { CapitalFlowPage } from './CapitalFlowPage';
import { NewsSentimentPage } from './NewsSentimentPage';
import { PredictionPage } from './PredictionPage';
import { HistoricalEvidencePage } from './HistoricalEvidencePage';
import { ExplainabilityPage } from './ExplainabilityPage';
import {
  LayoutDashboard,
  Coins,
  Layers,
  GitCompare,
  TrendingUp,
  Newspaper,
  Target,
  History,
  BrainCircuit,
  Clock,
  Sparkles
} from 'lucide-react';

export const Component2Page: React.FC = () => {
  const [subTab, setSubTab] = useState<Component2SubTab>('overview');
  const [selectedAsset, setSelectedAsset] = useState<AssetType>('ALL');
  const [selectedTimeRange, setSelectedTimeRange] = useState<TimeRange>('1H');
  const [forecastHorizon, setForecastHorizon] = useState<ForecastHorizon>('4H');

  const subNavItems = [
    { id: 'overview' as Component2SubTab, label: 'Overview', icon: LayoutDashboard },
    { id: 'btc-intelligence' as Component2SubTab, label: 'BTC Intelligence', icon: Coins },
    { id: 'eth-intelligence' as Component2SubTab, label: 'ETH Intelligence', icon: Layers },
    { id: 'cross-market' as Component2SubTab, label: 'Cross-Market', icon: GitCompare },
    { id: 'capital-flow' as Component2SubTab, label: 'Capital Flow', icon: TrendingUp },
    { id: 'news-sentiment' as Component2SubTab, label: 'News & Sentiment', icon: Newspaper },
    { id: 'prediction' as Component2SubTab, label: 'Prediction', icon: Target },
    { id: 'historical-evidence' as Component2SubTab, label: 'Historical Evidence', icon: History },
    { id: 'explainability' as Component2SubTab, label: 'Explainability', icon: BrainCircuit },
  ];

  const renderSubContent = () => {
    switch (subTab) {
      case 'overview':
        return <OverviewPage selectedAsset={selectedAsset} selectedTimeRange={selectedTimeRange} />;
      case 'btc-intelligence':
        return <BtcIntelligencePage selectedTimeRange={selectedTimeRange} />;
      case 'eth-intelligence':
        return <EthIntelligencePage selectedTimeRange={selectedTimeRange} />;
      case 'cross-market':
        return <CrossMarketPage />;
      case 'capital-flow':
        return <CapitalFlowPage />;
      case 'news-sentiment':
        return <NewsSentimentPage />;
      case 'prediction':
        return <PredictionPage selectedAsset={selectedAsset} forecastHorizon={forecastHorizon} />;
      case 'historical-evidence':
        return <HistoricalEvidencePage />;
      case 'explainability':
        return <ExplainabilityPage />;
      default:
        return <OverviewPage selectedAsset={selectedAsset} selectedTimeRange={selectedTimeRange} />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Component 2 Title Header & Secondary Navigation Bar */}
      <div className="glass-panel p-6 rounded-xl border border-purple-500/30 glow-purple bg-gradient-to-r from-[#121524] via-[#12141c] to-[#181226] space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#1e2330] pb-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded bg-purple-500/20 text-purple-300 font-mono text-xs font-bold border border-purple-500/30">
                COMPONENT 2 CORE MODULE
              </span>
              <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 text-[10px] font-bold uppercase tracking-wider">
                <Sparkles className="w-3 h-3" />
                PROTOTYPE • MOCK DATA
              </span>
            </div>
            <h2 className="text-xl sm:text-2xl font-extrabold text-white tracking-tight mt-1">
              Cross-Market & Capital Flow Intelligence
            </h2>
            <p className="text-xs text-gray-300 mt-0.5">
              BTC & ETH short-horizon market intelligence — research engine module
            </p>
          </div>

          {/* Sub Controls Bar */}
          <div className="flex flex-wrap items-center gap-2">
            {/* Asset Selector */}
            <div className="flex items-center bg-[#0d0f17] border border-[#1f2330] rounded-lg p-1 text-xs">
              {(['ALL', 'BTC', 'ETH'] as AssetType[]).map((asset) => (
                <button
                  key={asset}
                  onClick={() => setSelectedAsset(asset)}
                  className={`px-3 py-1 rounded-md font-semibold transition-all cursor-pointer ${
                    selectedAsset === asset
                      ? 'bg-blue-600 text-white shadow-sm'
                      : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  {asset === 'ALL' ? 'Multi-Asset' : asset}
                </button>
              ))}
            </div>

            {/* Interval Selector */}
            <div className="flex items-center bg-[#0d0f17] border border-[#1f2330] rounded-lg p-1 text-xs">
              <span className="px-2 text-gray-400 font-medium text-[11px] flex items-center gap-1">
                <Clock className="w-3 h-3 text-gray-400" />
                Time:
              </span>
              {(['1H', '4H', '1D', '1W'] as TimeRange[]).map((range) => (
                <button
                  key={range}
                  onClick={() => setSelectedTimeRange(range)}
                  className={`px-2.5 py-1 rounded-md font-semibold transition-all cursor-pointer ${
                    selectedTimeRange === range
                      ? 'bg-indigo-600 text-white shadow-sm'
                      : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  {range}
                </button>
              ))}
            </div>

            {/* Horizon Selector */}
            <div className="flex items-center bg-[#0d0f17] border border-[#1f2330] rounded-lg p-1 text-xs">
              <span className="px-2 text-gray-400 font-medium text-[11px]">Forecast:</span>
              {(['1H', '2H', '3H', '4H'] as ForecastHorizon[]).map((horizon) => (
                <button
                  key={horizon}
                  onClick={() => setForecastHorizon(horizon)}
                  className={`px-2 py-1 rounded-md font-semibold transition-all cursor-pointer ${
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
        </div>

        {/* Secondary Sub-Navigation Tabs (9 Internal Component 2 Sections) */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1">
          {subNavItems.map((item) => {
            const Icon = item.icon;
            const isActive = subTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setSubTab(item.id)}
                className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-bold transition-all whitespace-nowrap cursor-pointer ${
                  isActive
                    ? 'bg-purple-600 text-white shadow-md border border-purple-400/40'
                    : 'bg-[#0d0f17] text-gray-400 hover:text-gray-200 border border-[#1e2330] hover:bg-[#151824]'
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-white' : 'text-gray-400'}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Render Selected Internal Component 2 Dashboard View */}
      {renderSubContent()}
    </div>
  );
};
