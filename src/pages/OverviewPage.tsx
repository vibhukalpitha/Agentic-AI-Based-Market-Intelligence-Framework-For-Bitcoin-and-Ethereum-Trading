import React from 'react';
import type { AssetType, TimeRange } from '../types';
import { MetricCard } from '../components/MetricCard';
import { MarketChart } from '../components/MarketChart';
import {
  AI_INTELLIGENCE_SUMMARY,
  MARKET_REGIME,
  OVERVIEW_KPI,
  getPriceData
} from '../data/mockData';
import {
  Coins,
  Layers,
  Flame,
  Activity,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  Sparkles,
  ShieldCheck
} from 'lucide-react';

interface OverviewPageProps {
  selectedAsset: AssetType;
  selectedTimeRange: TimeRange;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({
  selectedAsset,
  selectedTimeRange
}) => {
  const assetName = selectedAsset === 'ETH' ? 'ETH' : 'BTC';
  const priceData = getPriceData(assetName, selectedTimeRange);

  return (
    <div className="space-y-6">
      {/* Top KPI Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="BTC Price"
          value={OVERVIEW_KPI.btc.price}
          change={OVERVIEW_KPI.btc.change24h}
          isPositive={true}
          status={OVERVIEW_KPI.btc.status}
          subtext="24h High: $68,450.00"
          icon={<Coins className="w-4 h-4 text-amber-400" />}
          accentColor="amber"
        />

        <MetricCard
          title="ETH Price"
          value={OVERVIEW_KPI.eth.price}
          change={OVERVIEW_KPI.eth.change24h}
          isPositive={true}
          status={OVERVIEW_KPI.eth.status}
          subtext="24h High: $3,910.00"
          icon={<Layers className="w-4 h-4 text-indigo-400" />}
          accentColor="purple"
        />

        <MetricCard
          title="Market Sentiment"
          value={`${OVERVIEW_KPI.sentiment.score} / 100`}
          status={OVERVIEW_KPI.sentiment.label}
          subtext="Fear & Greed Index"
          icon={<Flame className="w-4 h-4 text-emerald-400" />}
          accentColor="green"
        />

        <MetricCard
          title="Derivatives Open Interest"
          value={OVERVIEW_KPI.openInterest.val}
          change={OVERVIEW_KPI.openInterest.change}
          isPositive={true}
          subtext="Aggregate Leverage"
          icon={<Activity className="w-4 h-4 text-blue-400" />}
          accentColor="blue"
        />
      </div>

      {/* Top Level Intelligence Summary Card */}
      <div className="glass-panel p-6 rounded-xl border border-blue-500/30 glow-blue relative overflow-hidden bg-gradient-to-r from-[#121626] via-[#12141c] to-[#151224]">
        <div className="flex items-center gap-2 mb-3">
          <div className="p-1.5 rounded-lg bg-blue-500/20 text-blue-400 border border-blue-500/30">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white tracking-tight">
              {AI_INTELLIGENCE_SUMMARY.title}
            </h2>
            <span className="text-[11px] text-blue-400 font-mono">
              XGBoost + LSTM + FinBERT Reasoning Synthesis
            </span>
          </div>
        </div>

        <p className="text-sm text-gray-200 leading-relaxed font-normal bg-[#0a0c14]/50 p-4 rounded-xl border border-[#1e2330]">
          "{AI_INTELLIGENCE_SUMMARY.synthesis}"
        </p>

        <div className="mt-4 pt-4 border-t border-[#1e2330]">
          <span className="text-xs font-bold text-gray-400 uppercase tracking-wider block mb-2.5">
            Key Intelligence Drivers
          </span>
          <div className="flex flex-wrap items-center gap-2.5">
            {AI_INTELLIGENCE_SUMMARY.keyDrivers.map((kd) => (
              <div
                key={kd.name}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#0d0f17] border border-[#1e2330] text-xs"
              >
                <span className="font-semibold text-gray-300">{kd.name}:</span>
                <span
                  className={`font-bold px-2 py-0.5 rounded text-[11px] ${
                    kd.state.toLowerCase().includes('bullish')
                      ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                      : kd.state.toLowerCase().includes('bearish')
                      ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                      : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                  }`}
                >
                  {kd.state}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main Chart + Market Regime Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <MarketChart
            data={priceData}
            title={`${assetName} Price & Volume Overview (${selectedTimeRange})`}
            asset={assetName}
            height={360}
          />
        </div>

        <div className="glass-panel p-5 rounded-xl border border-[#1e2330] flex flex-col justify-between space-y-4">
          <div>
            <div className="flex items-center justify-between border-b border-[#1e2330] pb-3 mb-4">
              <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                Market Regime
              </h3>
              <span className="px-2.5 py-1 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs font-bold">
                {MARKET_REGIME.status}
              </span>
            </div>

            <div className="p-4 rounded-xl bg-[#0d0f17] border border-[#1e2330] text-center mb-4">
              <span className="text-xs text-gray-400 block uppercase font-medium">Model Confidence</span>
              <div className="text-3xl font-black text-emerald-400 font-mono mt-1">
                {MARKET_REGIME.confidence}%
              </div>
              <div className="w-full bg-[#1e2330] h-1.5 rounded-full mt-3 overflow-hidden">
                <div
                  className="bg-emerald-400 h-full rounded-full"
                  style={{ width: `${MARKET_REGIME.confidence}%` }}
                ></div>
              </div>
            </div>

            <div className="space-y-2">
              <span className="text-xs font-bold text-gray-400 uppercase tracking-wider block">
                Regime Contributing Factors
              </span>
              {MARKET_REGIME.factors.map((factor) => (
                <div
                  key={factor.label}
                  className="flex items-center justify-between p-2.5 rounded-lg bg-[#0d0f17] border border-[#1e2330] text-xs"
                >
                  <span className="text-gray-300 font-medium">{factor.label}</span>
                  <span className="flex items-center gap-1 font-bold text-emerald-400">
                    {factor.direction === 'up' ? (
                      <ArrowUpRight className="w-3.5 h-3.5" />
                    ) : factor.direction === 'down' ? (
                      <ArrowDownRight className="w-3.5 h-3.5" />
                    ) : (
                      <Minus className="w-3.5 h-3.5 text-amber-400" />
                    )}
                    {factor.status}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="p-3 rounded-lg bg-[#12141c] border border-[#1e2330] text-[11px] text-gray-400 flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-blue-400 shrink-0" />
            <span>Regime inferred from 14 cross-market macro & crypto micro indicators.</span>
          </div>
        </div>
      </div>
    </div>
  );
};
