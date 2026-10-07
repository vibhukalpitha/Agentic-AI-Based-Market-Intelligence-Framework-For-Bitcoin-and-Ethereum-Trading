import React from 'react';
import type { TimeRange } from '../types';
import { MarketChart } from '../components/MarketChart';
import { DriverImpactCard } from '../components/DriverImpactCard';
import {
  ETH_DRIVERS,
  ETH_INTERNAL_SIGNALS,
  OVERVIEW_KPI,
  getPriceData
} from '../data/mockData';
import {
  ArrowUpRight,
  ArrowDownRight,
  Minus
} from 'lucide-react';

interface EthIntelligencePageProps {
  selectedTimeRange: TimeRange;
}

export const EthIntelligencePage: React.FC<EthIntelligencePageProps> = ({
  selectedTimeRange
}) => {
  const priceData = getPriceData('ETH', selectedTimeRange);

  return (
    <div className="space-y-6">
      {/* ETH Header Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
        <div className="glass-panel p-3.5 rounded-xl border border-[#1e2330]">
          <span className="text-[11px] font-semibold text-gray-400 uppercase">ETH Price</span>
          <div className="text-lg font-bold text-white mt-1">{OVERVIEW_KPI.eth.price}</div>
          <span className="text-[10px] text-emerald-400 font-semibold">{OVERVIEW_KPI.eth.change24h} (24h)</span>
        </div>

        <div className="glass-panel p-3.5 rounded-xl border border-[#1e2330]">
          <span className="text-[11px] font-semibold text-gray-400 uppercase">1h Return</span>
          <div className="text-lg font-bold text-emerald-400 mt-1">+0.29%</div>
          <span className="text-[10px] text-gray-400">Short Horizon</span>
        </div>

        <div className="glass-panel p-3.5 rounded-xl border border-[#1e2330]">
          <span className="text-[11px] font-semibold text-gray-400 uppercase">4h Return</span>
          <div className="text-lg font-bold text-emerald-400 mt-1">+1.15%</div>
          <span className="text-[10px] text-gray-400">Target Horizon</span>
        </div>

        <div className="glass-panel p-3.5 rounded-xl border border-[#1e2330]">
          <span className="text-[11px] font-semibold text-gray-400 uppercase">24h Volume</span>
          <div className="text-lg font-bold text-white mt-1">$18.2B</div>
          <span className="text-[10px] text-gray-400">Spot + Derivatives</span>
        </div>

        <div className="glass-panel p-3.5 rounded-xl border border-[#1e2330]">
          <span className="text-[11px] font-semibold text-gray-400 uppercase">Volatility</span>
          <div className="text-lg font-bold text-amber-400 mt-1">2.45%</div>
          <span className="text-[10px] text-gray-400">Annualized 1h σ</span>
        </div>

        <div className="glass-panel p-3.5 rounded-xl border border-[#1e2330]">
          <span className="text-[11px] font-semibold text-gray-400 uppercase">Open Interest</span>
          <div className="text-lg font-bold text-indigo-400 mt-1">$9.15B</div>
          <span className="text-[10px] text-emerald-400 font-semibold">+3.9%</span>
        </div>

        <div className="glass-panel p-3.5 rounded-xl border border-[#1e2330]">
          <span className="text-[11px] font-semibold text-gray-400 uppercase">Sentiment</span>
          <div className="text-lg font-bold text-emerald-400 mt-1">65 / 100</div>
          <span className="text-[10px] text-gray-400">Bullish Bias</span>
        </div>
      </div>

      <MarketChart
        data={priceData}
        title={`Ethereum (ETH) Price & Volume Dynamics (${selectedTimeRange})`}
        asset="ETH"
        height={340}
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <DriverImpactCard drivers={ETH_DRIVERS} assetName="Ethereum" />

        <div className="glass-panel p-5 rounded-xl border border-[#1e2330] space-y-4">
          <div className="flex items-center justify-between border-b border-[#1e2330] pb-3">
            <div>
              <h3 className="text-sm font-bold text-white uppercase tracking-wider">ETH Internal Network Signals</h3>
              <p className="text-xs text-gray-400 mt-0.5">Staking, gas burn, and L2 layer dynamics</p>
            </div>
            <span className="text-[11px] font-mono text-gray-400 bg-[#0d0f17] px-2.5 py-1 rounded border border-[#1e2330]">
              Ethereum Protocol
            </span>
          </div>

          <div className="space-y-3">
            {ETH_INTERNAL_SIGNALS.map((signal) => (
              <div
                key={signal.name}
                className="p-3.5 rounded-lg bg-[#0d0f17] border border-[#1e2330] flex items-center justify-between hover:border-[#2b3145] transition-all"
              >
                <div>
                  <div className="text-xs font-bold text-gray-200">{signal.name}</div>
                  <div className="text-sm font-extrabold text-white font-mono mt-0.5">{signal.value}</div>
                </div>

                <div className="text-right">
                  <div className="flex items-center gap-1 justify-end">
                    {signal.trend === 'up' ? (
                      <ArrowUpRight className="w-4 h-4 text-emerald-400" />
                    ) : signal.trend === 'down' ? (
                      <ArrowDownRight className="w-4 h-4 text-rose-400" />
                    ) : (
                      <Minus className="w-4 h-4 text-amber-400" />
                    )}
                    <span
                      className={`text-xs font-bold px-2 py-0.5 rounded capitalize ${
                        signal.status === 'bullish'
                          ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                          : signal.status === 'bearish'
                          ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                          : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                      }`}
                    >
                      {signal.status}
                    </span>
                  </div>
                  <span className="text-[11px] text-gray-400 mt-1 block">{signal.change24h}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
