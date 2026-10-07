import React from 'react';
import type { TimeRange } from '../types';
import { MarketChart } from '../components/MarketChart';
import { DriverImpactCard } from '../components/DriverImpactCard';
import {
  BTC_DRIVERS,
  BTC_INTERNAL_SIGNALS,
  OVERVIEW_KPI,
  getPriceData
} from '../data/mockData';
import {
  ArrowUpRight,
  ArrowDownRight,
  Minus
} from 'lucide-react';

interface BtcIntelligencePageProps {
  selectedTimeRange: TimeRange;
}

export const BtcIntelligencePage: React.FC<BtcIntelligencePageProps> = ({
  selectedTimeRange
}) => {
  const priceData = getPriceData('BTC', selectedTimeRange);

  return (
    <div className="space-y-6">
      {/* BTC Header Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
        <div className="glass-panel p-3.5 rounded-xl border border-[#1e2330]">
          <span className="text-[11px] font-semibold text-gray-400 uppercase">BTC Price</span>
          <div className="text-lg font-bold text-white mt-1">{OVERVIEW_KPI.btc.price}</div>
          <span className="text-[10px] text-emerald-400 font-semibold">{OVERVIEW_KPI.btc.change24h} (24h)</span>
        </div>

        <div className="glass-panel p-3.5 rounded-xl border border-[#1e2330]">
          <span className="text-[11px] font-semibold text-gray-400 uppercase">1h Return</span>
          <div className="text-lg font-bold text-emerald-400 mt-1">+0.38%</div>
          <span className="text-[10px] text-gray-400">Short Horizon</span>
        </div>

        <div className="glass-panel p-3.5 rounded-xl border border-[#1e2330]">
          <span className="text-[11px] font-semibold text-gray-400 uppercase">4h Return</span>
          <div className="text-lg font-bold text-emerald-400 mt-1">+1.42%</div>
          <span className="text-[10px] text-gray-400">Target Horizon</span>
        </div>

        <div className="glass-panel p-3.5 rounded-xl border border-[#1e2330]">
          <span className="text-[11px] font-semibold text-gray-400 uppercase">24h Volume</span>
          <div className="text-lg font-bold text-white mt-1">$34.8B</div>
          <span className="text-[10px] text-gray-400">Spot + Futures</span>
        </div>

        <div className="glass-panel p-3.5 rounded-xl border border-[#1e2330]">
          <span className="text-[11px] font-semibold text-gray-400 uppercase">Volatility</span>
          <div className="text-lg font-bold text-amber-400 mt-1">2.14%</div>
          <span className="text-[10px] text-gray-400">Annualized 1h σ</span>
        </div>

        <div className="glass-panel p-3.5 rounded-xl border border-[#1e2330]">
          <span className="text-[11px] font-semibold text-gray-400 uppercase">Open Interest</span>
          <div className="text-lg font-bold text-blue-400 mt-1">{OVERVIEW_KPI.openInterest.val}</div>
          <span className="text-[10px] text-emerald-400 font-semibold">{OVERVIEW_KPI.openInterest.change}</span>
        </div>

        <div className="glass-panel p-3.5 rounded-xl border border-[#1e2330]">
          <span className="text-[11px] font-semibold text-gray-400 uppercase">Fear & Greed</span>
          <div className="text-lg font-bold text-emerald-400 mt-1">68 / 100</div>
          <span className="text-[10px] text-gray-400">Greed Regime</span>
        </div>
      </div>

      <MarketChart
        data={priceData}
        title={`Bitcoin (BTC) Price & Volume Dynamics (${selectedTimeRange})`}
        asset="BTC"
        height={340}
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <DriverImpactCard drivers={BTC_DRIVERS} assetName="Bitcoin" />

        <div className="glass-panel p-5 rounded-xl border border-[#1e2330] space-y-4">
          <div className="flex items-center justify-between border-b border-[#1e2330] pb-3">
            <div>
              <h3 className="text-sm font-bold text-white uppercase tracking-wider">Internal Market Signals</h3>
              <p className="text-xs text-gray-400 mt-0.5">On-chain & market microstructure metrics</p>
            </div>
            <span className="text-[11px] font-mono text-gray-400 bg-[#0d0f17] px-2.5 py-1 rounded border border-[#1e2330]">
              Crypto Native
            </span>
          </div>

          <div className="space-y-3">
            {BTC_INTERNAL_SIGNALS.map((signal) => (
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
