import React from 'react';
import type { MainTab } from '../types';
import { Cpu, LineChart, Activity, ShieldCheck, ArrowRight, Sparkles } from 'lucide-react';

interface HomePageProps {
  onNavigateToTab: (tab: MainTab) => void;
}

export const HomePage: React.FC<HomePageProps> = ({ onNavigateToTab }) => {
  return (
    <div className="space-y-8">
      {/* System Hero Banner */}
      <div className="glass-panel p-8 rounded-2xl border border-blue-500/30 glow-blue bg-gradient-to-r from-[#101426] via-[#12141c] to-[#171228] relative overflow-hidden">
        <div className="max-w-3xl space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 text-blue-400 text-xs font-bold border border-blue-500/20 uppercase tracking-wider">
            <Sparkles className="w-3.5 h-3.5" />
            University Research Platform Framework
          </div>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
            Integrated University Market Intelligence Research System
          </h2>
          <p className="text-sm text-gray-300 leading-relaxed font-normal">
            A comprehensive four-module architecture combining real-time data ingestion, cross-market machine learning, portfolio risk optimization, and automated compliance auditing.
          </p>
        </div>
      </div>

      {/* 4 Components Grid Overview */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-base font-bold text-white uppercase tracking-wider">
            Research System Subsystems (4 Components)
          </h3>
          <span className="text-xs font-mono text-gray-400">Integrated Academic Architecture</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* COMPONENT 1 CARD */}
          <div className="glass-panel p-6 rounded-xl border border-[#1e2330] hover:border-blue-500/40 transition-all flex flex-col justify-between space-y-5">
            <div className="space-y-3">
              <div className="flex items-center justify-between border-b border-[#1e2330] pb-3">
                <div className="flex items-center gap-2.5">
                  <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/20">
                    <Cpu className="w-5 h-5" />
                  </div>
                  <div>
                    <span className="text-[11px] font-bold text-blue-400 uppercase font-mono">COMPONENT 1</span>
                    <h4 className="text-base font-bold text-white tracking-tight">Data Ingestion & Preprocessing Pipeline</h4>
                  </div>
                </div>
                <span className="px-2.5 py-1 rounded bg-amber-500/10 text-amber-400 text-xs font-bold border border-amber-500/20">
                  Prototype
                </span>
              </div>

              <p className="text-xs text-gray-300 leading-relaxed">
                Real-time streaming multi-exchange orderbook parser and high-frequency data ingestion framework.
              </p>

              <div className="grid grid-cols-2 gap-3 pt-2">
                <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
                  <span className="text-[11px] text-gray-400 uppercase">Throughput</span>
                  <div className="text-base font-extrabold text-white font-mono mt-0.5">142.8k msgs/s</div>
                </div>
                <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
                  <span className="text-[11px] text-gray-400 uppercase">Avg Latency</span>
                  <div className="text-base font-extrabold text-emerald-400 font-mono mt-0.5">&lt;4.2ms</div>
                </div>
              </div>
            </div>

            <button
              onClick={() => onNavigateToTab('component-1')}
              className="w-full py-2.5 rounded-lg bg-[#141824] hover:bg-blue-600 text-blue-400 hover:text-white border border-blue-500/30 text-xs font-bold transition-all flex items-center justify-center gap-2 cursor-pointer"
            >
              <span>Explore Component 1</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>

          {/* COMPONENT 2 CARD (PRIMARY ACTIVE MODULE) */}
          <div className="glass-panel p-6 rounded-xl border border-purple-500/40 glow-purple bg-gradient-to-br from-[#121526] via-[#12141c] to-[#181228] flex flex-col justify-between space-y-5">
            <div className="space-y-3">
              <div className="flex items-center justify-between border-b border-[#1e2330] pb-3">
                <div className="flex items-center gap-2.5">
                  <div className="p-2 rounded-lg bg-purple-500/20 text-purple-400 border border-purple-500/30">
                    <LineChart className="w-5 h-5" />
                  </div>
                  <div>
                    <span className="text-[11px] font-bold text-purple-400 uppercase font-mono">COMPONENT 2</span>
                    <h4 className="text-base font-bold text-white tracking-tight">Cross-Market & Capital Flow Intelligence</h4>
                  </div>
                </div>
                <span className="px-2.5 py-1 rounded bg-purple-500/20 text-purple-300 text-xs font-bold border border-purple-500/30">
                  Core Module Active
                </span>
              </div>

              <p className="text-xs text-gray-200 leading-relaxed">
                Short-horizon BTC and ETH intelligence using cross-market signals, internal crypto activity, news sentiment, historical evidence and explainable model outputs.
              </p>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2">
                <div className="p-2.5 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
                  <span className="text-[10px] text-gray-400 uppercase">BTC Price</span>
                  <div className="text-sm font-extrabold text-white font-mono mt-0.5">$67,842</div>
                </div>
                <div className="p-2.5 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
                  <span className="text-[10px] text-gray-400 uppercase">ETH Price</span>
                  <div className="text-sm font-extrabold text-white font-mono mt-0.5">$3,842</div>
                </div>
                <div className="p-2.5 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
                  <span className="text-[10px] text-gray-400 uppercase">Sentiment</span>
                  <div className="text-sm font-extrabold text-emerald-400 font-mono mt-0.5">68</div>
                </div>
                <div className="p-2.5 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
                  <span className="text-[10px] text-gray-400 uppercase">Confidence</span>
                  <div className="text-sm font-extrabold text-purple-400 font-mono mt-0.5">78%</div>
                </div>
              </div>
            </div>

            <button
              onClick={() => onNavigateToTab('component-2')}
              className="w-full py-2.5 rounded-lg bg-purple-600 hover:bg-purple-500 text-white text-xs font-bold transition-all shadow-md flex items-center justify-center gap-2 cursor-pointer"
            >
              <span>Open Component 2 Intelligence Engine</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>

          {/* COMPONENT 3 CARD */}
          <div className="glass-panel p-6 rounded-xl border border-[#1e2330] hover:border-blue-500/40 transition-all flex flex-col justify-between space-y-5">
            <div className="space-y-3">
              <div className="flex items-center justify-between border-b border-[#1e2330] pb-3">
                <div className="flex items-center gap-2.5">
                  <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                    <Activity className="w-5 h-5" />
                  </div>
                  <div>
                    <span className="text-[11px] font-bold text-indigo-400 uppercase font-mono">COMPONENT 3</span>
                    <h4 className="text-base font-bold text-white tracking-tight">Portfolio Risk & Execution Optimization</h4>
                  </div>
                </div>
                <span className="px-2.5 py-1 rounded bg-amber-500/10 text-amber-400 text-xs font-bold border border-amber-500/20">
                  Prototype
                </span>
              </div>

              <p className="text-xs text-gray-300 leading-relaxed">
                Algorithmic trade execution, dynamic slippage modeling, and multi-asset tail-risk protection module.
              </p>

              <div className="grid grid-cols-2 gap-3 pt-2">
                <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
                  <span className="text-[11px] text-gray-400 uppercase">Sharpe Ratio</span>
                  <div className="text-base font-extrabold text-emerald-400 font-mono mt-0.5">2.84</div>
                </div>
                <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
                  <span className="text-[11px] text-gray-400 uppercase">Max Drawdown</span>
                  <div className="text-base font-extrabold text-rose-400 font-mono mt-0.5">4.1%</div>
                </div>
              </div>
            </div>

            <button
              onClick={() => onNavigateToTab('component-3')}
              className="w-full py-2.5 rounded-lg bg-[#141824] hover:bg-blue-600 text-blue-400 hover:text-white border border-blue-500/30 text-xs font-bold transition-all flex items-center justify-center gap-2 cursor-pointer"
            >
              <span>Explore Component 3</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>

          {/* COMPONENT 4 CARD */}
          <div className="glass-panel p-6 rounded-xl border border-[#1e2330] hover:border-blue-500/40 transition-all flex flex-col justify-between space-y-5">
            <div className="space-y-3">
              <div className="flex items-center justify-between border-b border-[#1e2330] pb-3">
                <div className="flex items-center gap-2.5">
                  <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    <ShieldCheck className="w-5 h-5" />
                  </div>
                  <div>
                    <span className="text-[11px] font-bold text-emerald-400 uppercase font-mono">COMPONENT 4</span>
                    <h4 className="text-base font-bold text-white tracking-tight">Automated Compliance & Audit Trail</h4>
                  </div>
                </div>
                <span className="px-2.5 py-1 rounded bg-amber-500/10 text-amber-400 text-xs font-bold border border-amber-500/20">
                  Prototype
                </span>
              </div>

              <p className="text-xs text-gray-300 leading-relaxed">
                On-chain transaction verification, MEV detection, and regulatory reporting suite.
              </p>

              <div className="grid grid-cols-2 gap-3 pt-2">
                <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
                  <span className="text-[11px] text-gray-400 uppercase">Compliance Index</span>
                  <div className="text-base font-extrabold text-emerald-400 font-mono mt-0.5">99.4%</div>
                </div>
                <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
                  <span className="text-[11px] text-gray-400 uppercase">Audit Verification</span>
                  <div className="text-base font-extrabold text-blue-400 font-mono mt-0.5">Verified</div>
                </div>
              </div>
            </div>

            <button
              onClick={() => onNavigateToTab('component-4')}
              className="w-full py-2.5 rounded-lg bg-[#141824] hover:bg-blue-600 text-blue-400 hover:text-white border border-blue-500/30 text-xs font-bold transition-all flex items-center justify-center gap-2 cursor-pointer"
            >
              <span>Explore Component 4</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
