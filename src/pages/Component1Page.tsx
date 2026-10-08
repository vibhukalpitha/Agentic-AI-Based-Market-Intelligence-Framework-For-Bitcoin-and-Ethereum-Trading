import React from 'react';
import { Cpu, Sparkles, Database, Layers } from 'lucide-react';

export const Component1Page: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="glass-panel p-6 rounded-xl border border-[#1e2330] space-y-3 bg-gradient-to-r from-[#121522] via-[#12141c] to-[#161226]">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-blue-500/10 text-blue-400 border border-blue-500/20">
              <Cpu className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold text-blue-400 uppercase font-mono">COMPONENT 1</span>
                <span className="px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 text-[10px] font-bold border border-amber-500/20">
                  PROTOTYPE STATUS
                </span>
              </div>
              <h2 className="text-xl font-extrabold text-white tracking-tight">
                Data Ingestion & Preprocessing Pipeline
              </h2>
            </div>
          </div>
          <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-blue-500/10 text-blue-400 text-xs font-mono font-bold border border-blue-500/20">
            <Sparkles className="w-3.5 h-3.5" /> MOCK MODULE
          </span>
        </div>

        {/* User Required Placeholder Text */}
        <div className="p-4 rounded-xl bg-[#0d0f17] border border-[#1e2330] text-center py-6">
          <p className="text-lg font-bold text-gray-200">
            “Module interface will be integrated here.”
          </p>
          <p className="text-xs text-gray-400 mt-1">
            Real-time streaming multi-exchange orderbook parser and high-frequency data ingestion framework placeholder.
          </p>
        </div>
      </div>

      {/* 2-3 Generic Mock KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="glass-panel p-4 rounded-xl border border-[#1e2330]">
          <span className="text-xs text-gray-400 uppercase font-semibold">Active Exchange Streams</span>
          <div className="text-2xl font-extrabold text-white font-mono mt-1">48 Streams</div>
          <span className="text-[11px] text-emerald-400 font-semibold">Binance, Coinbase, Kraken, OKX</span>
        </div>

        <div className="glass-panel p-4 rounded-xl border border-[#1e2330]">
          <span className="text-xs text-gray-400 uppercase font-semibold">Processing Throughput</span>
          <div className="text-2xl font-extrabold text-blue-400 font-mono mt-1">1.4 GB/s</div>
          <span className="text-[10px] text-gray-400">Low-latency Memory Buffer</span>
        </div>

        <div className="glass-panel p-4 rounded-xl border border-[#1e2330]">
          <span className="text-xs text-gray-400 uppercase font-semibold">Queue Buffer Loss Rate</span>
          <div className="text-2xl font-extrabold text-emerald-400 font-mono mt-1">0.00%</div>
          <span className="text-[10px] text-gray-400">Zero Packet Loss Protocol</span>
        </div>
      </div>

      {/* Simple Placeholder Visualization */}
      <div className="glass-panel p-6 rounded-xl border border-[#1e2330] space-y-4">
        <h3 className="text-sm font-bold text-white uppercase tracking-wider">Stream Queue & Ingestion Pipeline Buffer Visualizer</h3>
        <div className="h-44 bg-[#0d0f17] rounded-xl border border-[#1e2330] flex items-center justify-center p-6 text-center">
          <div className="space-y-2">
            <div className="flex items-center justify-center gap-3 text-blue-400">
              <Database className="w-8 h-8 animate-pulse" />
              <div className="w-32 h-1 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"></div>
              <Layers className="w-8 h-8 text-purple-400" />
            </div>
            <p className="text-xs text-gray-400 font-mono">
              [Pipeline Stream Simulation Interface Placeholder]
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
