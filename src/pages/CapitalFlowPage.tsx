import React from 'react';
import { CAPITAL_FLOW_STATS, CAPITAL_FLOW_TIMELINE, WHALE_TRANSACTIONS } from '../data/mockData';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid
} from 'recharts';
import { Waves } from 'lucide-react';

export const CapitalFlowPage: React.FC = () => {
  const CustomFlowTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-[#12141c]/95 border border-[#2b3042] p-3 rounded-lg shadow-xl text-xs space-y-1">
          <p className="font-semibold text-gray-300">Time: {label}</p>
          <p className="text-emerald-400 font-mono">BTC Outflow: ${payload[0]?.value}M</p>
          <p className="text-rose-400 font-mono">BTC Inflow: ${payload[1]?.value}M</p>
          <p className="text-indigo-400 font-mono">ETH Net Accumulation: ${payload[2]?.value}M</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6">
      {/* Capital Flow Dual Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* BTC Capital Flow */}
        <div className="glass-panel p-5 rounded-xl border border-amber-500/20 space-y-4">
          <div className="flex items-center justify-between border-b border-[#1e2330] pb-3">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-400"></span>
              <h3 className="text-sm font-bold text-white uppercase tracking-wider">BTC Capital Flow</h3>
            </div>
            <span className="px-2.5 py-1 rounded bg-emerald-500/10 text-emerald-400 text-xs font-bold border border-emerald-500/20">
              Net Outflow (Bullish)
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
              <span className="text-[11px] text-gray-400 uppercase font-medium">Inflow</span>
              <div className="text-base font-extrabold text-rose-400 mt-0.5">{CAPITAL_FLOW_STATS.btc.inflow}</div>
            </div>
            <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
              <span className="text-[11px] text-gray-400 uppercase font-medium">Outflow</span>
              <div className="text-base font-extrabold text-emerald-400 mt-0.5">{CAPITAL_FLOW_STATS.btc.outflow}</div>
            </div>
            <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
              <span className="text-[11px] text-gray-400 uppercase font-medium">Net Flow</span>
              <div className="text-base font-extrabold text-emerald-400 mt-0.5">{CAPITAL_FLOW_STATS.btc.netFlow}</div>
            </div>
            <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
              <span className="text-[11px] text-gray-400 uppercase font-medium">Whale Activity</span>
              <div className="text-base font-extrabold text-amber-400 mt-0.5">{CAPITAL_FLOW_STATS.btc.whaleActivity}</div>
            </div>
          </div>
        </div>

        {/* ETH Capital Flow */}
        <div className="glass-panel p-5 rounded-xl border border-indigo-500/20 space-y-4">
          <div className="flex items-center justify-between border-b border-[#1e2330] pb-3">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-indigo-400"></span>
              <h3 className="text-sm font-bold text-white uppercase tracking-wider">ETH Capital Flow</h3>
            </div>
            <span className="px-2.5 py-1 rounded bg-emerald-500/10 text-emerald-400 text-xs font-bold border border-emerald-500/20">
              Net Outflow (Bullish)
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
              <span className="text-[11px] text-gray-400 uppercase font-medium">Inflow</span>
              <div className="text-base font-extrabold text-rose-400 mt-0.5">{CAPITAL_FLOW_STATS.eth.inflow}</div>
            </div>
            <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
              <span className="text-[11px] text-gray-400 uppercase font-medium">Outflow</span>
              <div className="text-base font-extrabold text-emerald-400 mt-0.5">{CAPITAL_FLOW_STATS.eth.outflow}</div>
            </div>
            <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
              <span className="text-[11px] text-gray-400 uppercase font-medium">Net Flow</span>
              <div className="text-base font-extrabold text-emerald-400 mt-0.5">{CAPITAL_FLOW_STATS.eth.netFlow}</div>
            </div>
            <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
              <span className="text-[11px] text-gray-400 uppercase font-medium">Whale Activity</span>
              <div className="text-base font-extrabold text-indigo-400 mt-0.5">{CAPITAL_FLOW_STATS.eth.whaleActivity}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Capital Flow Timeline Chart */}
      <div className="glass-panel p-5 rounded-xl border border-[#1e2330]">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">
              Exchange Flow Volume & Net Accumulation Timeline
            </h3>
            <p className="text-xs text-gray-400 mt-0.5">24h aggregate inflow vs outflow ($ millions)</p>
          </div>
          <div className="flex items-center gap-2 text-xs font-mono text-gray-400">
            <span className="w-2.5 h-2.5 rounded bg-emerald-500"></span> Outflow
            <span className="w-2.5 h-2.5 rounded bg-rose-500"></span> Inflow
          </div>
        </div>

        <div style={{ width: '100%', height: 300 }}>
          <ResponsiveContainer>
            <ComposedChart data={CAPITAL_FLOW_TIMELINE} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2330" vertical={false} />
              <XAxis dataKey="time" stroke="#64748b" fontSize={11} tickLine={false} />
              <YAxis stroke="#64748b" fontSize={11} orientation="right" tickFormatter={(val) => `$${val}M`} tickLine={false} />
              <Tooltip content={<CustomFlowTooltip />} />
              <Bar dataKey="btcOut" fill="#10b981" radius={[3, 3, 0, 0]} opacity={0.8} />
              <Bar dataKey="btcIn" fill="#ef4444" radius={[3, 3, 0, 0]} opacity={0.8} />
              <Line type="monotone" dataKey="ethOut" stroke="#8b5cf6" strokeWidth={2.5} dot={{ r: 3 }} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Whale Activity Table */}
      <div className="glass-panel p-5 rounded-xl border border-[#1e2330] space-y-4">
        <div className="flex items-center justify-between border-b border-[#1e2330] pb-3">
          <div className="flex items-center gap-2">
            <div className="p-1.5 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/20">
              <Waves className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white tracking-tight">Whale Activity Timeline</h3>
              <p className="text-xs text-gray-400">Simulated hourly high-value blockchain transfers ($50M+ threshold)</p>
            </div>
          </div>
          <span className="text-xs font-mono text-blue-400 bg-blue-500/10 px-2.5 py-1 rounded border border-blue-500/20">
            Live Simulated Feed
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-[#0d0f17] text-gray-400 border-b border-[#1e2330]">
              <tr>
                <th className="p-3 font-semibold">Timestamp</th>
                <th className="p-3 font-semibold">Asset</th>
                <th className="p-3 font-semibold">Amount</th>
                <th className="p-3 font-semibold">USD Value</th>
                <th className="p-3 font-semibold">Origin & Destination</th>
                <th className="p-3 font-semibold">Flow Type</th>
                <th className="p-3 font-semibold text-right">Market Impact</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1e2330] text-gray-300">
              {WHALE_TRANSACTIONS.map((tx) => (
                <tr key={tx.id} className="hover:bg-[#151824] transition-all">
                  <td className="p-3 font-mono text-gray-400">{tx.timestamp}</td>
                  <td className="p-3 font-bold text-white">
                    <span className={`px-2 py-0.5 rounded text-[11px] font-mono ${
                      tx.asset === 'BTC' ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20' : 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/20'
                    }`}>
                      {tx.asset}
                    </span>
                  </td>
                  <td className="p-3 font-mono font-bold text-white">{tx.amount}</td>
                  <td className="p-3 font-mono text-emerald-400 font-bold">{tx.valueUsd}</td>
                  <td className="p-3 text-gray-300 font-medium">
                    <div className="flex items-center gap-1">
                      <span>{tx.from}</span>
                      <span className="text-gray-400">→</span>
                      <span className="text-white font-semibold">{tx.to}</span>
                    </div>
                  </td>
                  <td className="p-3">
                    <span className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
                      tx.type === 'Outflow' ? 'bg-emerald-500/10 text-emerald-400' : tx.type === 'Inflow' ? 'bg-rose-500/10 text-rose-400' : 'bg-blue-500/10 text-blue-400'
                    }`}>
                      {tx.type}
                    </span>
                  </td>
                  <td className="p-3 text-right">
                    <span className="px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 font-bold text-[11px]">
                      {tx.impact} Impact
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
