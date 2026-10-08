import React from 'react';
import type { HistoricalEvidenceItem } from '../types';
import { X, ShieldAlert, Cpu, GitBranch, ArrowUpRight, ArrowDownRight } from 'lucide-react';
import { RAG_CURRENT_SITUATION } from '../data/mockData';

interface HistoricalEvidenceModalProps {
  item: HistoricalEvidenceItem | null;
  onClose: () => void;
}

export const HistoricalEvidenceModal: React.FC<HistoricalEvidenceModalProps> = ({ item, onClose }) => {
  if (!item) return null;

  const isPositive = item.outcomeReturn >= 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-[#0e1017] border border-[#232838] rounded-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto shadow-2xl shadow-blue-500/10 p-6 space-y-6">
        {/* Header */}
        <div className="flex items-start justify-between border-b border-[#1e2330] pb-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="p-1.5 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/20">
                <GitBranch className="w-5 h-5" />
              </span>
              <div>
                <h3 className="text-lg font-bold text-white tracking-tight">
                  RAG Vector Evidence Analysis — {item.date}
                </h3>
                <p className="text-xs text-gray-400">
                  Retrieval-Augmented Generation historical market match
                </p>
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg bg-[#181b26] text-gray-400 hover:text-white hover:bg-[#252a3b] transition-all"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Similarity Score Banner */}
        <div className="p-4 rounded-xl bg-[#121522] border border-[#23293c] flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="text-xs text-gray-400 font-medium">Vector Cosine Similarity</div>
            <div className="text-3xl font-extrabold text-blue-400 font-mono mt-0.5">{item.similarity}% Match</div>
          </div>
          <div className="text-right">
            <div className="text-xs text-gray-400 font-medium">Historical Outcome (+4H Horizon)</div>
            <div className={`text-xl font-bold font-mono mt-0.5 flex items-center justify-end gap-1 ${isPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
              {isPositive ? <ArrowUpRight className="w-5 h-5" /> : <ArrowDownRight className="w-5 h-5" />}
              {item.outcome}
            </div>
          </div>
        </div>

        {/* Situation Comparison Table */}
        <div className="space-y-3">
          <h4 className="text-xs font-bold uppercase tracking-wider text-gray-400">
            Feature Vector Match Breakdown
          </h4>
          <div className="rounded-xl border border-[#1e2330] overflow-hidden bg-[#0a0c12]">
            <table className="w-full text-xs text-left">
              <thead className="bg-[#121522] text-gray-400 border-b border-[#1e2330]">
                <tr>
                  <th className="p-3 font-semibold">Feature Dimension</th>
                  <th className="p-3 font-semibold">Current State</th>
                  <th className="p-3 font-semibold">Historical Match ({item.date})</th>
                  <th className="p-3 font-semibold text-right">Alignment</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1e2330] text-gray-300">
                <tr>
                  <td className="p-3 font-medium text-white">BTC Momentum</td>
                  <td className="p-3 font-mono text-emerald-400">{RAG_CURRENT_SITUATION.btcMomentum}</td>
                  <td className="p-3 font-mono text-emerald-400">{item.conditions.btcMomentum}</td>
                  <td className="p-3 text-right text-emerald-400 font-bold">96%</td>
                </tr>
                <tr>
                  <td className="p-3 font-medium text-white">DXY Return</td>
                  <td className="p-3 font-mono text-rose-400">{RAG_CURRENT_SITUATION.dxy}</td>
                  <td className="p-3 font-mono text-rose-400">{item.conditions.dxy}</td>
                  <td className="p-3 text-right text-emerald-400 font-bold">92%</td>
                </tr>
                <tr>
                  <td className="p-3 font-medium text-white">S&P 500 Direction</td>
                  <td className="p-3 font-mono text-emerald-400">{RAG_CURRENT_SITUATION.sp500}</td>
                  <td className="p-3 font-mono text-emerald-400">{item.conditions.sp500}</td>
                  <td className="p-3 text-right text-emerald-400 font-bold">89%</td>
                </tr>
                <tr>
                  <td className="p-3 font-medium text-white">News Sentiment</td>
                  <td className="p-3 font-mono text-blue-400">{RAG_CURRENT_SITUATION.newsSentiment}</td>
                  <td className="p-3 font-mono text-blue-400">{item.conditions.newsSentiment}</td>
                  <td className="p-3 text-right text-emerald-400 font-bold">94%</td>
                </tr>
                <tr>
                  <td className="p-3 font-medium text-white">Fear & Greed Index</td>
                  <td className="p-3 font-mono text-amber-400">{RAG_CURRENT_SITUATION.fearGreed} / 100</td>
                  <td className="p-3 font-mono text-amber-400">{item.conditions.fearGreed} / 100</td>
                  <td className="p-3 text-right text-emerald-400 font-bold">91%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Narrative Synthesis */}
        <div className="p-4 rounded-xl bg-[#121520] border border-[#1e2330] space-y-2">
          <div className="flex items-center gap-2 text-xs font-bold text-blue-400">
            <Cpu className="w-4 h-4" />
            <span>LLM Synthesis & Historical Context Note</span>
          </div>
          <p className="text-xs text-gray-300 leading-relaxed">{item.notes}</p>
          <div className="text-[11px] text-gray-400 font-mono pt-1">
            Embedding Euclidean distance: d = {item.vectorDistance.toFixed(3)}
          </div>
        </div>

        {/* Mandatory Disclaimer */}
        <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-start gap-3">
          <ShieldAlert className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
          <p className="text-xs text-amber-300/90 leading-normal font-medium">
            <strong>Important Research Notice:</strong> Historical similarity does not guarantee future performance.
            This RAG module retrieves high-dimensional vector representations to assist contextual evaluation. Simulated frontend output.
          </p>
        </div>

        {/* Modal Action Footer */}
        <div className="flex justify-end pt-2">
          <button
            onClick={onClose}
            className="px-5 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-bold transition-all shadow-md"
          >
            Close Evidence View
          </button>
        </div>
      </div>
    </div>
  );
};
