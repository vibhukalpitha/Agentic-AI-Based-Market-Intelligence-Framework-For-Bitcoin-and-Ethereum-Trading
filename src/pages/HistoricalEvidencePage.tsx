import React, { useState } from 'react';
import { HISTORICAL_EVIDENCE, RAG_CURRENT_SITUATION } from '../data/mockData';
import type { HistoricalEvidenceItem } from '../types';
import { HistoricalEvidenceModal } from '../components/HistoricalEvidenceModal';
import { History, ShieldAlert, ExternalLink, ArrowUpRight, ArrowDownRight } from 'lucide-react';

export const HistoricalEvidencePage: React.FC = () => {
  const [selectedItem, setSelectedItem] = useState<HistoricalEvidenceItem | null>(null);

  return (
    <div className="space-y-6">
      {/* RAG Header Banner */}
      <div className="glass-panel p-6 rounded-xl border border-[#1e2330] bg-gradient-to-r from-[#121522] via-[#12141c] to-[#1a1226] space-y-3">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-blue-500/10 text-blue-400 border border-blue-500/20">
            <History className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white tracking-tight">
              Retrieval-Augmented Generation (RAG) Historical Matcher
            </h2>
            <p className="text-xs text-gray-300 mt-0.5">
              Find historically similar market situations in high-dimensional vector space to support current model decisions.
            </p>
          </div>
        </div>

        <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs flex items-center gap-2">
          <ShieldAlert className="w-4 h-4 shrink-0" />
          <span>Notice: Historical similarity does not guarantee future performance.</span>
        </div>
      </div>

      {/* Current Situation Vector Card */}
      <div className="glass-panel p-5 rounded-xl border border-blue-500/30 space-y-3">
        <div className="flex items-center justify-between border-b border-[#1e2330] pb-3">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider">Current Situation Feature Vector</h3>
          <span className="text-xs font-mono text-blue-400 bg-blue-500/10 px-2.5 py-1 rounded border border-blue-500/20">
            Active Query Vector
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-center">
          <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
            <span className="text-[11px] text-gray-400 uppercase font-medium">BTC Momentum</span>
            <div className="text-base font-extrabold text-emerald-400 font-mono mt-0.5">{RAG_CURRENT_SITUATION.btcMomentum}</div>
          </div>
          <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
            <span className="text-[11px] text-gray-400 uppercase font-medium">DXY Return</span>
            <div className="text-base font-extrabold text-rose-400 font-mono mt-0.5">{RAG_CURRENT_SITUATION.dxy}</div>
          </div>
          <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
            <span className="text-[11px] text-gray-400 uppercase font-medium">S&P 500</span>
            <div className="text-base font-extrabold text-emerald-400 font-mono mt-0.5">{RAG_CURRENT_SITUATION.sp500}</div>
          </div>
          <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
            <span className="text-[11px] text-gray-400 uppercase font-medium">Fear & Greed</span>
            <div className="text-base font-extrabold text-amber-400 font-mono mt-0.5">{RAG_CURRENT_SITUATION.fearGreed}</div>
          </div>
          <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
            <span className="text-[11px] text-gray-400 uppercase font-medium">News Sentiment</span>
            <div className="text-base font-extrabold text-blue-400 font-mono mt-0.5">{RAG_CURRENT_SITUATION.newsSentiment}</div>
          </div>
        </div>
      </div>

      {/* Similar Historical Situations List */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold uppercase tracking-wider text-gray-200">
            Top Historical Match Retrievals
          </h3>
          <span className="text-xs text-gray-400 font-mono">Cosine Vector Ranking</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {HISTORICAL_EVIDENCE.map((item) => {
            const isPos = item.outcomeReturn >= 0;
            return (
              <div
                key={item.id}
                className="glass-panel p-5 rounded-xl border border-[#1e2330] hover:border-blue-500/40 transition-all flex flex-col justify-between space-y-4"
              >
                <div>
                  <div className="flex items-center justify-between border-b border-[#1e2330] pb-3 mb-3">
                    <span className="text-sm font-bold text-white font-mono">{item.date}</span>
                    <span className="px-2.5 py-1 rounded bg-blue-500/10 text-blue-400 font-mono font-bold text-xs border border-blue-500/20">
                      {item.similarity}% Similarity
                    </span>
                  </div>

                  <div className="space-y-2 text-xs">
                    <div className="p-2.5 rounded-lg bg-[#0d0f17] border border-[#1e2330] space-y-1">
                      <div className="text-[11px] text-gray-400 font-medium uppercase">Market Conditions:</div>
                      <div className="text-gray-300 font-mono text-[11px] space-y-0.5">
                        <div>BTC Momentum: {item.conditions.btcMomentum}</div>
                        <div>DXY: {item.conditions.dxy} | S&P: {item.conditions.sp500}</div>
                        <div>News: {item.conditions.newsSentiment}</div>
                      </div>
                    </div>

                    <div className="p-3 rounded-lg bg-[#121520] border border-[#1e2330] flex items-center justify-between">
                      <span className="text-gray-400 font-medium text-[11px]">Historical Outcome:</span>
                      <span
                        className={`font-mono font-bold text-xs flex items-center gap-1 ${
                          isPos ? 'text-emerald-400' : 'text-rose-400'
                        }`}
                      >
                        {isPos ? <ArrowUpRight className="w-3.5 h-3.5" /> : <ArrowDownRight className="w-3.5 h-3.5" />}
                        {item.outcome}
                      </span>
                    </div>
                  </div>
                </div>

                <button
                  onClick={() => setSelectedItem(item)}
                  className="w-full py-2.5 rounded-lg bg-[#181c2b] hover:bg-blue-600 text-blue-400 hover:text-white border border-blue-500/30 text-xs font-bold transition-all flex items-center justify-center gap-2"
                >
                  <span>View Evidence</span>
                  <ExternalLink className="w-3.5 h-3.5" />
                </button>
              </div>
            );
          })}
        </div>
      </div>

      <HistoricalEvidenceModal item={selectedItem} onClose={() => setSelectedItem(null)} />
    </div>
  );
};
