import React from 'react';
import { CorrelationHeatmap } from '../components/CorrelationHeatmap';
import { RELATIONSHIP_INSIGHTS } from '../data/mockData';
import { GitCompare, ShieldAlert, ArrowUpRight, ArrowDownRight } from 'lucide-react';

export const CrossMarketPage: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Intro Subtitle Bar */}
      <div className="glass-panel p-5 rounded-xl border border-[#1e2330] flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-gradient-to-r from-[#121522] via-[#12141c] to-[#161226]">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-purple-500/10 text-purple-400 border border-purple-500/20">
            <GitCompare className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white tracking-tight">Cross-Market Relationships Engine</h2>
            <p className="text-xs text-gray-400 mt-0.5">
              Monitoring co-movements between crypto assets, macro fiat metrics, and equity risk indices.
            </p>
          </div>
        </div>

        <div className="p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs flex items-center gap-2">
          <ShieldAlert className="w-4 h-4 shrink-0" />
          <span>Analytical co-movement signals — not causal assertions.</span>
        </div>
      </div>

      <CorrelationHeatmap />

      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold uppercase tracking-wider text-gray-300">
            Key Structural Relationship Insights
          </h3>
          <span className="text-xs text-gray-400 font-mono">XGBoost Feature Co-Variance</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {RELATIONSHIP_INSIGHTS.map((insight) => {
            const isPos = insight.type === 'positive';
            return (
              <div
                key={insight.pair}
                className="glass-panel p-5 rounded-xl border border-[#1e2330] hover:border-[#2f374e] transition-all space-y-3"
              >
                <div className="flex items-center justify-between">
                  <span className="px-2.5 py-1 rounded bg-[#161a26] text-blue-400 font-mono text-xs font-bold border border-[#232a3d]">
                    {insight.pair}
                  </span>
                  <div className={`flex items-center gap-1 font-mono text-xs font-extrabold px-2.5 py-1 rounded ${
                    isPos ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                  }`}>
                    {isPos ? <ArrowUpRight className="w-3.5 h-3.5" /> : <ArrowDownRight className="w-3.5 h-3.5" />}
                    Score: {insight.value}
                  </div>
                </div>

                <h4 className="text-sm font-bold text-white tracking-tight">{insight.title}</h4>
                <p className="text-xs text-gray-300 leading-relaxed">{insight.description}</p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
