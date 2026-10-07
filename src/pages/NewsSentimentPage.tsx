import React, { useState } from 'react';
import { NEWS_ARTICLES, SENTIMENT_OVERALL } from '../data/mockData';
import type { SentimentFilter } from '../types';
import { BrainCircuit, Filter } from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid } from 'recharts';

export const NewsSentimentPage: React.FC = () => {
  const [activeAssetTab, setActiveAssetTab] = useState<'ALL' | 'BTC' | 'ETH'>('ALL');
  const [sentimentFilter, setSentimentFilter] = useState<SentimentFilter>('ALL');

  const filteredNews = NEWS_ARTICLES.filter((article) => {
    const matchesAsset = activeAssetTab === 'ALL' || article.asset === 'ALL' || article.asset === activeAssetTab;
    const matchesSentiment = sentimentFilter === 'ALL' || article.sentiment === sentimentFilter;
    return matchesAsset && matchesSentiment;
  });

  const timelineData = [
    { time: '00:00', positive: 55, neutral: 30, negative: 15 },
    { time: '04:00', positive: 58, neutral: 28, negative: 14 },
    { time: '08:00', positive: 64, neutral: 22, negative: 14 },
    { time: '12:00', positive: 60, neutral: 25, negative: 15 },
    { time: '16:00', positive: 66, neutral: 21, negative: 13 },
    { time: '20:00', positive: 62, neutral: 24, negative: 14 }
  ];

  return (
    <div className="space-y-6">
      {/* Top Sentiment Summary & FinBERT Output */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Aggregate Sentiment Summary */}
        <div className="glass-panel p-5 rounded-xl border border-[#1e2330] space-y-4">
          <div className="flex items-center justify-between border-b border-[#1e2330] pb-3">
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">Sentiment Summary</h3>
            <span className="text-xs text-emerald-400 font-bold bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
              62% Positive
            </span>
          </div>

          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-xs font-semibold mb-1">
                <span className="text-emerald-400">Positive</span>
                <span className="text-emerald-400 font-mono">{SENTIMENT_OVERALL.positive}%</span>
              </div>
              <div className="w-full bg-[#181b26] h-2 rounded-full overflow-hidden">
                <div className="bg-emerald-400 h-full rounded-full" style={{ width: `${SENTIMENT_OVERALL.positive}%` }}></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs font-semibold mb-1">
                <span className="text-amber-400">Neutral</span>
                <span className="text-amber-400 font-mono">{SENTIMENT_OVERALL.neutral}%</span>
              </div>
              <div className="w-full bg-[#181b26] h-2 rounded-full overflow-hidden">
                <div className="bg-amber-400 h-full rounded-full" style={{ width: `${SENTIMENT_OVERALL.neutral}%` }}></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs font-semibold mb-1">
                <span className="text-rose-400">Negative</span>
                <span className="text-rose-400 font-mono">{SENTIMENT_OVERALL.negative}%</span>
              </div>
              <div className="w-full bg-[#181b26] h-2 rounded-full overflow-hidden">
                <div className="bg-rose-400 h-full rounded-full" style={{ width: `${SENTIMENT_OVERALL.negative}%` }}></div>
              </div>
            </div>
          </div>
        </div>

        {/* FinBERT Output Model Card */}
        <div className="glass-panel p-5 rounded-xl border border-purple-500/30 glow-purple space-y-4 bg-gradient-to-br from-[#121524] to-[#14121c]">
          <div className="flex items-center justify-between border-b border-[#1e2330] pb-3">
            <div className="flex items-center gap-2">
              <BrainCircuit className="w-4 h-4 text-purple-400" />
              <h3 className="text-sm font-bold text-white uppercase tracking-wider">FinBERT Analysis</h3>
            </div>
            <span className="text-[10px] font-mono text-purple-300 bg-purple-500/20 px-2 py-0.5 rounded border border-purple-500/30">
              Mock FinBERT Output
            </span>
          </div>

          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="p-2.5 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
              <span className="text-[10px] text-gray-400 uppercase font-medium">Positive</span>
              <div className="text-lg font-bold text-emerald-400 font-mono mt-0.5">
                {SENTIMENT_OVERALL.finbert.positive.toFixed(2)}
              </div>
            </div>
            <div className="p-2.5 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
              <span className="text-[10px] text-gray-400 uppercase font-medium">Neutral</span>
              <div className="text-lg font-bold text-amber-400 font-mono mt-0.5">
                {SENTIMENT_OVERALL.finbert.neutral.toFixed(2)}
              </div>
            </div>
            <div className="p-2.5 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
              <span className="text-[10px] text-gray-400 uppercase font-medium">Negative</span>
              <div className="text-lg font-bold text-rose-400 font-mono mt-0.5">
                {SENTIMENT_OVERALL.finbert.negative.toFixed(2)}
              </div>
            </div>
          </div>

          <p className="text-[11px] text-gray-400">
            FinBERT domain-specific Transformer model evaluating financial syntax & press release vectors.
          </p>
        </div>

        {/* Sentiment Timeline Chart */}
        <div className="glass-panel p-5 rounded-xl border border-[#1e2330] flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-xs font-bold text-white uppercase tracking-wider">Sentiment Index Dynamics (24h)</h3>
            <span className="text-[10px] text-gray-400 font-mono">Rolling Average</span>
          </div>
          <div style={{ width: '100%', height: 140 }}>
            <ResponsiveContainer>
              <AreaChart data={timelineData} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
                <CartesianGrid strokeDasharray="2 2" stroke="#1f2330" />
                <XAxis dataKey="time" stroke="#64748b" fontSize={10} tickLine={false} />
                <YAxis stroke="#64748b" fontSize={10} domain={[0, 100]} tickLine={false} />
                <Area type="monotone" dataKey="positive" stroke="#10b981" fill="#10b981" fillOpacity={0.2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[#1e2330] pb-4">
        <div className="flex items-center bg-[#12141c] border border-[#1f2330] rounded-lg p-1 text-xs">
          {(['ALL', 'BTC', 'ETH'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveAssetTab(tab)}
              className={`px-4 py-1.5 rounded-md font-bold transition-all ${
                activeAssetTab === tab
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              {tab === 'ALL' ? 'All News' : tab === 'BTC' ? 'Bitcoin' : 'Ethereum'}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400 font-medium flex items-center gap-1">
            <Filter className="w-3.5 h-3.5" />
            Filter Sentiment:
          </span>
          {(['ALL', 'Positive', 'Neutral', 'Negative'] as SentimentFilter[]).map((filter) => (
            <button
              key={filter}
              onClick={() => setSentimentFilter(filter)}
              className={`px-3 py-1 rounded-md text-xs font-semibold transition-all ${
                sentimentFilter === filter
                  ? 'bg-[#252a3b] text-white border border-[#3b4257]'
                  : 'bg-[#0d0f17] text-gray-400 border border-[#1e2330] hover:text-gray-200'
              }`}
            >
              {filter}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {filteredNews.map((article) => {
          const isPos = article.sentiment === 'Positive';
          const isNeg = article.sentiment === 'Negative';
          return (
            <div
              key={article.id}
              className="glass-panel p-5 rounded-xl border border-[#1e2330] hover:border-[#2f374e] transition-all flex flex-col justify-between space-y-4"
            >
              <div>
                <div className="flex items-center justify-between gap-2 mb-2.5">
                  <div className="flex items-center gap-2">
                    <span className="px-2 py-0.5 rounded bg-[#161a26] text-blue-400 font-mono text-[11px] font-bold border border-[#232a3d]">
                      {article.asset}
                    </span>
                    <span className="text-xs text-gray-400">{article.source}</span>
                    <span className="text-[11px] text-gray-400 font-mono">• {article.time}</span>
                  </div>

                  <span
                    className={`px-2.5 py-0.5 rounded text-xs font-bold border ${
                      isPos
                        ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                        : isNeg
                        ? 'bg-rose-500/10 text-rose-400 border-rose-500/20'
                        : 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                    }`}
                  >
                    {article.sentiment}
                  </span>
                </div>

                <h4 className="text-sm font-bold text-white tracking-tight leading-snug">
                  {article.headline}
                </h4>

                <p className="text-xs text-gray-300 mt-2 leading-relaxed">
                  {article.summary}
                </p>
              </div>

              <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330] flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <BrainCircuit className="w-3.5 h-3.5 text-purple-400" />
                  <span className="text-gray-400 font-mono text-[11px]">
                    Pos: {article.finbertScores.positive} | Neu: {article.finbertScores.neutral} | Neg: {article.finbertScores.negative}
                  </span>
                </div>

                <span className="text-[11px] font-bold text-purple-300 bg-purple-500/10 px-2 py-0.5 rounded border border-purple-500/20">
                  Confidence: {article.confidence}%
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
