import React, { useState } from 'react';
import { XGBOOST_SHAP_FEATURES, SENTIMENT_OVERALL, getPriceData } from '../data/mockData';
import { MarketChart } from '../components/MarketChart';
import { BrainCircuit } from 'lucide-react';

export const ExplainabilityPage: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<'XGBoost' | 'LSTM' | 'FinBERT'>('XGBoost');
  const priceData = getPriceData('BTC', '1D');

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[#1e2330] pb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-purple-500/10 text-purple-400 border border-purple-500/20">
            <BrainCircuit className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white tracking-tight">Model Explainability & SHAP Decomposition</h2>
            <p className="text-xs text-gray-400 mt-0.5">Understand feature contributions and predictive neural weights</p>
          </div>
        </div>

        <div className="flex items-center bg-[#12141c] border border-[#1f2330] rounded-lg p-1 text-xs">
          {(['XGBoost', 'LSTM', 'FinBERT'] as const).map((model) => (
            <button
              key={model}
              onClick={() => setSelectedModel(model)}
              className={`px-4 py-2 rounded-md font-bold transition-all ${
                selectedModel === model
                  ? 'bg-purple-600 text-white shadow-sm'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              {model}
            </button>
          ))}
        </div>
      </div>

      <div className="glass-panel p-5 rounded-xl border border-[#1e2330] space-y-3">
        <h3 className="text-xs font-bold uppercase tracking-wider text-gray-300">
          System Architecture & Subsystem Mapping
        </h3>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 text-center text-xs">
          <div className={`p-3 rounded-lg border transition-all ${selectedModel === 'XGBoost' ? 'bg-purple-500/10 border-purple-500/50 text-white' : 'bg-[#0d0f17] border-[#1e2330] text-gray-400'}`}>
            <span className="font-bold block text-purple-400">XGBoost</span>
            <span className="text-[10px] mt-1 block">Cross-Market Non-linear Analysis</span>
          </div>

          <div className={`p-3 rounded-lg border transition-all ${selectedModel === 'LSTM' ? 'bg-purple-500/10 border-purple-500/50 text-white' : 'bg-[#0d0f17] border-[#1e2330] text-gray-400'}`}>
            <span className="font-bold block text-blue-400">LSTM</span>
            <span className="text-[10px] mt-1 block">1-4h Sequential Forecast</span>
          </div>

          <div className={`p-3 rounded-lg border transition-all ${selectedModel === 'FinBERT' ? 'bg-purple-500/10 border-purple-500/50 text-white' : 'bg-[#0d0f17] border-[#1e2330] text-gray-400'}`}>
            <span className="font-bold block text-emerald-400">FinBERT</span>
            <span className="text-[10px] mt-1 block">Financial Sentiment NLP</span>
          </div>

          <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330] text-gray-400">
            <span className="font-bold block text-amber-400">SHAP</span>
            <span className="text-[10px] mt-1 block">Feature Attribution</span>
          </div>

          <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330] text-gray-400">
            <span className="font-bold block text-indigo-400">RAG</span>
            <span className="text-[10px] mt-1 block">Vector Retrieval</span>
          </div>

          <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330] text-gray-400">
            <span className="font-bold block text-rose-400">LLM Reasoning</span>
            <span className="text-[10px] mt-1 block">Intelligence Synthesis</span>
          </div>
        </div>
      </div>

      {selectedModel === 'XGBoost' && (
        <div className="space-y-6">
          <div className="glass-panel p-6 rounded-xl border border-[#1e2330] space-y-4">
            <div className="flex items-center justify-between border-b border-[#1e2330] pb-3">
              <div>
                <h3 className="text-base font-bold text-white tracking-tight">
                  Feature Importance / SHAP (Shapley Additive exPlanations)
                </h3>
                <p className="text-xs text-gray-400 mt-0.5">
                  SHAP values represent the marginal contribution of features to the final ensemble prediction.
                </p>
              </div>
              <span className="text-xs font-mono text-purple-400 bg-purple-500/10 px-2.5 py-1 rounded border border-purple-500/20">
                XGBoost Model Decomposition
              </span>
            </div>

            <div className="space-y-4 pt-2">
              {XGBOOST_SHAP_FEATURES.map((item) => {
                const isPos = item.importance > 0;
                const widthPercent = (Math.abs(item.importance) / 0.35) * 100;
                return (
                  <div key={item.feature} className="p-3.5 rounded-lg bg-[#0d0f17] border border-[#1e2330] space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-bold text-gray-200">{item.feature}</span>
                      <div className="flex items-center gap-2 font-mono">
                        <span className="text-gray-400 text-[11px] uppercase">Category: {item.category}</span>
                        <span className={`font-bold px-2 py-0.5 rounded ${isPos ? 'text-emerald-400 bg-emerald-500/10' : 'text-rose-400 bg-rose-500/10'}`}>
                          {isPos ? `+${item.importance}` : item.importance}
                        </span>
                      </div>
                    </div>

                    <div className="relative h-2.5 w-full bg-[#181b26] rounded-full overflow-hidden flex items-center">
                      <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-gray-600 z-10"></div>
                      {isPos ? (
                        <div
                          className="absolute left-1/2 h-full bg-gradient-to-r from-emerald-500 to-emerald-400 rounded-r-full"
                          style={{ width: `${widthPercent / 2}%` }}
                        ></div>
                      ) : (
                        <div
                          className="absolute right-1/2 h-full bg-gradient-to-l from-rose-500 to-rose-700 rounded-l-full"
                          style={{ width: `${widthPercent / 2}%` }}
                        ></div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="p-4 rounded-xl bg-[#121520] border border-[#1e2330] text-xs text-gray-300">
              <strong>SHAP Methodology Note:</strong> Positive values push the short-term prediction toward a higher return target, whereas negative values pull the prediction downward.
            </div>
          </div>
        </div>
      )}

      {selectedModel === 'LSTM' && (
        <div className="space-y-6">
          <div className="glass-panel p-6 rounded-xl border border-[#1e2330] space-y-4">
            <div className="flex items-center justify-between border-b border-[#1e2330] pb-3">
              <div>
                <h3 className="text-base font-bold text-white tracking-tight">Temporal Prediction & Recurrent Sequence State</h3>
                <p className="text-xs text-gray-400 mt-0.5">LSTM network processing past lookback window to forecast next 4 hours</p>
              </div>
              <span className="px-3 py-1 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-bold font-mono text-xs">
                Prediction Confidence: 78%
              </span>
            </div>

            <MarketChart
              data={priceData}
              title="LSTM Lookback Sequence → Predicted 4-Hour Trajectory"
              asset="BTC"
              showForecast={true}
              forecastData={[
                { time: '+1h', price: 68127, predicted: 68127.25, lowerBound: 67900, upperBound: 68350 },
                { time: '+2h', price: 68296, predicted: 68296.88, lowerBound: 68000, upperBound: 68550 },
                { time: '+3h', price: 68405, predicted: 68405.51, lowerBound: 68050, upperBound: 68700 },
                { time: '+4h', price: 68459, predicted: 68459.76, lowerBound: 68000, upperBound: 68900 }
              ]}
              height={320}
            />

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
              <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
                <span className="text-gray-400 uppercase font-medium text-[11px]">Sequence Window</span>
                <div className="text-sm font-bold text-white mt-0.5 font-mono">168 Timesteps (1 Week 1h candles)</div>
              </div>
              <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
                <span className="text-gray-400 uppercase font-medium text-[11px]">Attention Weight</span>
                <div className="text-sm font-bold text-purple-400 mt-0.5 font-mono">Concentrated on recent 6h</div>
              </div>
              <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
                <span className="text-gray-400 uppercase font-medium text-[11px]">Hidden State Latent Dim</span>
                <div className="text-sm font-bold text-blue-400 mt-0.5 font-mono">128 Neurons (Bi-LSTM)</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {selectedModel === 'FinBERT' && (
        <div className="space-y-6">
          <div className="glass-panel p-6 rounded-xl border border-[#1e2330] space-y-5">
            <div className="flex items-center justify-between border-b border-[#1e2330] pb-3">
              <div>
                <h3 className="text-base font-bold text-white tracking-tight">FinBERT Sentiment Vector Decomposition</h3>
                <p className="text-xs text-gray-400 mt-0.5">Domain-adapted BERT transformer evaluating financial news text</p>
              </div>
              <span className="px-3 py-1 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20 font-bold font-mono text-xs">
                Transformer Classification
              </span>
            </div>

            <div className="p-4 rounded-xl bg-[#0d0f17] border border-[#1e2330] grid grid-cols-3 gap-4 text-center">
              <div>
                <span className="text-xs text-gray-400 uppercase font-medium">Positive Score</span>
                <div className="text-2xl font-black text-emerald-400 font-mono mt-1">
                  {(SENTIMENT_OVERALL.finbert.positive * 100).toFixed(0)}%
                </div>
              </div>
              <div>
                <span className="text-xs text-gray-400 uppercase font-medium">Neutral Score</span>
                <div className="text-2xl font-black text-amber-400 font-mono mt-1">
                  {(SENTIMENT_OVERALL.finbert.neutral * 100).toFixed(0)}%
                </div>
              </div>
              <div>
                <span className="text-xs text-gray-400 uppercase font-medium">Negative Score</span>
                <div className="text-2xl font-black text-rose-400 font-mono mt-1">
                  {(SENTIMENT_OVERALL.finbert.negative * 100).toFixed(0)}%
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <h4 className="text-xs font-bold uppercase tracking-wider text-gray-300">
                Sample Token Attention Highlight Visualization
              </h4>
              <div className="p-4 rounded-xl bg-[#12141c] border border-[#1e2330] text-xs leading-relaxed font-mono">
                <span className="text-gray-400">"Bitcoin ETF </span>
                <span className="bg-emerald-500/30 text-emerald-300 px-1 py-0.5 rounded font-bold">inflows (+0.88)</span>
                <span className="text-gray-400"> strengthen market </span>
                <span className="bg-emerald-500/20 text-emerald-300 px-1 py-0.5 rounded">confidence (+0.74)</span>
                <span className="text-gray-400"> as spot volumes </span>
                <span className="bg-blue-500/20 text-blue-300 px-1 py-0.5 rounded">surge (+0.65)</span>
                <span className="text-gray-400"> past </span>
                <span className="text-white font-bold">$4.2B</span>
                <span className="text-gray-400">"</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
