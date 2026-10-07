import React from 'react';
import type { AssetType, ForecastHorizon } from '../types';
import { MarketChart } from '../components/MarketChart';
import { FORECAST_BTC, FORECAST_ETH, getPriceData } from '../data/mockData';
import { Clock, Sparkles, ArrowUpRight } from 'lucide-react';

interface PredictionPageProps {
  selectedAsset: AssetType;
  forecastHorizon: ForecastHorizon;
}

export const PredictionPage: React.FC<PredictionPageProps> = ({
  selectedAsset
}) => {
  const assetName = selectedAsset === 'ETH' ? 'ETH' : 'BTC';
  const priceData = getPriceData(assetName, '1D');
  const forecastItems = assetName === 'BTC' ? FORECAST_BTC : FORECAST_ETH;

  const basePrice = assetName === 'BTC' ? 67842.31 : 3842.72;
  const forecastPoints = forecastItems.map((item, idx) => {
    const mult = 1 + parseFloat(item.expectedReturn.replace('+', '').replace('%', '')) / 100;
    const predicted = basePrice * mult;
    const spread = predicted * (0.004 * (idx + 1));
    return {
      time: item.horizon,
      price: predicted,
      predicted: parseFloat(predicted.toFixed(2)),
      lowerBound: parseFloat((predicted - spread).toFixed(2)),
      upperBound: parseFloat((predicted + spread).toFixed(2))
    };
  });

  return (
    <div className="space-y-6">
      {/* Prominent Simulated Disclaimer Banner */}
      <div className="p-4 rounded-xl bg-purple-500/10 border border-purple-500/30 flex items-start sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-purple-500/20 text-purple-400 border border-purple-500/30">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white tracking-tight">
              LSTM + XGBoost Multi-Step Short-Horizon Model Forecast
            </h3>
            <p className="text-xs text-purple-300/90 mt-0.5">
              Simulated machine learning output for 1-4 hour lookahead horizon
            </p>
          </div>
        </div>

        <div className="px-3 py-1.5 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-400 font-mono text-xs font-bold whitespace-nowrap">
          Prototype prediction — simulated output
        </div>
      </div>

      {/* Main Forecast Chart */}
      <MarketChart
        data={priceData}
        title={`${assetName} Historical Trajectory → 4-Hour Model Prediction Band`}
        asset={assetName}
        showForecast={true}
        forecastData={forecastPoints}
        height={360}
      />

      {/* 4 Forecast Cards Grid */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold uppercase tracking-wider text-gray-200">
            {assetName} Multi-Horizon Step Forecasts
          </h3>
          <span className="text-xs text-gray-400 font-mono">LSTM Sequence Horizon</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {forecastItems.map((item) => {
            const isBullish = item.direction.includes('Bullish');
            return (
              <div
                key={item.horizon}
                className="glass-panel p-5 rounded-xl border border-[#1e2330] hover:border-purple-500/40 transition-all flex flex-col justify-between space-y-4"
              >
                <div>
                  <div className="flex items-center justify-between border-b border-[#1e2330] pb-2.5 mb-3">
                    <span className="text-xs font-bold text-purple-400 font-mono flex items-center gap-1">
                      <Clock className="w-3.5 h-3.5" />
                      {item.horizon}
                    </span>
                    <span
                      className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                        isBullish
                          ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                          : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                      }`}
                    >
                      {item.direction}
                    </span>
                  </div>

                  <div className="space-y-1">
                    <div className="text-xs text-gray-400 font-medium">Expected Return</div>
                    <div className="text-2xl font-extrabold text-emerald-400 font-mono flex items-center gap-1">
                      <ArrowUpRight className="w-5 h-5" />
                      {item.expectedReturn}
                    </div>
                  </div>

                  <div className="mt-3 pt-3 border-t border-[#1a1d29] space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">Target Price:</span>
                      <span className="font-bold text-white font-mono">${item.targetPrice.toLocaleString()}</span>
                    </div>
                    <div>
                      <div className="flex justify-between text-[11px] text-gray-400 mb-1">
                        <span>Confidence:</span>
                        <span className="font-mono text-purple-300 font-bold">{item.confidence}%</span>
                      </div>
                      <div className="w-full bg-[#181b26] h-1.5 rounded-full overflow-hidden">
                        <div
                          className="bg-purple-400 h-full rounded-full"
                          style={{ width: `${item.confidence}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>

                <p className="text-[11px] text-gray-400 leading-snug pt-1 border-t border-[#1a1d29]">
                  {item.description}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
