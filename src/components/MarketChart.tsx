import React from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine
} from 'recharts';
import type { PricePoint } from '../types';

interface MarketChartProps {
  data: PricePoint[];
  title?: string;
  asset: string;
  showForecast?: boolean;
  forecastData?: { time: string; price: number; predicted: number; lowerBound: number; upperBound: number }[];
  height?: number;
}

export const MarketChart: React.FC<MarketChartProps> = ({
  data,
  title,
  asset,
  showForecast = false,
  forecastData = [],
  height = 320
}) => {
  const chartData = React.useMemo(() => {
    if (!showForecast || forecastData.length === 0) {
      return data;
    }
    const historicalFormatted = data.map(d => ({
      ...d,
      type: 'historical'
    }));
    const forecastFormatted = forecastData.map(f => ({
      time: f.time,
      price: f.price,
      predicted: f.predicted,
      lowerBound: f.lowerBound,
      upperBound: f.upperBound,
      volume: Math.floor(Math.random() * 2000 + 1000),
      type: 'forecast'
    }));

    return [...historicalFormatted, ...forecastFormatted];
  }, [data, showForecast, forecastData]);

  const minPrice = Math.min(...chartData.map(d => d.lowerBound || d.price || d.predicted || 0)) * 0.995;
  const maxPrice = Math.max(...chartData.map(d => d.upperBound || d.price || d.predicted || 0)) * 1.005;

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const point = payload[0].payload;
      return (
        <div className="bg-[#12141c]/95 border border-[#2b3042] p-3 rounded-lg shadow-xl backdrop-blur-md text-xs space-y-1">
          <p className="font-semibold text-gray-300 flex items-center justify-between gap-4">
            <span>Time: {label}</span>
            <span className="text-[10px] text-gray-400 font-mono">{point.type === 'forecast' ? 'PREDICTED' : 'HISTORICAL'}</span>
          </p>
          <div className="text-sm font-bold text-white">
            {asset === 'BTC' ? '$' : '$'}{(point.predicted || point.price).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </div>
          {point.upperBound && (
            <div className="text-[11px] text-purple-400">
              Confidence Range: ${point.lowerBound?.toLocaleString()} - ${point.upperBound?.toLocaleString()}
            </div>
          )}
          <div className="text-[11px] text-gray-400">
            Est. Vol: {point.volume?.toLocaleString()} units
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="glass-panel p-5 rounded-xl border border-[#1e2330]">
      {title && (
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-bold text-gray-200 uppercase tracking-wider">{title}</h3>
          <span className="text-xs text-gray-400 font-mono">Asset: {asset}</span>
        </div>
      )}

      <div style={{ width: '100%', height }}>
        <ResponsiveContainer>
          <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.0} />
              </linearGradient>
              <linearGradient id="forecastGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2330" vertical={false} />
            <XAxis dataKey="time" stroke="#64748b" fontSize={11} tickLine={false} />
            <YAxis
              yAxisId="price"
              domain={[minPrice, maxPrice]}
              stroke="#64748b"
              fontSize={11}
              orientation="right"
              tickFormatter={(val) => `$${val.toLocaleString()}`}
              tickLine={false}
            />
            <YAxis yAxisId="volume" domain={[0, 'dataMax * 3']} hide />
            <Tooltip content={<CustomTooltip />} />

            <Bar yAxisId="volume" dataKey="volume" fill="#252a3b" radius={[2, 2, 0, 0]} opacity={0.6} />

            <Area
              yAxisId="price"
              type="monotone"
              dataKey="price"
              stroke="#3b82f6"
              strokeWidth={2}
              fillOpacity={1}
              fill="url(#priceGradient)"
            />

            {showForecast && (
              <>
                <Area
                  yAxisId="price"
                  type="monotone"
                  dataKey="upperBound"
                  stroke="transparent"
                  fill="url(#forecastGradient)"
                />
                <Area
                  yAxisId="price"
                  type="monotone"
                  dataKey="predicted"
                  stroke="#8b5cf6"
                  strokeWidth={2}
                  strokeDasharray="4 4"
                  fill="none"
                />
                {data.length > 0 && (
                  <ReferenceLine
                    yAxisId="price"
                    x={data[data.length - 1].time}
                    stroke="#8b5cf6"
                    strokeDasharray="3 3"
                    label={{ value: 'FORECAST START', fill: '#8b5cf6', fontSize: 10, position: 'top' }}
                  />
                )}
              </>
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
