import React, { useState } from 'react';
import { CORRELATION_MATRIX, CROSS_MARKET_ASSETS } from '../data/mockData';
import { Info } from 'lucide-react';

interface CorrelationHeatmapProps {
  onSelectPair?: (asset1: string, asset2: string, val: number) => void;
}

export const CorrelationHeatmap: React.FC<CorrelationHeatmapProps> = ({ onSelectPair }) => {
  const [hoveredCell, setHoveredCell] = useState<{ r: string; c: string; val: number } | null>(null);
  const [selectedCell, setSelectedCell] = useState<{ r: string; c: string; val: number }>({
    r: 'BTC',
    c: 'DXY',
    val: CORRELATION_MATRIX['BTC']['DXY']
  });

  const getBgColor = (val: number) => {
    if (val === 1.0) return 'bg-[#1e2330] text-gray-400';
    if (val > 0.6) return 'bg-emerald-600/30 text-emerald-300 border border-emerald-500/30 font-bold';
    if (val > 0.2) return 'bg-emerald-800/20 text-emerald-400 border border-emerald-500/20';
    if (val >= -0.2 && val <= 0.2) return 'bg-[#161922] text-gray-400 border border-[#252a3a]';
    if (val < -0.6) return 'bg-rose-600/30 text-rose-300 border border-rose-500/30 font-bold';
    return 'bg-rose-800/20 text-rose-400 border border-rose-500/20';
  };

  const handleCellClick = (r: string, c: string, val: number) => {
    setSelectedCell({ r, c, val });
    if (onSelectPair) {
      onSelectPair(r, c, val);
    }
  };

  return (
    <div className="glass-panel p-6 rounded-xl border border-[#1e2330]">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6">
        <div>
          <h3 className="text-base font-bold text-white tracking-tight">Cross-Market Correlation Heatmap</h3>
          <p className="text-xs text-gray-400 mt-0.5">
            Short-horizon matrix showing statistical co-movement across asset classes
          </p>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-3 text-[11px] font-medium text-gray-400 bg-[#0d0f17] px-3 py-1.5 rounded-lg border border-[#1e2330]">
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded bg-rose-500"></span> -1.0 Inverse
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded bg-gray-600"></span> 0.0 Uncorrelated
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded bg-emerald-500"></span> +1.0 Positive
          </span>
        </div>
      </div>

      {/* Grid Container */}
      <div className="overflow-x-auto">
        <div className="min-w-[650px]">
          {/* Header Row */}
          <div className="grid grid-cols-9 gap-1.5 mb-1.5 text-center">
            <div className="p-2 text-xs font-semibold text-gray-400 text-left">Asset</div>
            {CROSS_MARKET_ASSETS.map((asset) => (
              <div key={asset} className="p-2 text-xs font-bold text-gray-300 bg-[#121520] rounded-md border border-[#1e2330]">
                {asset}
              </div>
            ))}
          </div>

          {/* Matrix Rows */}
          {CROSS_MARKET_ASSETS.map((rowAsset) => (
            <div key={rowAsset} className="grid grid-cols-9 gap-1.5 mb-1.5 items-center">
              {/* Row Label */}
              <div className="p-2 text-xs font-bold text-gray-300 bg-[#121520] rounded-md border border-[#1e2330]">
                {rowAsset}
              </div>

              {/* Row Cells */}
              {CROSS_MARKET_ASSETS.map((colAsset) => {
                const val = CORRELATION_MATRIX[rowAsset][colAsset];
                const isHovered = hoveredCell?.r === rowAsset && hoveredCell?.c === colAsset;
                const isSelected = selectedCell.r === rowAsset && selectedCell.c === colAsset;

                return (
                  <button
                    key={`${rowAsset}-${colAsset}`}
                    onMouseEnter={() => setHoveredCell({ r: rowAsset, c: colAsset, val })}
                    onMouseLeave={() => setHoveredCell(null)}
                    onClick={() => handleCellClick(rowAsset, colAsset, val)}
                    className={`p-2.5 rounded-md text-xs font-mono text-center transition-all cursor-pointer heatmap-cell relative ${getBgColor(val)} ${
                      isSelected ? 'ring-2 ring-blue-500 ring-offset-2 ring-offset-[#0d0f17] scale-105 z-10' : ''
                    } ${isHovered ? 'scale-105 z-10 brightness-125' : ''}`}
                  >
                    {val > 0 && val !== 1.0 ? `+${val.toFixed(2)}` : val.toFixed(2)}
                  </button>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Active Cell Detail Summary */}
      <div className="mt-5 p-4 rounded-lg bg-[#0d0f17] border border-[#1e2330] flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/20">
            <Info className="w-5 h-5" />
          </div>
          <div>
            <div className="text-xs font-bold text-white flex items-center gap-2">
              <span>Selected Pair:</span>
              <span className="px-2 py-0.5 rounded bg-[#1c202d] text-blue-400 font-mono text-xs">
                {selectedCell.r} ↔ {selectedCell.c}
              </span>
              <span className="font-mono text-xs text-gray-300">
                Score: {selectedCell.val > 0 ? `+${selectedCell.val}` : selectedCell.val}
              </span>
            </div>
            <p className="text-xs text-gray-400 mt-1">
              {selectedCell.r === selectedCell.c
                ? "Self correlation identity matrix standard value (1.00)."
                : selectedCell.val < 0
                ? `Negative co-movement: Increases in ${selectedCell.r} tend to coincide with short-horizon pullbacks in ${selectedCell.c}.`
                : `Positive co-movement: ${selectedCell.r} and ${selectedCell.c} exhibit directional alignment over recent test windows.`}
            </p>
          </div>
        </div>

        <div className="text-right text-[11px] text-gray-400 font-mono whitespace-nowrap bg-[#121520] p-2.5 rounded-lg border border-[#1e2330]">
          Analytical Relationship • Non-Causal
        </div>
      </div>
    </div>
  );
};
