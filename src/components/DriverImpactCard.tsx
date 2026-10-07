import React from 'react';
import type { DriverImpactData } from '../types';
import { ArrowUpRight, ArrowDownRight, Minus } from 'lucide-react';

interface DriverImpactCardProps {
  drivers: DriverImpactData[];
  title?: string;
  assetName: string;
}

export const DriverImpactCard: React.FC<DriverImpactCardProps> = ({
  drivers,
  title = "Cross-Market Drivers",
  assetName
}) => {
  return (
    <div className="glass-panel p-5 rounded-xl border border-[#1e2330] space-y-4">
      <div className="flex items-center justify-between border-b border-[#1e2330] pb-3">
        <div>
          <h3 className="text-sm font-bold text-white uppercase tracking-wider">{title}</h3>
          <p className="text-xs text-gray-400 mt-0.5">Factor weighting impact scores for {assetName}</p>
        </div>
        <span className="text-[11px] font-mono text-gray-400 bg-[#0d0f17] px-2.5 py-1 rounded border border-[#1e2330]">
          Normalized Scale [-100 to +100]
        </span>
      </div>

      <div className="space-y-3.5">
        {drivers.map((driver) => {
          const isPositive = driver.impactScore > 0;
          const absVal = Math.min(Math.abs(driver.impactScore), 100);
          const barWidthPercent = (absVal / 50) * 100;

          return (
            <div key={driver.name} className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330] hover:border-[#2b3145] transition-all space-y-2">
              <div className="flex items-center justify-between text-xs font-medium">
                <div className="flex items-center gap-2">
                  {driver.type === 'bullish' ? (
                    <span className="p-1 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                      <ArrowUpRight className="w-3.5 h-3.5" />
                    </span>
                  ) : driver.type === 'bearish' ? (
                    <span className="p-1 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20">
                      <ArrowDownRight className="w-3.5 h-3.5" />
                    </span>
                  ) : (
                    <span className="p-1 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
                      <Minus className="w-3.5 h-3.5" />
                    </span>
                  )}
                  <span className="font-bold text-gray-200">{driver.name}</span>
                </div>

                <div className="flex items-center gap-2 font-mono">
                  <span className="text-gray-400 text-[11px]">{driver.label}</span>
                  <span
                    className={`font-bold px-2 py-0.5 rounded text-xs ${
                      isPositive
                        ? 'text-emerald-400 bg-emerald-500/10'
                        : driver.impactScore === 0
                        ? 'text-amber-400 bg-amber-500/10'
                        : 'text-rose-400 bg-rose-500/10'
                    }`}
                  >
                    {isPositive ? `+${driver.impactScore}` : driver.impactScore}
                  </span>
                </div>
              </div>

              {/* Impact Bar */}
              <div className="relative h-2 w-full bg-[#181b26] rounded-full overflow-hidden flex items-center">
                <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-gray-600 z-10"></div>
                {!isPositive && (
                  <div
                    className="absolute right-1/2 h-full bg-gradient-to-l from-rose-500 to-rose-700 rounded-l-full"
                    style={{ width: `${Math.min(barWidthPercent / 2, 50)}%` }}
                  ></div>
                )}
                {isPositive && (
                  <div
                    className="absolute left-1/2 h-full bg-gradient-to-r from-emerald-500 to-emerald-400 rounded-r-full"
                    style={{ width: `${Math.min(barWidthPercent / 2, 50)}%` }}
                  ></div>
                )}
              </div>

              {driver.description && (
                <p className="text-[11px] text-gray-400 pt-0.5">{driver.description}</p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
