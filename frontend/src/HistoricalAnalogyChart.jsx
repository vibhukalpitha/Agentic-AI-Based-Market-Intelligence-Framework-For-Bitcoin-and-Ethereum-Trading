import React, { useState } from 'react';
import {
  TrendingUp,
  TrendingDown,
  Clock,
  History,
  Target,
  Sparkles,
  Info,
  Calendar,
  Layers
} from 'lucide-react';

export default function HistoricalAnalogyChart({ analogyData, activeCoin = 'BTC' }) {
  const [hoveredPoint, setHoveredPoint] = useState(null);

  if (!analogyData) {
    return (
      <div className="glass-panel" style={{ padding: '24px', textAlign: 'center', color: 'var(--text-dim)' }}>
        <History size={24} style={{ margin: '0 auto 8px', opacity: 0.5 }} />
        <span>Computing historical regime analogies...</span>
      </div>
    );
  }

  const matches = analogyData.historical_matches || [];
  const stats = analogyData.aggregate_stats || {};
  const currentSignal = analogyData.current_signal || {};

  // Colors for the 3 historical trajectories
  const trajectoryColors = [
    { line: '#00F0FF', bg: 'rgba(0, 240, 255, 0.15)', name: 'Match #1' },
    { line: '#A855F7', bg: 'rgba(168, 85, 247, 0.15)', name: 'Match #2' },
    { line: '#F59E0B', bg: 'rgba(245, 158, 11, 0.15)', name: 'Match #3' }
  ];

  // SVG Chart Geometry Calculations
  const chartWidth = 800;
  const chartHeight = 320;
  const padLeft = 60;
  const padRight = 50;
  const padTop = 40;
  const padBottom = 45;

  const innerWidth = chartWidth - padLeft - padRight;
  const innerHeight = chartHeight - padTop - padBottom;

  // Determine min and max Y values across all trajectories
  let allReturns = [0.0];
  matches.forEach(m => {
    if (Array.isArray(m.price_trajectory)) {
      allReturns.push(...m.price_trajectory);
    }
  });

  const minY = Math.floor(Math.min(...allReturns) - 1.0);
  const maxY = Math.ceil(Math.max(...allReturns) + 1.0);
  const yRange = Math.max(2.0, maxY - minY);

  // Coordinate mapping functions
  const getX = (t) => padLeft + (t / 8.0) * innerWidth;
  const getY = (val) => padTop + innerHeight - ((val - minY) / yRange) * innerHeight;

  const zeroY = getY(0.0);
  const x1Hour = getX(1.0);
  const x4Hour = getX(4.0);

  // Generate Y axis tick marks
  const yTicks = [];
  const tickStep = (maxY - minY) > 6 ? 2.0 : 1.0;
  for (let val = minY; val <= maxY; val += tickStep) {
    yTicks.push(val);
  }

  return (
    <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '22px' }}>
      {/* Header & Aggregate Stats Banner */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, rgba(0, 240, 255, 0.2), rgba(168, 85, 247, 0.2))',
            border: '1px solid rgba(0, 240, 255, 0.4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#00F0FF'
          }}>
            <History size={20} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h3 style={{ fontSize: '16px', fontWeight: 800, color: '#fff', letterSpacing: '-0.01em' }}>
                Historical Regime Similarity & Outcome Engine
              </h3>
              <span style={{ fontSize: '11px', padding: '2px 8px', borderRadius: '4px', background: 'rgba(0, 240, 255, 0.1)', color: '#00F0FF', border: '1px solid rgba(0, 240, 255, 0.25)', fontWeight: 700 }}>
                k-NN Vector Retrieval ({activeCoin})
              </span>
            </div>
            <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              When macro factors (S&P 500, DXY, Gold, Sentiment) looked identical in history, what happened to price at 1H and 4H?
            </p>
          </div>
        </div>

        {/* Quant Aggregate Metric Pills */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
          <div style={{ background: 'rgba(0,0,0,0.4)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '6px 14px', display: 'flex', flexDirection: 'column' }}>
            <span style={{ fontSize: '10px', color: 'var(--text-dim)', fontWeight: 700 }}>HISTORICAL WIN RATE</span>
            <span style={{ fontSize: '16px', fontWeight: 800, color: 'var(--emerald-buy)', fontFamily: 'var(--font-mono)' }}>
              {stats.win_rate_for_buy || '75%'}
            </span>
          </div>

          <div style={{ background: 'rgba(0,0,0,0.4)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '6px 14px', display: 'flex', flexDirection: 'column' }}>
            <span style={{ fontSize: '10px', color: 'var(--text-dim)', fontWeight: 700 }}>AVERAGE 4H GAIN</span>
            <span style={{ fontSize: '16px', fontWeight: 800, color: '#00F0FF', fontFamily: 'var(--font-mono)' }}>
              {stats.average_4H_gain || '+2.8%'}
            </span>
          </div>

          <div style={{ background: 'rgba(0,0,0,0.4)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '6px 14px', display: 'flex', flexDirection: 'column' }}>
            <span style={{ fontSize: '10px', color: 'var(--text-dim)', fontWeight: 700 }}>CONSENSUS REGIME</span>
            <span style={{ fontSize: '13px', fontWeight: 700, color: '#f1f5f9', marginTop: '2px' }}>
              {stats.consensus || 'Bullish Continuation'}
            </span>
          </div>
        </div>
      </div>

      {/* Interactive Multi-Trajectory SVG Chart */}
      <div style={{ background: '#0B0E14', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '14px', padding: '16px 12px 10px', position: 'relative', overflow: 'hidden' }}>
        {/* Chart Legend Bar */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0 12px 8px', borderBottom: '1px solid rgba(255,255,255,0.05)', flexWrap: 'wrap', gap: '8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            {matches.map((m, idx) => (
              <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px' }}>
                <span style={{ width: '12px', height: '3px', background: trajectoryColors[idx].line, borderRadius: '2px', boxShadow: `0 0 8px ${trajectoryColors[idx].line}` }}></span>
                <span style={{ color: '#fff', fontWeight: 600 }}>{trajectoryColors[idx].name}:</span>
                <span style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>{(m.similarity_score * 100).toFixed(0)}% Match</span>
              </div>
            ))}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '14px', fontSize: '11px', color: 'var(--text-muted)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <span style={{ width: '10px', borderBottom: '2px dashed #F43F5E' }}></span>
              <span>1H Horizon Target</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <span style={{ width: '10px', borderBottom: '2px dashed #10B981' }}></span>
              <span>4H Horizon Target</span>
            </div>
          </div>
        </div>

        {/* SVG Canvas */}
        <div style={{ width: '100%', overflowX: 'auto' }}>
          <svg
            viewBox={`0 0 ${chartWidth} ${chartHeight}`}
            style={{ width: '100%', height: 'auto', minWidth: '600px', display: 'block' }}
          >
            <defs>
              <filter id="cyanGlow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
              <filter id="purpleGlow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
              <filter id="goldGlow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            {/* Background Grid Lines (Y-Axis) */}
            {yTicks.map((tickVal) => {
              const yPos = getY(tickVal);
              return (
                <g key={tickVal}>
                  <line
                    x1={padLeft}
                    y1={yPos}
                    x2={chartWidth - padRight}
                    y2={yPos}
                    stroke="rgba(255, 255, 255, 0.04)"
                    strokeWidth="1"
                  />
                  <text
                    x={padLeft - 10}
                    y={yPos + 4}
                    fill="#64748b"
                    fontSize="10"
                    fontFamily="monospace"
                    textAnchor="end"
                  >
                    {tickVal > 0 ? `+${tickVal}%` : `${tickVal}%`}
                  </text>
                </g>
              );
            })}

            {/* Horizontal Breakeven Reference Line (Y = 0%) */}
            <line
              x1={padLeft}
              y1={zeroY}
              x2={chartWidth - padRight}
              y2={zeroY}
              stroke="rgba(255, 255, 255, 0.3)"
              strokeWidth="1.5"
              strokeDasharray="4 4"
            />
            <text
              x={chartWidth - padRight + 6}
              y={zeroY + 3}
              fill="#94a3b8"
              fontSize="10"
              fontFamily="monospace"
            >
              0% (BE)
            </text>

            {/* CRUCIAL VERTICAL HORIZON 1: 1-HOUR (X = 1) */}
            <line
              x1={x1Hour}
              y1={padTop - 10}
              x2={x1Hour}
              y2={chartHeight - padBottom}
              stroke="#F43F5E"
              strokeWidth="2"
              strokeDasharray="5 4"
            />
            {/* Horizon 1 Label Badge */}
            <rect
              x={x1Hour - 42}
              y={padTop - 26}
              width="84"
              height="18"
              rx="4"
              fill="rgba(244, 63, 94, 0.2)"
              stroke="#F43F5E"
              strokeWidth="1"
            />
            <text
              x={x1Hour}
              y={padTop - 13}
              fill="#F43F5E"
              fontSize="9.5"
              fontWeight="bold"
              textAnchor="middle"
            >
              1-HOUR HORIZON
            </text>

            {/* CRUCIAL VERTICAL HORIZON 2: 4-HOUR (X = 4) */}
            <line
              x1={x4Hour}
              y1={padTop - 10}
              x2={x4Hour}
              y2={chartHeight - padBottom}
              stroke="#10B981"
              strokeWidth="2"
              strokeDasharray="5 4"
            />
            {/* Horizon 2 Label Badge */}
            <rect
              x={x4Hour - 42}
              y={padTop - 26}
              width="84"
              height="18"
              rx="4"
              fill="rgba(16, 185, 129, 0.2)"
              stroke="#10B981"
              strokeWidth="1"
            />
            <text
              x={x4Hour}
              y={padTop - 13}
              fill="#10B981"
              fontSize="9.5"
              fontWeight="bold"
              textAnchor="middle"
            >
              4-HOUR HORIZON
            </text>

            {/* Trajectory Lines for the 3 Matches */}
            {matches.map((m, mIdx) => {
              const trajectory = m.price_trajectory || [];
              if (trajectory.length < 2) return null;

              const pathPoints = trajectory.map((ret, t) => `${getX(t)},${getY(ret)}`).join(' L ');
              const filterId = mIdx === 0 ? 'cyanGlow' : mIdx === 1 ? 'purpleGlow' : 'goldGlow';

              return (
                <g key={mIdx}>
                  <path
                    d={`M ${pathPoints}`}
                    fill="none"
                    stroke={trajectoryColors[mIdx].line}
                    strokeWidth="2.5"
                    filter={`url(#${filterId})`}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />

                  {/* Individual Data Points */}
                  {trajectory.map((ret, t) => {
                    const cx = getX(t);
                    const cy = getY(ret);
                    const isKeyHorizon = t === 1 || t === 4;

                    return (
                      <circle
                        key={t}
                        cx={cx}
                        cy={cy}
                        r={isKeyHorizon ? 5 : 3.5}
                        fill="#0B0E14"
                        stroke={trajectoryColors[mIdx].line}
                        strokeWidth={isKeyHorizon ? 2.5 : 2}
                        style={{ cursor: 'pointer', transition: 'all 0.2s' }}
                        onMouseEnter={() => setHoveredPoint({ match: trajectoryColors[mIdx].name, hour: t, return: ret, cx, cy })}
                        onMouseLeave={() => setHoveredPoint(null)}
                      />
                    );
                  })}
                </g>
              );
            })}

            {/* X-Axis Ticks & Labels (T+0 to T+8) */}
            {[0, 1, 2, 3, 4, 5, 6, 7, 8].map((t) => {
              const xPos = getX(t);
              const isKeyHorizon = t === 1 || t === 4;

              return (
                <g key={t}>
                  <line
                    x1={xPos}
                    y1={chartHeight - padBottom}
                    x2={xPos}
                    y2={chartHeight - padBottom + 5}
                    stroke={isKeyHorizon ? '#fff' : 'rgba(255,255,255,0.15)'}
                    strokeWidth={isKeyHorizon ? 2 : 1}
                  />
                  <text
                    x={xPos}
                    y={chartHeight - padBottom + 18}
                    fill={isKeyHorizon ? '#fff' : '#64748b'}
                    fontSize="10"
                    fontFamily="monospace"
                    fontWeight={isKeyHorizon ? 'bold' : 'normal'}
                    textAnchor="middle"
                  >
                    T+{t}
                  </text>
                  <text
                    x={xPos}
                    y={chartHeight - padBottom + 30}
                    fill="#475569"
                    fontSize="8.5"
                    textAnchor="middle"
                  >
                    {t === 0 ? 'Signal' : `${t}h`}
                  </text>
                </g>
              );
            })}

            {/* X-Axis Main Label */}
            <text
              x={padLeft + innerWidth / 2}
              y={chartHeight - 4}
              fill="#94a3b8"
              fontSize="10.5"
              fontWeight="600"
              textAnchor="middle"
              letterSpacing="0.04em"
            >
              HOURS FORWARD FROM SIGNAL (T+0 to T+8)
            </text>

            {/* Hover Tooltip inside SVG */}
            {hoveredPoint && (
              <g transform={`translate(${Math.min(chartWidth - 110, Math.max(padLeft, hoveredPoint.cx - 50))}, ${hoveredPoint.cy - 45})`}>
                <rect
                  x="0"
                  y="0"
                  width="100"
                  height="34"
                  rx="6"
                  fill="#0F172A"
                  stroke="rgba(255,255,255,0.2)"
                  strokeWidth="1"
                  filter="drop-shadow(0 4px 10px rgba(0,0,0,0.8))"
                />
                <text x="50" y="14" fill="#94a3b8" fontSize="9" textAnchor="middle">
                  {hoveredPoint.match} @ T+{hoveredPoint.hour}h
                </text>
                <text
                  x="50"
                  y="28"
                  fill={hoveredPoint.return >= 0 ? '#10B981' : '#F43F5E'}
                  fontSize="12"
                  fontWeight="bold"
                  fontFamily="monospace"
                  textAnchor="middle"
                >
                  {hoveredPoint.return >= 0 ? `+${hoveredPoint.return.toFixed(2)}%` : `${hoveredPoint.return.toFixed(2)}%`}
                </text>
              </g>
            )}
          </svg>
        </div>
      </div>

      {/* Historical Matches Details Table */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', fontWeight: 700, color: 'var(--text-main)' }}>
          <Target size={15} color="#00F0FF" />
          <span>Top 3 Historical Macro-Analogs & Actual Outcomes</span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '12px' }}>
          {matches.map((match, idx) => {
            const is1hPositive = String(match.outcome_1H_return).startsWith('+');
            const is4hPositive = String(match.outcome_4H_return).startsWith('+');

            return (
              <div
                key={idx}
                style={{
                  background: 'rgba(0,0,0,0.3)',
                  border: `1px solid ${trajectoryColors[idx].bg}`,
                  borderLeft: `4px solid ${trajectoryColors[idx].line}`,
                  borderRadius: '10px',
                  padding: '14px 16px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '10px'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Calendar size={13} color="var(--text-dim)" />
                    <strong style={{ fontSize: '12px', color: '#fff', fontFamily: 'var(--font-mono)' }}>
                      {match.match_date}
                    </strong>
                  </div>
                  <span style={{
                    fontSize: '11px',
                    fontWeight: 700,
                    color: trajectoryColors[idx].line,
                    background: trajectoryColors[idx].bg,
                    padding: '2px 8px',
                    borderRadius: '4px',
                    fontFamily: 'var(--font-mono)'
                  }}>
                    {(match.similarity_score * 100).toFixed(0)}% Similarity
                  </span>
                </div>

                {match.catalyst && (
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.4, background: 'rgba(255,255,255,0.02)', padding: '6px 8px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.04)' }}>
                    <span style={{ color: '#94a3b8', fontWeight: 600 }}>Catalyst: </span>
                    <span style={{ color: '#cbd5e1' }}>{match.catalyst}</span>
                  </div>
                )}

                {/* Outcome Return Badges at 1H and 4H */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                  <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: '6px', padding: '8px', display: 'flex', flexDirection: 'column', gap: '2px' }}>
                    <span style={{ fontSize: '10px', color: 'var(--text-dim)', fontWeight: 700 }}>1H OUTCOME</span>
                    <span style={{
                      fontSize: '14px',
                      fontWeight: 800,
                      color: is1hPositive ? 'var(--emerald-buy)' : 'var(--rose-sell)',
                      fontFamily: 'var(--font-mono)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px'
                    }}>
                      {is1hPositive ? <TrendingUp size={13} /> : <TrendingDown size={13} />}
                      {match.outcome_1H_return}
                    </span>
                  </div>

                  <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: '6px', padding: '8px', display: 'flex', flexDirection: 'column', gap: '2px' }}>
                    <span style={{ fontSize: '10px', color: 'var(--text-dim)', fontWeight: 700 }}>4H OUTCOME</span>
                    <span style={{
                      fontSize: '14px',
                      fontWeight: 800,
                      color: is4hPositive ? 'var(--emerald-buy)' : 'var(--rose-sell)',
                      fontFamily: 'var(--font-mono)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px'
                    }}>
                      {is4hPositive ? <TrendingUp size={13} /> : <TrendingDown size={13} />}
                      {match.outcome_4H_return}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
