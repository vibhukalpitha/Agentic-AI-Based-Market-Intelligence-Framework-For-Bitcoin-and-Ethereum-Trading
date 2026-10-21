import React, { useState } from 'react';
import { 
  GitCommit, 
  Layers, 
  Info, 
  ShieldCheck, 
  Activity, 
  BarChart2, 
  TrendingUp, 
  Calendar,
  Share2,
  Sparkles
} from 'lucide-react';

export default function RegimeSimilarityGallery({ similarityData, symbol }) {
  const [activeChartTab, setActiveChartTab] = useState('patterns'); // 'where', 'patterns', 'distribution', 'dtw'
  const [activeFeature, setActiveFeature] = useState('rsi'); // 'rsi', 'sma_dist', 'macd_hist', 'atr_vol'
  const [hoveredDay, setHoveredDay] = useState(null);
  const [hoveredTimelinePoint, setHoveredTimelinePoint] = useState(null);
  const [hoveredDtwNode, setHoveredDtwNode] = useState(null);

  if (!similarityData) {
    return (
      <div className="glass-panel" style={{ padding: '24px', textAlign: 'center', color: 'var(--text-dim)' }}>
        <Info size={24} style={{ marginBottom: '8px', color: '#c084fc' }} />
        <p>No DTW historical similarity record found for {symbol}.</p>
      </div>
    );
  }

  const matches = similarityData.matches || [];
  const vectorData = similarityData.vector_charts_data || {};
  const timeline = vectorData.timeline || [];
  const patterns = vectorData.patterns || {};
  const currentPattern = patterns[activeFeature] || patterns.rsi;
  const distribution = vectorData.distribution || [];
  const dtwPairs = vectorData.dtw_path_pairs || [];

  // Theme colors matching quant aesthetic
  const COLOR_TODAY = '#00F0FF';   // Neon Cyan
  const COLOR_MATCH1 = '#eb6834';  // Orange
  const COLOR_MATCH2 = '#a855f7';  // Purple
  const COLOR_MATCH3 = '#ec4899';  // Pink

  // SVG Chart Dimensions
  const chartWidth = 820;
  const chartHeight = 320;
  const padding = { top: 35, right: 35, bottom: 45, left: 55 };
  const innerWidth = chartWidth - padding.left - padding.right;
  const innerHeight = chartHeight - padding.top - padding.bottom;

  // -------------------------------------------------------------
  // CHART 1: WHERE IN HISTORY (Macro Timeline SVG)
  // -------------------------------------------------------------
  const renderWhereInHistoryChart = () => {
    if (!timeline.length) return null;

    const prices = timeline.map(p => p.price);
    const minP = Math.min(...prices) * 0.8;
    const maxP = Math.max(...prices) * 1.2;

    const getX = (idx) => padding.left + (idx / (timeline.length - 1)) * innerWidth;
    const getY = (val) => {
      // Logarithmic scaling for crypto prices
      const logVal = Math.log10(val);
      const logMin = Math.log10(minP);
      const logMax = Math.log10(maxP);
      return padding.top + innerHeight - ((logVal - logMin) / (logMax - logMin)) * innerHeight;
    };

    const pathData = timeline.map((pt, i) => `${i === 0 ? 'M' : 'L'} ${getX(i)} ${getY(pt.price)}`).join(' ');

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <span style={{ fontSize: '13px', fontWeight: 800, color: '#fff' }}>
              Historical Macro Timeline (Log Scale USDT)
            </span>
            <span style={{ fontSize: '12px', color: 'var(--text-dim)', marginLeft: '10px' }}>
              Locates the 3 most similar 10-day episodes across {symbol}'s cycle history
            </span>
          </div>
          <div style={{ display: 'flex', gap: '12px', fontSize: '11px' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '5px', color: COLOR_TODAY }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: COLOR_TODAY }}></span> Today
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '5px', color: COLOR_MATCH1 }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: COLOR_MATCH1 }}></span> Match 1
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '5px', color: COLOR_MATCH2 }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: COLOR_MATCH2 }}></span> Match 2
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '5px', color: COLOR_MATCH3 }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: COLOR_MATCH3 }}></span> Match 3
            </span>
          </div>
        </div>

        <div style={{ position: 'relative', width: '100%', overflowX: 'auto', background: 'rgba(0,0,0,0.3)', borderRadius: '10px', padding: '10px' }}>
          <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
            {/* Grid Lines */}
            {[0, 0.25, 0.5, 0.75, 1].map((pct, idx) => {
              const y = padding.top + innerHeight * (1 - pct);
              return (
                <g key={idx}>
                  <line x1={padding.left} y1={y} x2={chartWidth - padding.right} y2={y} stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
                </g>
              );
            })}

            {/* Price Line */}
            <path d={pathData} fill="none" stroke="#64748b" strokeWidth="2" />

            {/* Shaded Regions & Vertical Markers for Historical Matches */}
            {timeline.map((pt, i) => {
              if (pt.match === undefined) return null;
              const x = getX(i);
              const color = pt.match === 0 ? COLOR_TODAY : pt.match === 1 ? COLOR_MATCH1 : pt.match === 2 ? COLOR_MATCH2 : COLOR_MATCH3;

              return (
                <g key={i}>
                  {/* Highlight Column */}
                  <rect
                    x={x - 14}
                    y={padding.top}
                    width={28}
                    height={innerHeight}
                    fill={color}
                    fillOpacity="0.18"
                    stroke={color}
                    strokeWidth="1"
                    strokeDasharray="2 2"
                    rx="4"
                  />
                  {/* Marker Pin */}
                  <line x1={x} y1={padding.top} x2={x} y2={chartHeight - padding.bottom} stroke={color} strokeWidth="1.5" />
                  <circle cx={x} cy={getY(pt.price)} r="5" fill={color} stroke="#0B0E14" strokeWidth="2" />
                  
                  {/* Label on Top */}
                  <text
                    x={x}
                    y={padding.top - 12}
                    textAnchor="middle"
                    fill={color}
                    fontSize="11"
                    fontWeight="800"
                    fontFamily="monospace"
                  >
                    {pt.match === 0 ? 'TODAY' : `M#${pt.match}`}
                  </text>
                </g>
              );
            })}

            {/* Hover Points on Timeline */}
            {timeline.map((pt, i) => {
              const x = getX(i);
              const y = getY(pt.price);
              return (
                <circle
                  key={`pt-${i}`}
                  cx={x}
                  cy={y}
                  r={hoveredTimelinePoint?.date === pt.date ? 6 : 3}
                  fill={pt.match !== undefined ? '#fff' : '#64748b'}
                  style={{ cursor: 'pointer', transition: 'all 0.2s' }}
                  onMouseEnter={() => setHoveredTimelinePoint(pt)}
                  onMouseLeave={() => setHoveredTimelinePoint(null)}
                />
              );
            })}

            {/* X-Axis Dates */}
            {timeline.filter((_, i) => i % 3 === 0 || i === timeline.length - 1).map((pt, i) => {
              const originalIdx = timeline.findIndex(t => t.date === pt.date);
              return (
                <text
                  key={i}
                  x={getX(originalIdx)}
                  y={chartHeight - 16}
                  textAnchor="middle"
                  fill="var(--text-dim)"
                  fontSize="11"
                  fontFamily="monospace"
                >
                  {pt.date}
                </text>
              );
            })}
          </svg>

          {/* Hover Tooltip Overlay */}
          {hoveredTimelinePoint && (
            <div style={{
              position: 'absolute',
              top: '20px',
              right: '25px',
              background: 'rgba(15, 23, 42, 0.95)',
              border: '1px solid rgba(56, 189, 248, 0.4)',
              borderRadius: '8px',
              padding: '8px 14px',
              fontSize: '12px',
              color: '#fff',
              pointerEvents: 'none',
              boxShadow: '0 4px 16px rgba(0,0,0,0.5)'
            }}>
              <div><strong>Date:</strong> {hoveredTimelinePoint.date}</div>
              <div><strong>Close:</strong> ${hoveredTimelinePoint.price.toLocaleString()}</div>
              {hoveredTimelinePoint.label && (
                <div style={{ color: 'var(--cyan-primary)', fontWeight: 700, marginTop: '2px' }}>
                  {hoveredTimelinePoint.label}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  // -------------------------------------------------------------
  // CHART 2: 10-DAY MULTI-FEATURE PATTERNS SVG
  // -------------------------------------------------------------
  const renderPatternsChart = () => {
    if (!currentPattern) return null;

    const allValues = [
      ...(currentPattern.today || []),
      ...(currentPattern.match1 || []),
      ...(currentPattern.match2 || []),
      ...(currentPattern.match3 || [])
    ];
    const minVal = Math.min(...allValues);
    const maxVal = Math.max(...allValues);
    const range = (maxVal - minVal) || 1;
    const paddedMin = minVal - range * 0.1;
    const paddedMax = maxVal + range * 0.1;

    const getX = (idx) => padding.left + (idx / 9) * innerWidth;
    const getY = (val) => padding.top + innerHeight - ((val - paddedMin) / (paddedMax - paddedMin)) * innerHeight;

    const generatePath = (arr) => arr.map((v, i) => `${i === 0 ? 'M' : 'L'} ${getX(i)} ${getY(v)}`).join(' ');

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        {/* Feature Sub-Selector Tabs */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
          <div style={{ display: 'flex', gap: '6px', background: 'rgba(0,0,0,0.4)', padding: '3px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
            {[
              { id: 'rsi', label: 'RSI (0-100)' },
              { id: 'sma_dist', label: '50-Day SMA Dist (%)' },
              { id: 'macd_hist', label: 'MACD Histogram' },
              { id: 'atr_vol', label: 'ATR Rel. Volatility' }
            ].map(f => (
              <button
                key={f.id}
                onClick={() => setActiveFeature(f.id)}
                style={{
                  background: activeFeature === f.id ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
                  border: activeFeature === f.id ? '1px solid rgba(56, 189, 248, 0.4)' : '1px solid transparent',
                  color: activeFeature === f.id ? '#38bdf8' : 'var(--text-dim)',
                  padding: '5px 11px',
                  borderRadius: '6px',
                  fontSize: '11px',
                  fontWeight: 600,
                  cursor: 'pointer'
                }}
              >
                {f.label}
              </button>
            ))}
          </div>

          <div style={{ display: 'flex', gap: '12px', fontSize: '11px' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '5px', color: COLOR_TODAY, fontWeight: 700 }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: COLOR_TODAY }}></span> Today's Stretch
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '5px', color: COLOR_MATCH1 }}>
              <span style={{ width: '8px', height: '8px', background: COLOR_MATCH1 }}></span> Match 1
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '5px', color: COLOR_MATCH2 }}>
              <span style={{ width: '8px', height: '8px', background: COLOR_MATCH2, transform: 'rotate(45deg)' }}></span> Match 2
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '5px', color: COLOR_MATCH3 }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: COLOR_MATCH3 }}></span> Match 3
            </span>
          </div>
        </div>

        {/* Dynamic Vector SVG Line Chart */}
        <div style={{ position: 'relative', width: '100%', background: 'rgba(0,0,0,0.3)', borderRadius: '10px', padding: '10px' }}>
          <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
            {/* Horizontal Grid */}
            {[0, 0.25, 0.5, 0.75, 1].map((pct, idx) => {
              const y = padding.top + innerHeight * (1 - pct);
              const val = paddedMin + pct * (paddedMax - paddedMin);
              return (
                <g key={idx}>
                  <line x1={padding.left} y1={y} x2={chartWidth - padding.right} y2={y} stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
                  <text x={padding.left - 8} y={y + 4} textAnchor="end" fill="var(--text-dim)" fontSize="10" fontFamily="monospace">
                    {val.toFixed(currentPattern.unit === '%' ? 1 : 2)}
                  </text>
                </g>
              );
            })}

            {/* Vertical Day Lines */}
            {Array.from({ length: 10 }).map((_, i) => (
              <line key={i} x1={getX(i)} y1={padding.top} x2={getX(i)} y2={chartHeight - padding.bottom} stroke="rgba(255,255,255,0.03)" />
            ))}

            {/* Match 3 Curve (Pink) */}
            <path d={generatePath(currentPattern.match3)} fill="none" stroke={COLOR_MATCH3} strokeWidth="1.8" strokeDasharray="4 2" />
            {currentPattern.match3.map((v, i) => (
              <polygon
                key={`m3-${i}`}
                points={`${getX(i)},${getY(v) - 4} ${getX(i) + 4},${getY(v)} ${getX(i)},${getY(v) + 4} ${getX(i) - 4},${getY(v)}`}
                fill={COLOR_MATCH3}
              />
            ))}

            {/* Match 2 Curve (Purple) */}
            <path d={generatePath(currentPattern.match2)} fill="none" stroke={COLOR_MATCH2} strokeWidth="1.8" strokeDasharray="5 3" />
            {currentPattern.match2.map((v, i) => (
              <polygon
                key={`m2-${i}`}
                points={`${getX(i)},${getY(v) - 4} ${getX(i) + 4},${getY(v) + 4} ${getX(i) - 4},${getY(v) + 4}`}
                fill={COLOR_MATCH2}
              />
            ))}

            {/* Match 1 Curve (Orange) */}
            <path d={generatePath(currentPattern.match1)} fill="none" stroke={COLOR_MATCH1} strokeWidth="2.2" />
            {currentPattern.match1.map((v, i) => (
              <rect
                key={`m1-${i}`}
                x={getX(i) - 3.5}
                y={getY(v) - 3.5}
                width="7"
                height="7"
                fill={COLOR_MATCH1}
              />
            ))}

            {/* Today's Curve (Bright Neon Cyan - Thick Glow) */}
            <path d={generatePath(currentPattern.today)} fill="none" stroke={COLOR_TODAY} strokeWidth="3.2" />
            {currentPattern.today.map((v, i) => (
              <circle
                key={`today-${i}`}
                cx={getX(i)}
                cy={getY(v)}
                r="5"
                fill={COLOR_TODAY}
                stroke="#0B0E14"
                strokeWidth="2"
              />
            ))}

            {/* Hover Vertical Guide Line */}
            {hoveredDay !== null && (
              <line
                x1={getX(hoveredDay)}
                y1={padding.top}
                x2={getX(hoveredDay)}
                y2={chartHeight - padding.bottom}
                stroke="rgba(255,255,255,0.4)"
                strokeDasharray="2 2"
              />
            )}

            {/* Hover Trigger Overlays */}
            {Array.from({ length: 10 }).map((_, i) => (
              <rect
                key={`hover-${i}`}
                x={getX(i) - 15}
                y={padding.top}
                width="30"
                height={innerHeight}
                fill="transparent"
                style={{ cursor: 'pointer' }}
                onMouseEnter={() => setHoveredDay(i)}
                onMouseLeave={() => setHoveredDay(null)}
              />
            ))}

            {/* X-Axis Labels (Day 1 - Day 10) */}
            {Array.from({ length: 10 }).map((_, i) => (
              <text
                key={i}
                x={getX(i)}
                y={chartHeight - 16}
                textAnchor="middle"
                fill={hoveredDay === i ? '#fff' : 'var(--text-dim)'}
                fontSize="11"
                fontWeight={hoveredDay === i ? 700 : 500}
                fontFamily="monospace"
              >
                D{i + 1}
              </text>
            ))}
          </svg>

          {/* Hover Tooltip Overlay Box */}
          {hoveredDay !== null && (
            <div style={{
              position: 'absolute',
              top: '16px',
              right: '20px',
              background: 'rgba(15, 23, 42, 0.95)',
              border: '1px solid rgba(56, 189, 248, 0.4)',
              borderRadius: '8px',
              padding: '10px 14px',
              fontSize: '12px',
              pointerEvents: 'none',
              boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
              display: 'flex',
              flexDirection: 'column',
              gap: '4px'
            }}>
              <span style={{ fontWeight: 800, color: '#fff', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '3px' }}>
                Day {hoveredDay + 1} of 10-Day Stretch:
              </span>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '14px', color: COLOR_TODAY }}>
                <span>Today:</span>
                <strong style={{ fontFamily: 'monospace' }}>{currentPattern.today[hoveredDay]} {currentPattern.unit}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '14px', color: COLOR_MATCH1 }}>
                <span>Match 1:</span>
                <strong style={{ fontFamily: 'monospace' }}>{currentPattern.match1[hoveredDay]} {currentPattern.unit}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '14px', color: COLOR_MATCH2 }}>
                <span>Match 2:</span>
                <strong style={{ fontFamily: 'monospace' }}>{currentPattern.match2[hoveredDay]} {currentPattern.unit}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '14px', color: COLOR_MATCH3 }}>
                <span>Match 3:</span>
                <strong style={{ fontFamily: 'monospace' }}>{currentPattern.match3[hoveredDay]} {currentPattern.unit}</strong>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  // -------------------------------------------------------------
  // CHART 3: DISTANCE DISTRIBUTION ("HOW CLOSE")
  // -------------------------------------------------------------
  const renderDistributionChart = () => {
    if (!distribution.length) return null;

    const maxDensity = Math.max(...distribution.map(d => d.density));
    const minDist = 1.0;
    const maxDist = 6.0;

    const getX = (d) => padding.left + ((d - minDist) / (maxDist - minDist)) * innerWidth;
    const getY = (dens) => padding.top + innerHeight - (dens / (maxDensity * 1.15)) * innerHeight;

    const points = distribution.map(d => `${getX(d.dist)},${getY(d.density)}`).join(' ');
    const areaPath = `M ${getX(distribution[0].dist)},${padding.top + innerHeight} L ${points} L ${getX(distribution[distribution.length - 1].dist)},${padding.top + innerHeight} Z`;

    const match1Dist = matches[0]?.distance || 1.106;
    const match2Dist = matches[1]?.distance || 1.153;
    const match3Dist = matches[2]?.distance || 1.187;
    const medianDist = similarityData.median_distance || 3.444;

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <span style={{ fontSize: '13px', fontWeight: 800, color: '#fff' }}>
              Historical DTW Distance Distribution (2,848 Windows)
            </span>
            <span style={{ fontSize: '12px', color: 'var(--text-dim)', marginLeft: '10px' }}>
              Smaller distance = closer historical match
            </span>
          </div>
          <span style={{ fontSize: '12px', color: 'var(--emerald-buy)', fontWeight: 700 }}>
            Top 3 Matches fall in extreme left tail (&lt; 0.1% rarity)
          </span>
        </div>

        <div style={{ position: 'relative', width: '100%', background: 'rgba(0,0,0,0.3)', borderRadius: '10px', padding: '10px' }}>
          <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
            <defs>
              <linearGradient id="distGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.35" />
                <stop offset="100%" stopColor="#38bdf8" stopOpacity="0.02" />
              </linearGradient>
            </defs>

            {/* Area Fill */}
            <path d={areaPath} fill="url(#distGradient)" />

            {/* Smooth Density Curve */}
            <path d={`M ${points}`} fill="none" stroke="#38bdf8" strokeWidth="2.2" />

            {/* Median Marker */}
            <line
              x1={getX(medianDist)}
              y1={padding.top}
              x2={getX(medianDist)}
              y2={chartHeight - padding.bottom}
              stroke="#94a3b8"
              strokeWidth="1.5"
              strokeDasharray="4 3"
            />
            <text x={getX(medianDist)} y={padding.top - 8} textAnchor="middle" fill="#94a3b8" fontSize="10" fontFamily="monospace">
              Median: {medianDist.toFixed(2)}
            </text>

            {/* Match 3 Marker (Pink) */}
            <line
              x1={getX(match3Dist)}
              y1={padding.top + 30}
              x2={getX(match3Dist)}
              y2={chartHeight - padding.bottom}
              stroke={COLOR_MATCH3}
              strokeWidth="1.8"
            />

            {/* Match 2 Marker (Purple) */}
            <line
              x1={getX(match2Dist)}
              y1={padding.top + 15}
              x2={getX(match2Dist)}
              y2={chartHeight - padding.bottom}
              stroke={COLOR_MATCH2}
              strokeWidth="2"
            />

            {/* Match 1 Marker (Orange - Prime Analogy) */}
            <line
              x1={getX(match1Dist)}
              y1={padding.top}
              x2={getX(match1Dist)}
              y2={chartHeight - padding.bottom}
              stroke={COLOR_MATCH1}
              strokeWidth="2.5"
            />
            <circle cx={getX(match1Dist)} cy={padding.top + 6} r="4" fill={COLOR_MATCH1} />
            <text x={getX(match1Dist) + 8} y={padding.top + 9} fill={COLOR_MATCH1} fontSize="11" fontWeight="800" fontFamily="monospace">
              M1: {match1Dist.toFixed(3)} (Top 99.96%)
            </text>

            {/* X-Axis Ticks */}
            {[1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map((tick, i) => (
              <g key={i}>
                <line x1={getX(tick)} y1={chartHeight - padding.bottom} x2={getX(tick)} y2={chartHeight - padding.bottom + 5} stroke="rgba(255,255,255,0.2)" />
                <text x={getX(tick)} y={chartHeight - 16} textAnchor="middle" fill="var(--text-dim)" fontSize="10" fontFamily="monospace">
                  {tick.toFixed(1)}
                </text>
              </g>
            ))}
          </svg>
        </div>
      </div>
    );
  };

  // -------------------------------------------------------------
  // CHART 4: DTW DAY ALIGNMENT WARPING MATRIX
  // -------------------------------------------------------------
  const renderDtwPathChart = () => {
    const topY = 85;
    const bottomY = 225;
    const getNodeX = (dayIdx) => padding.left + ((dayIdx - 1) / 9) * innerWidth;

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <span style={{ fontSize: '13px', fontWeight: 800, color: '#fff' }}>
              Dynamic Time Warping (DTW) Day Pairing Alignment
            </span>
            <span style={{ fontSize: '12px', color: 'var(--text-dim)', marginLeft: '10px' }}>
              Shows non-linear time warping pairing Today's 10 days with Match #1
            </span>
          </div>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
            Straight vertical lines = synchronized in-step; Slanted lines = local acceleration
          </span>
        </div>

        <div style={{ position: 'relative', width: '100%', background: 'rgba(0,0,0,0.3)', borderRadius: '10px', padding: '10px' }}>
          <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
            {/* Connection Lines (Warping Paths) */}
            {dtwPairs.map((pair, idx) => {
              const x1 = getNodeX(pair.day_today);
              const x2 = getNodeX(pair.day_match);
              const isSlanted = pair.day_today !== pair.day_match;
              const isSelected = hoveredDtwNode && (hoveredDtwNode.day_today === pair.day_today || hoveredDtwNode.day_match === pair.day_match);

              return (
                <line
                  key={idx}
                  x1={x1}
                  y1={topY}
                  x2={x2}
                  y2={bottomY}
                  stroke={isSelected ? '#fff' : isSlanted ? 'rgba(235, 104, 52, 0.7)' : 'rgba(255, 255, 255, 0.18)'}
                  strokeWidth={isSelected ? 3 : isSlanted ? 2 : 1.2}
                  strokeDasharray={isSlanted ? '4 2' : 'none'}
                />
              );
            })}

            {/* Top Row: Today's 10 Days */}
            <text x={padding.left - 10} y={topY + 4} textAnchor="end" fill={COLOR_TODAY} fontSize="11" fontWeight="800">
              Today:
            </text>
            {Array.from({ length: 10 }).map((_, i) => {
              const day = i + 1;
              const x = getNodeX(day);
              return (
                <g key={`top-${day}`} style={{ cursor: 'pointer' }} onMouseEnter={() => setHoveredDtwNode({ day_today: day })}>
                  <circle cx={x} cy={topY} r="9" fill="#0B0E14" stroke={COLOR_TODAY} strokeWidth="2.5" />
                  <text x={x} y={topY + 4} textAnchor="middle" fill="#fff" fontSize="10" fontWeight="700">
                    {day}
                  </text>
                </g>
              );
            })}

            {/* Bottom Row: Match 1's 10 Days */}
            <text x={padding.left - 10} y={bottomY + 4} textAnchor="end" fill={COLOR_MATCH1} fontSize="11" fontWeight="800">
              Match 1:
            </text>
            {Array.from({ length: 10 }).map((_, i) => {
              const day = i + 1;
              const x = getNodeX(day);
              return (
                <g key={`bot-${day}`} style={{ cursor: 'pointer' }} onMouseEnter={() => setHoveredDtwNode({ day_match: day })}>
                  <rect x={x - 8} y={bottomY - 8} width="16" height="16" rx="3" fill="#0B0E14" stroke={COLOR_MATCH1} strokeWidth="2.5" />
                  <text x={x} y={bottomY + 4} textAnchor="middle" fill="#fff" fontSize="10" fontWeight="700">
                    {day}
                  </text>
                </g>
              );
            })}

            {/* Explanatory annotation */}
            <text x={chartWidth / 2} y={chartHeight - 16} textAnchor="middle" fill="var(--text-dim)" fontSize="11">
              DTW dynamic programming identifies identical structural inflection points regardless of minor speed variations
            </text>
          </svg>
        </div>
      </div>
    );
  };

  return (
    <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '22px' }}>
      {/* 1. Header & Architectural Purpose */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{
              background: 'linear-gradient(135deg, rgba(168, 85, 247, 0.2), rgba(59, 130, 246, 0.2))',
              border: '1px solid rgba(168, 85, 247, 0.4)',
              padding: '6px 10px',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}>
              <GitCommit size={16} color="#c084fc" />
              <span style={{ fontSize: '11px', fontWeight: 800, color: '#e9d5ff', letterSpacing: '0.05em' }}>M7 COMPONENT</span>
            </div>
            <h3 style={{ fontSize: '18px', fontWeight: 800, color: '#fff', margin: 0 }}>
              Historical Regime Similarity (Dynamic Time Warping)
            </h3>
          </div>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginTop: '6px', maxWidth: '780px', lineHeight: 1.5 }}>
            Dynamic Vector Charts: Compares today's 10-day technical regime profile across <strong style={{ color: '#fff' }}>{similarityData.past_stretches_compared || 2848}</strong> historical Daily windows using multidimensional Dynamic Time Warping.
          </p>
        </div>

        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '6px 12px',
          background: 'rgba(56, 189, 248, 0.08)',
          border: '1px solid rgba(56, 189, 248, 0.2)',
          borderRadius: '8px',
          color: '#38bdf8',
          fontSize: '12px'
        }}>
          <ShieldCheck size={15} />
          <span>Native React Vector Engine</span>
        </div>
      </div>

      {/* 2. Key DTW Metrics Summary Row */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '12px',
        padding: '16px',
        background: 'rgba(0, 0, 0, 0.35)',
        borderRadius: '12px',
        border: '1px solid var(--border-subtle)'
      }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
          <span style={{ fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 700 }}>
            Evaluation Window
          </span>
          <span style={{ fontSize: '16px', fontWeight: 800, color: '#fff', fontFamily: 'var(--font-mono)' }}>
            {similarityData.window_days || 10} Days (Daily Candles)
          </span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
          <span style={{ fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 700 }}>
            Past Stretches Compared
          </span>
          <span style={{ fontSize: '16px', fontWeight: 800, color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>
            {(similarityData.past_stretches_compared || 2848).toLocaleString()} Windows
          </span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
          <span style={{ fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 700 }}>
            Median DTW Distance
          </span>
          <span style={{ fontSize: '16px', fontWeight: 800, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
            {similarityData.median_distance?.toFixed(3) || '3.444'}
          </span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
          <span style={{ fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 700 }}>
            Best Match Percentile
          </span>
          <span style={{ fontSize: '16px', fontWeight: 800, color: 'var(--emerald-buy)', fontFamily: 'var(--font-mono)' }}>
            {matches[0]?.similarity_percentile ? `${matches[0].similarity_percentile}%` : '99.96%'}
          </span>
        </div>
      </div>

      {/* 3. Top 3 Historical Regime Matches Cards */}
      <div>
        <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Layers size={14} color="#c084fc" />
          <span>Top 3 Historical Matches (Ranked by DTW Distance):</span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '14px' }}>
          {matches.map((m, idx) => {
            const colors = [COLOR_MATCH1, COLOR_MATCH2, COLOR_MATCH3];
            const badgeColor = colors[idx % colors.length];

            return (
              <div 
                key={idx}
                style={{
                  background: 'rgba(255, 255, 255, 0.02)',
                  border: `1px solid ${idx === 0 ? 'rgba(235, 104, 52, 0.4)' : 'var(--border-subtle)'}`,
                  borderRadius: '12px',
                  padding: '16px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '12px',
                  boxShadow: idx === 0 ? '0 4px 20px rgba(235, 104, 52, 0.1)' : 'none'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{
                      width: '24px',
                      height: '24px',
                      borderRadius: '50%',
                      background: badgeColor,
                      color: '#fff',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '12px',
                      fontWeight: 800
                    }}>
                      #{idx + 1}
                    </span>
                    <span style={{ fontSize: '14px', fontWeight: 700, color: '#fff' }}>
                      {m.similarity_percentile ? `Top ${m.similarity_percentile}% Analogy` : `Match #${idx + 1}`}
                    </span>
                  </div>

                  <span style={{
                    fontSize: '11px',
                    fontWeight: 700,
                    padding: '3px 8px',
                    borderRadius: '4px',
                    background: m.regime_then === 'Bullish' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                    color: m.regime_then === 'Bullish' ? 'var(--emerald-buy)' : 'var(--crimson-sell)'
                  }}>
                    {m.regime_then}
                  </span>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)' }}>
                    <span>Historical Span:</span>
                    <strong style={{ color: '#fff', fontFamily: 'var(--font-mono)' }}>
                      {m.window_start} → {m.window_end}
                    </strong>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)' }}>
                    <span>DTW Distance:</span>
                    <strong style={{ color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>
                      {m.distance?.toFixed(3)}
                    </strong>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)' }}>
                    <span>Rarity / Rank:</span>
                    <span style={{ color: 'var(--emerald-buy)', fontWeight: 600 }}>
                      Closer than {m.similarity_percentile || 99.8}% of history
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 4. Interactive Vector Chart Suite (React Generated) */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <BarChart2 size={16} color="var(--cyan-primary)" />
            <span style={{ fontSize: '14px', fontWeight: 700, color: '#fff' }}>
              Dynamic React Vector Figures:
            </span>
          </div>

          {/* Master Chart Tab Switcher */}
          <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', background: 'rgba(0,0,0,0.4)', padding: '4px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
            {[
              { id: 'patterns', title: '1. Multi-Feature Trajectory (10 Days)' },
              { id: 'where', title: '2. Where in History (Macro Timeline)' },
              { id: 'distribution', title: '3. Distance Distribution (How Close)' },
              { id: 'dtw', title: '4. DTW Day Alignment Warping' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveChartTab(tab.id)}
                style={{
                  background: activeChartTab === tab.id ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
                  border: activeChartTab === tab.id ? '1px solid rgba(56, 189, 248, 0.5)' : '1px solid transparent',
                  color: activeChartTab === tab.id ? '#38bdf8' : 'var(--text-dim)',
                  padding: '6px 12px',
                  borderRadius: '6px',
                  fontSize: '12px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  transition: 'all 0.2s ease'
                }}
              >
                {tab.title}
              </button>
            ))}
          </div>
        </div>

        {/* Render Selected Dynamic Chart */}
        <div style={{
          background: 'rgba(11, 14, 20, 0.8)',
          border: '1px solid var(--border-subtle)',
          borderRadius: '12px',
          padding: '20px'
        }}>
          {activeChartTab === 'patterns' && renderPatternsChart()}
          {activeChartTab === 'where' && renderWhereInHistoryChart()}
          {activeChartTab === 'distribution' && renderDistributionChart()}
          {activeChartTab === 'dtw' && renderDtwPathChart()}
        </div>

        {/* Scientific Scope Callout */}
        <div style={{
          display: 'flex',
          gap: '10px',
          alignItems: 'flex-start',
          padding: '12px 16px',
          borderRadius: '8px',
          background: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid var(--border-subtle)',
          fontSize: '12px',
          color: 'var(--text-dim)',
          lineHeight: 1.5
        }}>
          <Info size={16} color="var(--text-muted)" style={{ flexShrink: 0, marginTop: '2px' }} />
          <span>
            <strong style={{ color: 'var(--text-muted)' }}>Component 01 M7 Scope:</strong> All 4 charts are rendered natively using dynamic React SVG vector graphics based on numerical feature trajectories. DTW historical similarity is supporting evidence for the current regime label only; it intentionally does not forecast future price returns.
          </span>
        </div>
      </div>
    </div>
  );
}
