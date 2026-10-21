import React, { useState, useEffect } from 'react';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Clock,
  Calendar,
  Zap,
  Globe,
  DollarSign,
  BarChart3,
  ShieldAlert,
  Layers,
  RefreshCw,
  Cpu,
  Sliders,
  ChevronDown,
  ChevronUp
} from 'lucide-react';

export default function App() {
  const [mode, setMode] = useState('live'); // 'live' or 'custom'
  
  // Format current UTC / local date & time
  const now = new Date();
  const defaultDate = now.toISOString().split('T')[0];
  const defaultTime = now.toTimeString().slice(0, 5);

  const [tradeDate, setTradeDate] = useState(defaultDate);
  const [tradeTime, setTradeTime] = useState(defaultTime);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);
  const [showJson, setShowJson] = useState(false);

  // Initial Fetch on component mount
  useEffect(() => {
    fetchInference(true);
  }, []);

  const fetchInference = async (isLive = false) => {
    setLoading(true);
    setError(null);
    try {
      const payload = {
        trade_date: isLive ? null : tradeDate,
        trade_time: isLive ? null : tradeTime,
        is_live: isLive
      };

      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`Server returned status: ${response.status}`);
      }

      const result = await response.json();
      setData(result);
    } catch (err) {
      console.error("Prediction fetch failed:", err);
      setError("Failed to fetch market intelligence. Ensure FastAPI backend is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  const handlePreset = (hoursAgo) => {
    setMode('custom');
    const target = new Date(Date.now() - hoursAgo * 3600 * 1000);
    const dateStr = target.toISOString().split('T')[0];
    const timeStr = target.toTimeString().slice(0, 5);
    setTradeDate(dateStr);
    setTradeTime(timeStr);
  };

  const getSignalBadge = (signal) => {
    if (signal === 'BUY') {
      return (
        <span className="signal-pill buy">
          <TrendingUp size={14} /> BUY
        </span>
      );
    } else if (signal === 'SELL') {
      return (
        <span className="signal-pill sell">
          <TrendingDown size={14} /> SELL
        </span>
      );
    }
    return (
      <span className="signal-pill hold">
        <Activity size={14} /> HOLD
      </span>
    );
  };

  const getSynthesis = (sig1h, sig4h, coin) => {
    if (sig1h === 'BUY' && sig4h === 'BUY') {
      return `Strong Bullish Convergence: Both 1-Hour and 4-Hour models detect upward momentum for ${coin}. Favorable for long entries.`;
    } else if (sig1h === 'SELL' && sig4h === 'SELL') {
      return `Bearish Alignment: Both 1-Hour and 4-Hour horizons suggest downward pressure on ${coin}. Exercise caution or consider short setups.`;
    } else if (sig1h === 'BUY' && sig4h === 'HOLD') {
      return `Short-Term Bullish Rebound: 1-Hour shows immediate upward impulse, while 4-Hour remains in neutral consolidation for ${coin}.`;
    } else if (sig1h === 'SELL' && sig4h === 'HOLD') {
      return `Short-Term Pullback: 1-Hour signals immediate weakness within a broader neutral 4-Hour channel for ${coin}.`;
    }
    return `Mixed Signals: 1-Hour is ${sig1h} while 4-Hour is ${sig4h}. High volatility expected; manage risk prudently.`;
  };

  return (
    <div className="dashboard-container">
      {/* 1. Header Bar */}
      <header className="glass-panel header-bar">
        <div className="brand-wrapper">
          <div className="brand-icon">
            <Layers size={22} />
          </div>
          <div>
            <h1 className="brand-title">Agentic AI Market Intelligence</h1>
            <div className="brand-subtitle">
              <span>Bitcoin & Ethereum Multi-Horizon Inference</span>
              <span>•</span>
              <span style={{ color: 'var(--cyan-primary)' }}>Component: Vibhu Kalpitha</span>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div className="status-badge">
            <span className="pulse-dot"></span>
            <span>20 Models Active (1H & 4H)</span>
          </div>

          <button 
            onClick={() => fetchInference(mode === 'live')} 
            className="mode-btn"
            disabled={loading}
            style={{ border: '1px solid var(--border-subtle)' }}
            title="Refresh Data"
          >
            <RefreshCw size={14} className={loading ? 'spinner' : ''} />
            <span>Sync</span>
          </button>
        </div>
      </header>

      {/* 2. Control Deck (Date/Time Input) */}
      <section className="glass-panel control-deck">
        <div className="control-title-row">
          <div className="control-heading">
            <Sliders size={18} color="var(--cyan-primary)" />
            <span>Trade Horizon & Execution Controls</span>
          </div>

          <div className="mode-toggle-group">
            <button
              className={`mode-btn ${mode === 'live' ? 'active' : ''}`}
              onClick={() => {
                setMode('live');
                fetchInference(true);
              }}
            >
              <Zap size={14} /> Real-Time Live
            </button>
            <button
              className={`mode-btn ${mode === 'custom' ? 'active' : ''}`}
              onClick={() => setMode('custom')}
            >
              <Clock size={14} /> Select Date & Time
            </button>
          </div>
        </div>

        <div className="inputs-row">
          <div className="input-box">
            <label className="input-label">
              <Calendar size={13} /> Trade Date
            </label>
            <input
              type="date"
              className="custom-input"
              value={tradeDate}
              onChange={(e) => setTradeDate(e.target.value)}
              disabled={mode === 'live'}
            />
          </div>

          <div className="input-box">
            <label className="input-label">
              <Clock size={13} /> Trade Time (UTC / Local)
            </label>
            <input
              type="time"
              className="custom-input"
              value={tradeTime}
              onChange={(e) => setTradeTime(e.target.value)}
              disabled={mode === 'live'}
            />
          </div>

          <div className="input-box" style={{ justifyContent: 'flex-end' }}>
            <button
              className="btn-analyze"
              onClick={() => fetchInference(mode === 'live')}
              disabled={loading}
            >
              {loading ? (
                <>
                  <RefreshCw size={16} className="spinner" />
                  <span>Computing Models...</span>
                </>
              ) : (
                <>
                  <Cpu size={16} />
                  <span>Run AI Inference</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Quick Presets for Custom Time */}
        {mode === 'custom' && (
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center', paddingTop: '4px' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-dim)', fontWeight: 600 }}>QUICK PRESETS:</span>
            <button onClick={() => handlePreset(1)} className="mode-btn" style={{ fontSize: '11px', padding: '4px 10px', background: 'rgba(255,255,255,0.05)' }}>
              -1 Hour
            </button>
            <button onClick={() => handlePreset(4)} className="mode-btn" style={{ fontSize: '11px', padding: '4px 10px', background: 'rgba(255,255,255,0.05)' }}>
              -4 Hours
            </button>
            <button onClick={() => handlePreset(24)} className="mode-btn" style={{ fontSize: '11px', padding: '4px 10px', background: 'rgba(255,255,255,0.05)' }}>
              -24 Hours
            </button>
          </div>
        )}
      </section>

      {/* Error Message if Any */}
      {error && (
        <div className="glass-panel" style={{ padding: '16px 20px', borderColor: 'var(--rose-sell)', display: 'flex', alignItems: 'center', gap: '12px', color: '#fda4af' }}>
          <ShieldAlert size={20} />
          <span style={{ fontSize: '13px' }}>{error}</span>
        </div>
      )}

      {/* 3. Global Macro Intelligence Bar */}
      {data && (
        <section className="macro-grid">
          <div className="glass-panel macro-card">
            <div className="macro-icon-wrap" style={{ color: '#f59e0b' }}>
              <span style={{ fontWeight: 800 }}>₿</span>
            </div>
            <div className="macro-info">
              <span className="macro-label">BTC / USDT Price</span>
              <span className="macro-value">${data.prices?.BTC?.toLocaleString()}</span>
            </div>
          </div>

          <div className="glass-panel macro-card">
            <div className="macro-icon-wrap" style={{ color: '#c084fc' }}>
              <span style={{ fontWeight: 800 }}>Ξ</span>
            </div>
            <div className="macro-info">
              <span className="macro-label">ETH / USDT Price</span>
              <span className="macro-value">${data.prices?.ETH?.toLocaleString()}</span>
            </div>
          </div>

          <div className="glass-panel macro-card">
            <div className="macro-icon-wrap" style={{ color: '#3b82f6' }}>
              <BarChart3 size={18} />
            </div>
            <div className="macro-info">
              <span className="macro-label">S&P 500 (^GSPC)</span>
              <span className="macro-value">{data.macro?.sp500?.toFixed(2)}</span>
            </div>
          </div>

          <div className="glass-panel macro-card">
            <div className="macro-icon-wrap" style={{ color: '#eab308' }}>
              <DollarSign size={18} />
            </div>
            <div className="macro-info">
              <span className="macro-label">Gold (GLD) / DXY</span>
              <span className="macro-value">{data.macro?.gold?.toFixed(1)} / {data.macro?.dxy?.toFixed(1)}</span>
            </div>
          </div>

          <div className="glass-panel macro-card">
            <div className="macro-icon-wrap" style={{ color: data.macro?.sentiment >= 0 ? 'var(--emerald-buy)' : 'var(--rose-sell)' }}>
              <Globe size={18} />
            </div>
            <div className="macro-info">
              <span className="macro-label">News Polarity (RSS)</span>
              <span className="macro-value" style={{ color: data.macro?.sentiment >= 0 ? 'var(--emerald-buy)' : 'var(--rose-sell)' }}>
                {data.macro?.sentiment >= 0 ? '+' : ''}{data.macro?.sentiment}
              </span>
            </div>
          </div>
        </section>
      )}

      {/* 4. Main Multi-Horizon Cards Grid (BTC & ETH) */}
      {data && (
        <section className="coins-grid">
          {/* BITCOIN CARD */}
          <div className="glass-panel coin-card">
            <div className="coin-header">
              <div className="coin-identity">
                <div className="coin-badge-icon btc">₿</div>
                <div>
                  <h2 className="coin-name">Bitcoin (BTC)</h2>
                  <span style={{ fontSize: '12px', color: 'var(--text-dim)' }}>Reference Timestamp: {data.trade_datetime}</span>
                </div>
              </div>
              <div className="coin-price">${data.prices?.BTC?.toLocaleString()}</div>
            </div>

            {/* 1H & 4H Horizons Deck */}
            <div className="horizons-deck">
              {/* BTC 1-HOUR */}
              <div className="horizon-box">
                <div className="horizon-head">
                  <div className="horizon-tag">
                    <Clock size={12} /> 1-Hour Horizon
                  </div>
                  {getSignalBadge(data.predictions_1h?.BTC?.signal)}
                </div>

                <div className="confidence-wrap">
                  <div className="confidence-label-row">
                    <span>Meta-Learner Confidence</span>
                    <span className="confidence-val">{data.predictions_1h?.BTC?.display_confidence}%</span>
                  </div>
                  <div className="progress-track">
                    <div
                      className={`progress-bar-fill ${data.predictions_1h?.BTC?.signal === 'BUY' ? 'fill-buy' : 'fill-sell'}`}
                      style={{ width: `${Math.min(100, Math.max(10, data.predictions_1h?.BTC?.display_confidence))}%` }}
                    ></div>
                  </div>
                </div>

                {/* Sub-model Experts */}
                <div className="experts-mini-list">
                  <span style={{ fontSize: '10px', color: 'var(--text-dim)', fontWeight: 700 }}>SUB-MODEL EXPERTS</span>
                  <div className="expert-row">
                    <span className="expert-name">Macro Expert</span>
                    <div className="expert-bar-wrapper">
                      <div className="expert-bar-track">
                        <div className="expert-bar-fill" style={{ width: `${data.predictions_1h?.BTC?.experts?.macro}%` }}></div>
                      </div>
                      <span className="expert-score">{data.predictions_1h?.BTC?.experts?.macro}%</span>
                    </div>
                  </div>
                  <div className="expert-row">
                    <span className="expert-name">Whale Flow</span>
                    <div className="expert-bar-wrapper">
                      <div className="expert-bar-track">
                        <div className="expert-bar-fill" style={{ width: `${data.predictions_1h?.BTC?.experts?.whale}%` }}></div>
                      </div>
                      <span className="expert-score">{data.predictions_1h?.BTC?.experts?.whale}%</span>
                    </div>
                  </div>
                  <div className="expert-row">
                    <span className="expert-name">S&P 500 Correlation</span>
                    <div className="expert-bar-wrapper">
                      <div className="expert-bar-track">
                        <div className="expert-bar-fill" style={{ width: `${data.predictions_1h?.BTC?.experts?.sp500}%` }}></div>
                      </div>
                      <span className="expert-score">{data.predictions_1h?.BTC?.experts?.sp500}%</span>
                    </div>
                  </div>
                  <div className="expert-row">
                    <span className="expert-name">Sentiment / News</span>
                    <div className="expert-bar-wrapper">
                      <div className="expert-bar-track">
                        <div className="expert-bar-fill" style={{ width: `${data.predictions_1h?.BTC?.experts?.news}%` }}></div>
                      </div>
                      <span className="expert-score">{data.predictions_1h?.BTC?.experts?.news}%</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* BTC 4-HOUR */}
              <div className="horizon-box">
                <div className="horizon-head">
                  <div className="horizon-tag">
                    <Clock size={12} /> 4-Hour Horizon
                  </div>
                  {getSignalBadge(data.predictions_4h?.BTC?.signal)}
                </div>

                <div className="confidence-wrap">
                  <div className="confidence-label-row">
                    <span>Meta-Learner Confidence</span>
                    <span className="confidence-val">{data.predictions_4h?.BTC?.display_confidence}%</span>
                  </div>
                  <div className="progress-track">
                    <div
                      className={`progress-bar-fill ${data.predictions_4h?.BTC?.signal === 'BUY' ? 'fill-buy' : data.predictions_4h?.BTC?.signal === 'SELL' ? 'fill-sell' : 'fill-hold'}`}
                      style={{ width: `${Math.min(100, Math.max(10, data.predictions_4h?.BTC?.display_confidence))}%` }}
                    ></div>
                  </div>
                </div>

                {/* Sub-model Experts */}
                <div className="experts-mini-list">
                  <span style={{ fontSize: '10px', color: 'var(--text-dim)', fontWeight: 700 }}>SUB-MODEL EXPERTS</span>
                  <div className="expert-row">
                    <span className="expert-name">Macro Expert</span>
                    <div className="expert-bar-wrapper">
                      <div className="expert-bar-track">
                        <div className="expert-bar-fill" style={{ width: `${data.predictions_4h?.BTC?.experts?.macro}%` }}></div>
                      </div>
                      <span className="expert-score">{data.predictions_4h?.BTC?.experts?.macro}%</span>
                    </div>
                  </div>
                  <div className="expert-row">
                    <span className="expert-name">Whale Flow</span>
                    <div className="expert-bar-wrapper">
                      <div className="expert-bar-track">
                        <div className="expert-bar-fill" style={{ width: `${data.predictions_4h?.BTC?.experts?.whale}%` }}></div>
                      </div>
                      <span className="expert-score">{data.predictions_4h?.BTC?.experts?.whale}%</span>
                    </div>
                  </div>
                  <div className="expert-row">
                    <span className="expert-name">S&P 500 Correlation</span>
                    <div className="expert-bar-wrapper">
                      <div className="expert-bar-track">
                        <div className="expert-bar-fill" style={{ width: `${data.predictions_4h?.BTC?.experts?.sp500}%` }}></div>
                      </div>
                      <span className="expert-score">{data.predictions_4h?.BTC?.experts?.sp500}%</span>
                    </div>
                  </div>
                  <div className="expert-row">
                    <span className="expert-name">Sentiment / News</span>
                    <div className="expert-bar-wrapper">
                      <div className="expert-bar-track">
                        <div className="expert-bar-fill" style={{ width: `${data.predictions_4h?.BTC?.experts?.news}%` }}></div>
                      </div>
                      <span className="expert-score">{data.predictions_4h?.BTC?.experts?.news}%</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Strategic Synthesis */}
            <div className="synthesis-card">
              <Zap size={18} color="var(--cyan-primary)" style={{ flexShrink: 0 }} />
              <div className="synthesis-text">
                {getSynthesis(data.predictions_1h?.BTC?.signal, data.predictions_4h?.BTC?.signal, 'BTC')}
              </div>
            </div>
          </div>

          {/* ETHEREUM CARD */}
          <div className="glass-panel coin-card">
            <div className="coin-header">
              <div className="coin-identity">
                <div className="coin-badge-icon eth">Ξ</div>
                <div>
                  <h2 className="coin-name">Ethereum (ETH)</h2>
                  <span style={{ fontSize: '12px', color: 'var(--text-dim)' }}>Reference Timestamp: {data.trade_datetime}</span>
                </div>
              </div>
              <div className="coin-price">${data.prices?.ETH?.toLocaleString()}</div>
            </div>

            {/* 1H & 4H Horizons Deck */}
            <div className="horizons-deck">
              {/* ETH 1-HOUR */}
              <div className="horizon-box">
                <div className="horizon-head">
                  <div className="horizon-tag">
                    <Clock size={12} /> 1-Hour Horizon
                  </div>
                  {getSignalBadge(data.predictions_1h?.ETH?.signal)}
                </div>

                <div className="confidence-wrap">
                  <div className="confidence-label-row">
                    <span>Meta-Learner Confidence</span>
                    <span className="confidence-val">{data.predictions_1h?.ETH?.display_confidence}%</span>
                  </div>
                  <div className="progress-track">
                    <div
                      className={`progress-bar-fill ${data.predictions_1h?.ETH?.signal === 'BUY' ? 'fill-buy' : 'fill-sell'}`}
                      style={{ width: `${Math.min(100, Math.max(10, data.predictions_1h?.ETH?.display_confidence))}%` }}
                    ></div>
                  </div>
                </div>

                {/* Sub-model Experts */}
                <div className="experts-mini-list">
                  <span style={{ fontSize: '10px', color: 'var(--text-dim)', fontWeight: 700 }}>SUB-MODEL EXPERTS</span>
                  <div className="expert-row">
                    <span className="expert-name">Macro Expert</span>
                    <div className="expert-bar-wrapper">
                      <div className="expert-bar-track">
                        <div className="expert-bar-fill" style={{ width: `${data.predictions_1h?.ETH?.experts?.macro}%` }}></div>
                      </div>
                      <span className="expert-score">{data.predictions_1h?.ETH?.experts?.macro}%</span>
                    </div>
                  </div>
                  <div className="expert-row">
                    <span className="expert-name">Whale Flow</span>
                    <div className="expert-bar-wrapper">
                      <div className="expert-bar-track">
                        <div className="expert-bar-fill" style={{ width: `${data.predictions_1h?.ETH?.experts?.whale}%` }}></div>
                      </div>
                      <span className="expert-score">{data.predictions_1h?.ETH?.experts?.whale}%</span>
                    </div>
                  </div>
                  <div className="expert-row">
                    <span className="expert-name">S&P 500 Correlation</span>
                    <div className="expert-bar-wrapper">
                      <div className="expert-bar-track">
                        <div className="expert-bar-fill" style={{ width: `${data.predictions_1h?.ETH?.experts?.sp500}%` }}></div>
                      </div>
                      <span className="expert-score">{data.predictions_1h?.ETH?.experts?.sp500}%</span>
                    </div>
                  </div>
                  <div className="expert-row">
                    <span className="expert-name">Sentiment / News</span>
                    <div className="expert-bar-wrapper">
                      <div className="expert-bar-track">
                        <div className="expert-bar-fill" style={{ width: `${data.predictions_1h?.ETH?.experts?.news}%` }}></div>
                      </div>
                      <span className="expert-score">{data.predictions_1h?.ETH?.experts?.news}%</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* ETH 4-HOUR */}
              <div className="horizon-box">
                <div className="horizon-head">
                  <div className="horizon-tag">
                    <Clock size={12} /> 4-Hour Horizon
                  </div>
                  {getSignalBadge(data.predictions_4h?.ETH?.signal)}
                </div>

                <div className="confidence-wrap">
                  <div className="confidence-label-row">
                    <span>Meta-Learner Confidence</span>
                    <span className="confidence-val">{data.predictions_4h?.ETH?.display_confidence}%</span>
                  </div>
                  <div className="progress-track">
                    <div
                      className={`progress-bar-fill ${data.predictions_4h?.ETH?.signal === 'BUY' ? 'fill-buy' : data.predictions_4h?.ETH?.signal === 'SELL' ? 'fill-sell' : 'fill-hold'}`}
                      style={{ width: `${Math.min(100, Math.max(10, data.predictions_4h?.ETH?.display_confidence))}%` }}
                    ></div>
                  </div>
                </div>

                {/* Sub-model Experts */}
                <div className="experts-mini-list">
                  <span style={{ fontSize: '10px', color: 'var(--text-dim)', fontWeight: 700 }}>SUB-MODEL EXPERTS</span>
                  <div className="expert-row">
                    <span className="expert-name">Macro Expert</span>
                    <div className="expert-bar-wrapper">
                      <div className="expert-bar-track">
                        <div className="expert-bar-fill" style={{ width: `${data.predictions_4h?.ETH?.experts?.macro}%` }}></div>
                      </div>
                      <span className="expert-score">{data.predictions_4h?.ETH?.experts?.macro}%</span>
                    </div>
                  </div>
                  <div className="expert-row">
                    <span className="expert-name">Whale Flow</span>
                    <div className="expert-bar-wrapper">
                      <div className="expert-bar-track">
                        <div className="expert-bar-fill" style={{ width: `${data.predictions_4h?.ETH?.experts?.whale}%` }}></div>
                      </div>
                      <span className="expert-score">{data.predictions_4h?.ETH?.experts?.whale}%</span>
                    </div>
                  </div>
                  <div className="expert-row">
                    <span className="expert-name">S&P 500 Correlation</span>
                    <div className="expert-bar-wrapper">
                      <div className="expert-bar-track">
                        <div className="expert-bar-fill" style={{ width: `${data.predictions_4h?.ETH?.experts?.sp500}%` }}></div>
                      </div>
                      <span className="expert-score">{data.predictions_4h?.ETH?.experts?.sp500}%</span>
                    </div>
                  </div>
                  <div className="expert-row">
                    <span className="expert-name">Sentiment / News</span>
                    <div className="expert-bar-wrapper">
                      <div className="expert-bar-track">
                        <div className="expert-bar-fill" style={{ width: `${data.predictions_4h?.ETH?.experts?.news}%` }}></div>
                      </div>
                      <span className="expert-score">{data.predictions_4h?.ETH?.experts?.news}%</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Strategic Synthesis */}
            <div className="synthesis-card">
              <Zap size={18} color="var(--purple-accent)" style={{ flexShrink: 0 }} />
              <div className="synthesis-text">
                {getSynthesis(data.predictions_1h?.ETH?.signal, data.predictions_4h?.ETH?.signal, 'ETH')}
              </div>
            </div>
          </div>
        </section>
      )}

      {/* 5. Collapsible Raw Audit Log / Payload */}
      {data && (
        <section className="glass-panel" style={{ padding: '16px 24px' }}>
          <div
            onClick={() => setShowJson(!showJson)}
            style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', fontWeight: 600, color: 'var(--text-muted)' }}>
              <Cpu size={15} />
              <span>Inference Raw Telemetry & JSON Payload</span>
            </div>
            {showJson ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </div>

          {showJson && (
            <div style={{ marginTop: '14px', paddingTop: '14px', borderTop: '1px solid var(--border-subtle)' }}>
              <pre
                style={{
                  background: 'rgba(0,0,0,0.5)',
                  padding: '16px',
                  borderRadius: '10px',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '12px',
                  color: '#38bdf8',
                  overflowX: 'auto',
                  maxHeight: '300px'
                }}
              >
                {JSON.stringify(data, null, 2)}
              </pre>
            </div>
          )}
        </section>
      )}

      {/* 6. Footer */}
      <footer className="footer-bar">
        Agentic AI Market Intelligence Platform • Vibhu Kalpitha Component Module • CCXT & Yahoo Finance Live Feed
      </footer>
    </div>
  );
}
