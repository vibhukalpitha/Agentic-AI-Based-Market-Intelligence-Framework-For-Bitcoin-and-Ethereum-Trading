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
  ChevronUp,
  GitMerge,
  Compass,
  CheckCircle2,
  AlertTriangle,
  Info,
  History
} from 'lucide-react';
import HistoricalAnalogyChart from './HistoricalAnalogyChart.jsx';
import RegimeSimilarityGallery from './RegimeSimilarityGallery.jsx';

export default function App() {
  const [activeEngineTab, setActiveEngineTab] = useState('cross_market'); // 'cross_market' or 'regime_engine'
  const [activeRegimeCoin, setActiveRegimeCoin] = useState('BTC'); // 'BTC' or 'ETH'
  const [activeAnalogyCoin, setActiveAnalogyCoin] = useState('BTC'); // 'BTC' or 'ETH'
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

  const getRegimePill = (regime) => {
    const reg = regime?.toLowerCase() || 'sideways';
    if (reg.includes('bull')) {
      return <span className="regime-pill bullish"><TrendingUp size={12} /> {regime}</span>;
    } else if (reg.includes('bear')) {
      return <span className="regime-pill bearish"><TrendingDown size={12} /> {regime}</span>;
    } else if (reg.includes('volat')) {
      return <span className="regime-pill volatile"><Activity size={12} /> {regime}</span>;
    }
    return <span className="regime-pill sideways"><Compass size={12} /> {regime}</span>;
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

  const currentRegimeData = data?.regimes?.[activeRegimeCoin];

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
              <span>Dual-Engine Intelligence Platform for Bitcoin & Ethereum</span>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div className="status-badge">
            <span className="pulse-dot"></span>
            <span>All Engines Active</span>
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

      {/* 2. Common Control Deck (Date/Time Input for BOTH components) */}
      <section className="glass-panel control-deck">
        <div className="control-title-row">
          <div className="control-heading">
            <Sliders size={18} color="var(--cyan-primary)" />
            <span>Trade Horizon & Execution Controls (Synchronized Common Input)</span>
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
                  <span>Computing Both Engines...</span>
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

      {/* 3. Engine Tabs Navigation Bar */}
      <nav className="engine-tabs-bar">
        <button
          className={`engine-tab-btn ${activeEngineTab === 'cross_market' ? 'active' : ''}`}
          onClick={() => setActiveEngineTab('cross_market')}
        >
          <div className="tab-icon-wrap">
            <Cpu size={20} />
          </div>
          <div className="tab-text-wrap">
            <span className="tab-title">Cross-Market & Internal Crypto Data Engine</span>
            <span className="tab-subtitle">1-Hour & 4-Hour Future Predictions • Vibhu Kalpitha Component</span>
          </div>
        </button>

        <button
          className={`engine-tab-btn ${activeEngineTab === 'regime_engine' ? 'active' : ''}`}
          onClick={() => setActiveEngineTab('regime_engine')}
        >
          <div className="tab-icon-wrap">
            <GitMerge size={20} />
          </div>
          <div className="tab-text-wrap">
            <span className="tab-title">Multi-Timeframe Regime Engine</span>
            <span className="tab-subtitle">Adaptive HMM Regimes (1D, 4H, 1H, 15M, 5M) • Component 01</span>
          </div>
        </button>
      </nav>

      {/* Error Message if Any */}
      {error && (
        <div className="glass-panel" style={{ padding: '16px 20px', borderColor: 'var(--rose-sell)', display: 'flex', alignItems: 'center', gap: '12px', color: '#fda4af' }}>
          <ShieldAlert size={20} />
          <span style={{ fontSize: '13px' }}>{error}</span>
        </div>
      )}

      {/* =========================================================================
          TAB 1: CROSS-MARKET & INTERNAL CRYPTO DATA ENGINE (Vibhu Kalpitha)
         ========================================================================= */}
      {activeEngineTab === 'cross_market' && data && (
        <>
          {/* Global Macro Intelligence Bar */}
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

          {/* Main Dual-Horizon Cards Grid (BTC & ETH) */}
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

          {/* Historical Regime Similarity & Outcome Trajectory Section */}
          <section style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '15px', fontWeight: 800 }}>
                <History size={18} color="#00F0FF" />
                <span>Historical Analogy & Trajectory Visualizer (T+0 to T+8)</span>
              </div>

              {/* Coin Switcher for Analogy Chart */}
              <div style={{ display: 'flex', background: 'rgba(0,0,0,0.4)', padding: '3px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                <button
                  className={`mode-btn ${activeAnalogyCoin === 'BTC' ? 'active' : ''}`}
                  onClick={() => setActiveAnalogyCoin('BTC')}
                  style={{ fontSize: '11px', padding: '5px 14px' }}
                >
                  <span>₿ Bitcoin (BTC)</span>
                </button>
                <button
                  className={`mode-btn ${activeAnalogyCoin === 'ETH' ? 'active' : ''}`}
                  onClick={() => setActiveAnalogyCoin('ETH')}
                  style={{ fontSize: '11px', padding: '5px 14px' }}
                >
                  <span>Ξ Ethereum (ETH)</span>
                </button>
              </div>
            </div>

            <HistoricalAnalogyChart
              analogyData={data?.historical_analogies?.[activeAnalogyCoin]}
              activeCoin={activeAnalogyCoin}
            />
          </section>
        </>
      )}

      {/* =========================================================================
          TAB 2: MULTI-TIMEFRAME REGIME ENGINE (Component 01)
         ========================================================================= */}
      {activeEngineTab === 'regime_engine' && currentRegimeData && (
        <section style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Asset Selector & Snapshot Time Header */}
          <div className="glass-panel" style={{ padding: '16px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
              <div style={{ display: 'flex', background: 'rgba(0,0,0,0.4)', padding: '4px', borderRadius: '10px', border: '1px solid var(--border-subtle)' }}>
                <button
                  className={`mode-btn ${activeRegimeCoin === 'BTC' ? 'active' : ''}`}
                  onClick={() => setActiveRegimeCoin('BTC')}
                >
                  <span>₿ Bitcoin (BTCUSDT)</span>
                </button>
                <button
                  className={`mode-btn ${activeRegimeCoin === 'ETH' ? 'active' : ''}`}
                  onClick={() => setActiveRegimeCoin('ETH')}
                >
                  <span>Ξ Ethereum (ETHUSDT)</span>
                </button>
              </div>
              <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
                Evaluated At: <strong style={{ color: '#fff' }}>{currentRegimeData.as_of}</strong>
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span style={{ fontSize: '12px', color: 'var(--text-dim)' }}>Engine Schema: v{currentRegimeData.schema_version}</span>
              <div className="status-badge" style={{ background: 'rgba(168, 85, 247, 0.1)', borderColor: 'rgba(168, 85, 247, 0.3)', color: '#c084fc' }}>
                <span>5 Timeframes Synced</span>
              </div>
            </div>
          </div>

          {/* Fusion Engine Overview Banner */}
          <div className="glass-panel fusion-overview-bar">
            {/* Dominant Regime Box */}
            <div className="fusion-stat-box">
              <span style={{ fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 700 }}>
                Dominant Market Regime
              </span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                {getRegimePill(currentRegimeData.fusion?.dominant_regime)}
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  ({currentRegimeData.fusion?.timeframes_agreeing?.length || 0} of 5 timeframes agree)
                </span>
              </div>
            </div>

            {/* Alignment Score Meter */}
            <div className="fusion-stat-box">
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 700 }}>
                <span>Multi-Timeframe Alignment</span>
                <span style={{ color: '#fff', fontFamily: 'var(--font-mono)' }}>{currentRegimeData.fusion?.alignment_score?.toFixed(1)}%</span>
              </div>
              <div className="progress-track" style={{ height: '7px' }}>
                <div
                  className="progress-bar-fill"
                  style={{
                    width: `${currentRegimeData.fusion?.alignment_score}%`,
                    background: 'linear-gradient(90deg, #3b82f6, #06b6d4)'
                  }}
                ></div>
              </div>
              <span style={{ fontSize: '11px', color: 'var(--text-dim)' }}>
                Degree of consensus across Daily, 4h, 1h, 15m, and 5m models
              </span>
            </div>

            {/* Stability Forecast */}
            <div className="fusion-stat-box">
              <span style={{ fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 700 }}>
                3-Day Regime Stability Score
              </span>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
                <span style={{ fontSize: '20px', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--emerald-buy)' }}>
                  {((currentRegimeData.stability?.probability_regime_persists || 0.85) * 100).toFixed(1)}%
                </span>
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  Chance {currentRegimeData.fusion?.dominant_regime} persists
                </span>
              </div>
            </div>
          </div>

          {/* 5-Timeframe Matrix (1D, 4H, 1H, 15M, 5M) */}
          <div className="glass-panel" style={{ padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '18px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '15px', fontWeight: 700 }}>
                <BarChart3 size={17} color="var(--cyan-primary)" />
                <span>Multi-Timeframe Regime Detection Matrix (Gaussian HMM)</span>
              </div>
              <span style={{ fontSize: '12px', color: 'var(--text-dim)' }}>Independent statistical state per timeframe</span>
            </div>

            <div className="regime-matrix-grid">
              {['1d', '4h', '1h', '15m', '5m'].map((tfKey) => {
                const tfData = currentRegimeData.timeframes?.[tfKey];
                const tfLabelMap = {
                  '1d': 'Daily (1D)',
                  '4h': '4-Hour (4H)',
                  '1h': '1-Hour (1H)',
                  '15m': '15-Minute (15M)',
                  '5m': '5-Minute (5M)'
                };

                return (
                  <div key={tfKey} className="regime-tf-card">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '13px', fontWeight: 800, color: 'var(--text-main)', letterSpacing: '0.03em' }}>
                        {tfLabelMap[tfKey]}
                      </span>
                      {getRegimePill(tfData?.regime)}
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-dim)' }}>
                        <span>Confidence</span>
                        <span style={{ fontFamily: 'var(--font-mono)', color: '#fff' }}>
                          {((tfData?.confidence || 0.95) * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="progress-track" style={{ height: '4px' }}>
                        <div
                          className="progress-bar-fill"
                          style={{
                            width: `${(tfData?.confidence || 0.95) * 100}%`,
                            background: 'var(--cyan-primary)'
                          }}
                        ></div>
                      </div>
                    </div>

                    <div className="regime-stats-row">
                      <span>Candles Held:</span>
                      <strong style={{ color: '#fff', fontFamily: 'var(--font-mono)' }}>{tfData?.candles_in_regime || 12}</strong>
                    </div>

                    <div className="regime-stats-row">
                      <span>Typical RSI:</span>
                      <strong style={{ color: '#fff', fontFamily: 'var(--font-mono)' }}>{tfData?.regime_profile?.typical_rsi?.toFixed(1) || '52.0'}</strong>
                    </div>

                    <div className="regime-stats-row">
                      <span>Volat. vs Norm:</span>
                      <strong style={{ color: '#fff', fontFamily: 'var(--font-mono)' }}>{tfData?.regime_profile?.volatility_vs_norm?.toFixed(2) || '1.00'}x</strong>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Timeframe Propagation & Conflict Archetype */}
          <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '15px', fontWeight: 700 }}>
              <Compass size={17} color="#c084fc" />
              <span>Timeframe Cascade & Disagreement Archetype</span>
            </div>

            <div className="explanation-callout">
              <Info size={22} color="#c084fc" style={{ flexShrink: 0, marginTop: '2px' }} />
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span style={{ fontSize: '14px', fontWeight: 800, color: '#fff' }}>
                    Archetype: {currentRegimeData.explanation?.archetype || 'Aligned'}
                  </span>
                  {currentRegimeData.propagation?.['4h']?.direction && (
                    <span style={{ fontSize: '11px', padding: '3px 8px', borderRadius: '4px', background: 'rgba(255,255,255,0.08)', color: 'var(--text-muted)' }}>
                      Cascade: {currentRegimeData.propagation?.['4h']?.direction}
                    </span>
                  )}
                </div>
                <p style={{ fontSize: '13px', lineHeight: 1.5, color: 'var(--text-main)' }}>
                  {currentRegimeData.explanation?.text}
                </p>
              </div>
            </div>
          </div>

          {/* M7 Historical Regime Similarity (DTW Pattern Matching & Visual Charts) */}
          <RegimeSimilarityGallery
            similarityData={currentRegimeData.dtw_similarity || currentRegimeData.similar_history}
            symbol={activeRegimeCoin}
          />
        </section>
      )}

      {/* 4. Collapsible Raw Telemetry & JSON */}
      {data && (
        <section className="glass-panel" style={{ padding: '16px 24px' }}>
          <div
            onClick={() => setShowJson(!showJson)}
            style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', fontWeight: 600, color: 'var(--text-muted)' }}>
              <Cpu size={15} />
              <span>Inference Raw Telemetry & Multi-Engine JSON Payload</span>
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

      {/* 5. Footer */}
      <footer className="footer-bar">
        Agentic AI Market Intelligence Platform • Vibhu Kalpitha Component (Cross-Market & Crypto Data) & Component 01 (Multi-Timeframe Regime Engine)
      </footer>
    </div>
  );
}
