import type {
  DriverImpactData,
  ForecastItem,
  HistoricalEvidenceItem,
  InternalSignalData,
  NewsArticle,
  PricePoint,
  ShapFeature,
  WhaleTransaction
} from '../types';

export const OVERVIEW_KPI = {
  btc: {
    price: "$67,842.31",
    rawPrice: 67842.31,
    change24h: "+2.41%",
    status: "Bullish"
  },
  eth: {
    price: "$3,842.72",
    rawPrice: 3842.72,
    change24h: "+1.87%",
    status: "Bullish"
  },
  sentiment: {
    score: 68,
    label: "Greed"
  },
  openInterest: {
    val: "$18.42B",
    change: "+4.8%"
  }
};

export const MARKET_REGIME = {
  status: "Moderately Bullish",
  confidence: 78,
  factors: [
    { label: "BTC momentum", direction: "up", status: "bullish" },
    { label: "S&P 500", direction: "up", status: "bullish" },
    { label: "DXY", direction: "down", status: "bullish" },
    { label: "Gold", direction: "right", status: "neutral" },
    { label: "Open Interest", direction: "up", status: "bullish" },
    { label: "News sentiment", direction: "up", status: "positive" }
  ]
};

export const AI_INTELLIGENCE_SUMMARY = {
  title: "AI Market Intelligence Summary",
  synthesis: "BTC currently shows a moderately bullish short-term structure. Positive BTC momentum and increasing open interest support the upside scenario, while a weaker DXY provides a favorable cross-market signal. News sentiment remains positive. Historical evidence shows similar market conditions have more frequently resulted in positive 1–4 hour BTC movement, although uncertainty increases toward the 4-hour horizon.",
  keyDrivers: [
    { name: "BTC momentum", state: "Bullish", impact: "High" },
    { name: "Open Interest", state: "Bullish", impact: "High" },
    { name: "DXY", state: "Bullish (Inverse correlate weaker)", impact: "Medium" },
    { name: "News Sentiment", state: "Bullish", impact: "Medium" },
    { name: "VIX", state: "Neutral / Bearish Risk", impact: "Low" }
  ]
};

// Generates time series mock price data based on range
export const getPriceData = (asset: 'BTC' | 'ETH', range: '1H' | '4H' | '1D' | '1W'): PricePoint[] => {
  const basePrice = asset === 'BTC' ? 67842.31 : 3842.72;
  const count = range === '1H' ? 12 : range === '4H' ? 24 : range === '1D' ? 30 : 40;
  
  const points: PricePoint[] = [];
  let currentPrice = basePrice * 0.96;
  
  for (let i = 0; i < count; i++) {
    const timeLabel = range === '1H' ? `${i * 5}m` : range === '4H' ? `${i * 10}m` : range === '1D' ? `${i}:00` : `Day ${i+1}`;
    const randomDelta = (Math.random() - 0.44) * (basePrice * 0.006);
    currentPrice += randomDelta;
    points.push({
      time: timeLabel,
      price: parseFloat(currentPrice.toFixed(2)),
      volume: Math.floor(Math.random() * 4000 + 2000)
    });
  }
  
  // Make last price match current static price for consistency
  if (points.length > 0) {
    points[points.length - 1].price = basePrice;
  }
  
  return points;
};

// BTC Drivers & Internal Signals
export const BTC_DRIVERS: DriverImpactData[] = [
  { name: "DXY (US Dollar Index)", impactScore: -18, label: "Bearish impact (-18)", type: "bearish", description: "Dollar weakening exerts upward pressure on BTC price." },
  { name: "S&P 500 Correlation", impactScore: 14, label: "Bullish impact (+14)", type: "bullish", description: "US equity market strength lifting crypto risk appetite." },
  { name: "Gold Index", impactScore: 3, label: "Neutral (+3)", type: "neutral", description: "Precious metals trading in tight range." },
  { name: "VIX Risk Index", impactScore: -9, label: "Bearish risk signal (-9)", type: "bearish", description: "Low volatility spike probability." },
  { name: "Open Interest Momentum", impactScore: 17, label: "Bullish momentum (+17)", type: "bullish", description: "Derivatives leverage expanding with positive delta." }
];

export const BTC_INTERNAL_SIGNALS: InternalSignalData[] = [
  { name: "Transaction Activity", value: "348,120 tx/day", trend: "up", status: "bullish", change24h: "+5.4%" },
  { name: "Whale Activity", value: "High Outflow", trend: "up", status: "bullish", change24h: "Accumulation phase" },
  { name: "Exchange Net Flow", value: "-$55.0M", trend: "up", status: "bullish", change24h: "Cold storage transfers" },
  { name: "Open Interest", value: "$18.42 Billion", trend: "up", status: "bullish", change24h: "+4.8%" },
  { name: "Funding Rate", value: "+0.0142%", trend: "neutral", status: "neutral", change24h: "Healthy long bias" }
];

// ETH Drivers & Internal Signals
export const ETH_DRIVERS: DriverImpactData[] = [
  { name: "BTC Correlation", impactScore: 24, label: "Strong Bullish (+24)", type: "bullish", description: "High short-horizon beta with Bitcoin price action." },
  { name: "DXY (US Dollar Index)", impactScore: -15, label: "Bearish impact (-15)", type: "bearish", description: "Weaker dollar supportive of DeFi liquidity." },
  { name: "S&P 500 Correlation", impactScore: 12, label: "Bullish impact (+12)", type: "bullish", description: "Tech stock alignment." },
  { name: "Gold Index", impactScore: 2, label: "Neutral (+2)", type: "neutral", description: "Minimal macro gold spillovers." },
  { name: "Open Interest", impactScore: 14, label: "Bullish momentum (+14)", type: "bullish", description: "Ether futures open interest +3.9%." },
  { name: "News Sentiment", impactScore: 11, label: "Positive (+11)", type: "bullish", description: "L2 adoption and staking expansion news." }
];

export const ETH_INTERNAL_SIGNALS: InternalSignalData[] = [
  { name: "Staking Ratio", value: "28.4% of Supply", trend: "up", status: "bullish", change24h: "+0.4%" },
  { name: "Gas / Burn Rate", value: "24 Gwei / 940 ETH/day", trend: "up", status: "bullish", change24h: "+12.1%" },
  { name: "Layer 2 Volume", value: "$4.82 Billion", trend: "up", status: "bullish", change24h: "+8.7%" },
  { name: "Exchange Net Flow", value: "-$45.0M", trend: "up", status: "bullish", change24h: "Net outflow" },
  { name: "Funding Rate", value: "+0.0118%", trend: "neutral", status: "neutral", change24h: "Balanced leverage" }
];

// Cross-Market Matrix Data
export const CROSS_MARKET_ASSETS = ['BTC', 'ETH', 'DXY', 'Gold', 'S&P 500', 'Nasdaq', 'VIX', 'Open Interest'];

export const CORRELATION_MATRIX: Record<string, Record<string, number>> = {
  'BTC':           { 'BTC': 1.00, 'ETH': 0.82, 'DXY': -0.41, 'Gold': 0.18, 'S&P 500': 0.36, 'Nasdaq': 0.42, 'VIX': -0.38, 'Open Interest': 0.64 },
  'ETH':           { 'BTC': 0.82, 'ETH': 1.00, 'DXY': -0.37, 'Gold': 0.14, 'S&P 500': 0.39, 'Nasdaq': 0.45, 'VIX': -0.34, 'Open Interest': 0.58 },
  'DXY':           { 'BTC': -0.41, 'ETH': -0.37, 'DXY': 1.00, 'Gold': -0.52, 'S&P 500': -0.48, 'Nasdaq': -0.51, 'VIX': 0.41, 'Open Interest': -0.29 },
  'Gold':          { 'BTC': 0.18, 'ETH': 0.14, 'DXY': -0.52, 'Gold': 1.00, 'S&P 500': 0.21, 'Nasdaq': 0.19, 'VIX': -0.12, 'Open Interest': 0.08 },
  'S&P 500':       { 'BTC': 0.36, 'ETH': 0.39, 'DXY': -0.48, 'Gold': 0.21, 'S&P 500': 1.00, 'Nasdaq': 0.94, 'VIX': -0.76, 'Open Interest': 0.31 },
  'Nasdaq':        { 'BTC': 0.42, 'ETH': 0.45, 'DXY': -0.51, 'Gold': 0.19, 'S&P 500': 0.94, 'Nasdaq': 1.00, 'VIX': -0.79, 'Open Interest': 0.35 },
  'VIX':           { 'BTC': -0.38, 'ETH': -0.34, 'DXY': 0.41, 'Gold': -0.12, 'S&P 500': -0.76, 'Nasdaq': -0.79, 'VIX': 1.00, 'Open Interest': -0.22 },
  'Open Interest': { 'BTC': 0.64, 'ETH': 0.58, 'DXY': -0.29, 'Gold': 0.08, 'S&P 500': 0.31, 'Nasdaq': 0.35, 'VIX': -0.22, 'Open Interest': 1.00 }
};

export const RELATIONSHIP_INSIGHTS = [
  {
    pair: "BTC ↔ DXY",
    title: "Moderate inverse relationship detected",
    value: "-0.41",
    type: "negative",
    description: "Statistical inverse co-movement: when the US Dollar index drops, BTC tends to absorb capital inflows as a macro hedge. Note: Analytical correlation, not direct causality."
  },
  {
    pair: "BTC ↔ S&P 500",
    title: "Positive short-term relationship",
    value: "+0.36",
    type: "positive",
    description: "Crypto assets continue to display moderate alignment with risk-on equity market regimes during NY trading sessions."
  },
  {
    pair: "BTC ↔ ETH",
    title: "Strong positive relationship",
    value: "+0.82",
    type: "positive",
    description: "High co-integration baseline. ETH follows BTC directional impulses with approximately 1.15x higher short-horizon volatility."
  },
  {
    pair: "BTC ↔ Open Interest",
    title: "High structural co-expansion",
    value: "+0.64",
    type: "positive",
    description: "Derivatives volume and leverage expansion correlate strongly with upward momentum pushes."
  }
];

// Capital Flow Data
export const CAPITAL_FLOW_STATS = {
  btc: {
    inflow: "$482M",
    outflow: "$537M",
    netFlow: "-$55M",
    whaleActivity: "High",
    netFlowTrend: "bullish"
  },
  eth: {
    inflow: "$291M",
    outflow: "$336M",
    netFlow: "-$45M",
    whaleActivity: "Moderate",
    netFlowTrend: "bullish"
  }
};

export const CAPITAL_FLOW_TIMELINE = [
  { time: "00:00", btcIn: 120, btcOut: 145, ethIn: 70, ethOut: 85 },
  { time: "04:00", btcIn: 95, btcOut: 110, ethIn: 55, ethOut: 62 },
  { time: "08:00", btcIn: 160, btcOut: 195, ethIn: 88, ethOut: 105 },
  { time: "12:00", btcIn: 210, btcOut: 240, ethIn: 115, ethOut: 130 },
  { time: "16:00", btcIn: 185, btcOut: 220, ethIn: 98, ethOut: 118 },
  { time: "20:00", btcIn: 140, btcOut: 165, ethIn: 75, ethOut: 90 },
  { time: "24:00", btcIn: 482, btcOut: 537, ethIn: 291, ethOut: 336 }
];

export const WHALE_TRANSACTIONS: WhaleTransaction[] = [
  { id: "tx-1", timestamp: "12 mins ago", asset: "BTC", amount: "2,450 BTC", valueUsd: "$166.2M", from: "Binance Hot Wallet", to: "Unknown Cold Storage", type: "Outflow", impact: "High" },
  { id: "tx-2", timestamp: "34 mins ago", asset: "ETH", amount: "18,200 ETH", valueUsd: "$69.9M", from: "Kraken Custody", to: "Lido Staking Contract", type: "Transfer", impact: "High" },
  { id: "tx-3", timestamp: "1 hour ago", asset: "BTC", amount: "1,120 BTC", valueUsd: "$75.9M", from: "Coinbase Prime", to: "Fidelity ETF Custody", type: "Transfer", impact: "High" },
  { id: "tx-4", timestamp: "2 hours ago", asset: "ETH", amount: "12,500 ETH", valueUsd: "$48.0M", from: "Unknown Whale 0x7a...f9", to: "Binance Exchange", type: "Inflow", impact: "Medium" },
  { id: "tx-5", timestamp: "3 hours ago", asset: "BTC", amount: "890 BTC", valueUsd: "$60.3M", from: "Bitfinex Wallet", to: "Institutional Treasury", type: "Outflow", impact: "Medium" }
];

// News & Sentiment Data
export const SENTIMENT_OVERALL = {
  positive: 62,
  neutral: 24,
  negative: 14,
  finbert: {
    positive: 0.72,
    neutral: 0.19,
    negative: 0.09
  }
};

export const NEWS_ARTICLES: NewsArticle[] = [
  {
    id: "news-1",
    headline: "Bitcoin ETF inflows strengthen market confidence as spot volumes surge past $4.2B",
    source: "Bloomberg Crypto",
    time: "25 mins ago",
    asset: "BTC",
    sentiment: "Positive",
    impact: "High",
    confidence: 91,
    summary: "Institutional appetite for spot Bitcoin exchange-traded funds recorded the third consecutive day of net positive inflows, boosting dealer delta positions.",
    finbertScores: { positive: 0.88, neutral: 0.09, negative: 0.03 }
  },
  {
    id: "news-2",
    headline: "Federal Reserve policy signal hints at favorable liquidity expansion in upcoming quarter",
    source: "Reuters Financial",
    time: "1 hour ago",
    asset: "ALL",
    sentiment: "Positive",
    impact: "High",
    confidence: 86,
    summary: "Macro signals indicate stable inflation data, encouraging lower bond yields and dollar softening across risk assets.",
    finbertScores: { positive: 0.76, neutral: 0.18, negative: 0.06 }
  },
  {
    id: "news-3",
    headline: "Ethereum Layer-2 daily active addresses cross record 3.4M threshold",
    source: "CoinDesk Research",
    time: "2 hours ago",
    asset: "ETH",
    sentiment: "Positive",
    impact: "Medium",
    confidence: 84,
    summary: "Arbitrum and Base scale rollups show record transaction velocity while mainnet gas fees remain constrained.",
    finbertScores: { positive: 0.81, neutral: 0.14, negative: 0.05 }
  },
  {
    id: "news-4",
    headline: "Derivatives leverage reaches 3-month peak ahead of options expiry",
    source: "Deribit Analytics",
    time: "3 hours ago",
    asset: "BTC",
    sentiment: "Neutral",
    impact: "Medium",
    confidence: 78,
    summary: "Open interest accumulation indicates potential volatility compression prior to the Friday call-put rebalancing.",
    finbertScores: { positive: 0.22, neutral: 0.68, negative: 0.10 }
  },
  {
    id: "news-5",
    headline: "Regulatory warning issued over algorithmic synthetic stablecoin spreads",
    source: "Financial Times",
    time: "5 hours ago",
    asset: "ETH",
    sentiment: "Negative",
    impact: "Low",
    confidence: 89,
    summary: "Minor regulatory scrutiny over specialized yield protocols causes brief hesitation in algorithmic liquidity pools.",
    finbertScores: { positive: 0.05, neutral: 0.21, negative: 0.74 }
  }
];

// Forecast Data (Prediction Page)
export const FORECAST_BTC: ForecastItem[] = [
  { horizon: "+1 Hour", direction: "Bullish", expectedReturn: "+0.42%", confidence: 81, targetPrice: 68127.25, description: "High short-term momentum backed by ETF volume & DXY drop." },
  { horizon: "+2 Hours", direction: "Bullish", expectedReturn: "+0.67%", confidence: 77, targetPrice: 68296.88, description: "Sustained order book bid depth above $67,500 pivot level." },
  { horizon: "+3 Hours", direction: "Bullish", expectedReturn: "+0.83%", confidence: 73, targetPrice: 68405.51, description: "LSTM temporal vector aligns with historical morning rally window." },
  { horizon: "+4 Hours", direction: "Neutral-Bullish", expectedReturn: "+0.91%", confidence: 68, targetPrice: 68459.76, description: "Variance widens near upper resistance boundary; moderate profit taking expected." }
];

export const FORECAST_ETH: ForecastItem[] = [
  { horizon: "+1 Hour", direction: "Bullish", expectedReturn: "+0.38%", confidence: 80, targetPrice: 3857.32, description: "ETH follows BTC lead with positive gas consumption delta." },
  { horizon: "+2 Hours", direction: "Bullish", expectedReturn: "+0.61%", confidence: 76, targetPrice: 3866.16, description: "Staking withdrawals low; net exchange outflow continues." },
  { horizon: "+3 Hours", direction: "Bullish", expectedReturn: "+0.79%", confidence: 71, targetPrice: 3873.07, description: "L2 volume growth supporting mainnet value settlement." },
  { horizon: "+4 Hours", direction: "Neutral-Bullish", expectedReturn: "+0.88%", confidence: 66, targetPrice: 3876.53, description: "Broader resistance near $3,880 may cap immediate extension." }
];

// Historical Evidence / RAG Data
export const RAG_CURRENT_SITUATION = {
  btcMomentum: "+1.8%",
  dxy: "-0.4%",
  sp500: "+0.6%",
  fearGreed: 68,
  newsSentiment: "Positive"
};

export const HISTORICAL_EVIDENCE: HistoricalEvidenceItem[] = [
  {
    id: "hist-1",
    date: "March 14, 2025",
    similarity: 91,
    conditions: {
      btcMomentum: "↑ (+2.1%)",
      dxy: "↓ (-0.45%)",
      sp500: "↑ (+0.7%)",
      newsSentiment: "Positive (0.78)",
      fearGreed: 71
    },
    outcome: "BTC +2.7% over next 4 hours",
    outcomeReturn: 2.7,
    notes: "Spot ETF inflow surge coincided with soft inflation print. Market followed high-confidence XGBoost/LSTM prediction pathway.",
    vectorDistance: 0.089
  },
  {
    id: "hist-2",
    date: "September 18, 2024",
    similarity: 86,
    conditions: {
      btcMomentum: "↑ (+1.6%)",
      dxy: "↓ (-0.38%)",
      sp500: "→ (+0.1%)",
      newsSentiment: "Neutral-Positive",
      fearGreed: 64
    },
    outcome: "BTC +1.9% over next 4 hours",
    outcomeReturn: 1.9,
    notes: "Federal Reserve rate cut announcement environment. Open interest expanded rapidly during the 2nd hour.",
    vectorDistance: 0.142
  },
  {
    id: "hist-3",
    date: "January 12, 2025",
    similarity: 81,
    conditions: {
      btcMomentum: "→ (+0.2%)",
      dxy: "↑ (+0.31%)",
      sp500: "↓ (-0.4%)",
      newsSentiment: "Positive (0.64)",
      fearGreed: 66
    },
    outcome: "BTC -0.8% over next 4 hours",
    outcomeReturn: -0.8,
    notes: "Macro headwinds overrode positive crypto news sentiment. Sharp DXY bounce caused leverage flush in derivatives.",
    vectorDistance: 0.187
  },
  {
    id: "hist-4",
    date: "June 04, 2024",
    similarity: 79,
    conditions: {
      btcMomentum: "↑ (+1.9%)",
      dxy: "→ (-0.05%)",
      sp500: "↑ (+0.5%)",
      newsSentiment: "Positive",
      fearGreed: 69
    },
    outcome: "BTC +3.1% over next 4 hours",
    outcomeReturn: 3.1,
    notes: "Strong institutional accumulation. Whale outflows from exchanges preceded a multi-hour breakout.",
    vectorDistance: 0.205
  }
];

// XGBoost SHAP Feature Importance
export const XGBOOST_SHAP_FEATURES: ShapFeature[] = [
  { feature: "BTC Momentum (1h/4h)", importance: 0.31, impact: "positive", category: "internal" },
  { feature: "Open Interest Delta", importance: 0.22, impact: "positive", category: "internal" },
  { feature: "DXY Return (USD Index)", importance: -0.18, impact: "negative", category: "cross-market" },
  { feature: "S&P 500 Return", importance: 0.14, impact: "positive", category: "cross-market" },
  { feature: "News Sentiment Score", importance: 0.11, impact: "positive", category: "sentiment" },
  { feature: "Gold Return", importance: 0.04, impact: "positive", category: "cross-market" }
];
