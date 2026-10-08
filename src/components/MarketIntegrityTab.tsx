import React, { useState, useEffect, useCallback } from 'react';
import {
  ComposedChart,
  LineChart,
  BarChart,
  Line,
  Area,
  Bar,
  XAxis,
  YAxis,
  Tooltip as ReTooltip,
  ResponsiveContainer,
  CartesianGrid,
  ReferenceLine,
  Cell,
  Legend,
} from 'recharts';
import {
  ShieldCheck,
  AlertTriangle,
  XCircle,
  Activity,
  Cpu,
  Eye,
  Network,
  TrendingUp,
  RefreshCw,
  Clock,
  Zap,
  BarChart2,
  LayoutDashboard,
  GitMerge,
  History,
  BrainCircuit,
  SlidersHorizontal,
  ChevronRight,
  Info,
  Code2,
  Sparkles,
  Coins,
  Layers,
  Flame,
  ArrowUpRight,
  ArrowDownRight,
  Target,
  CheckCircle2,
  ExternalLink,
  ShieldAlert,
  GitBranch,
  X,
  FileText,
  FileSearch,
  Share2,
  MousePointerClick,
} from 'lucide-react';

// ─────────────────────────────────────────────────────────────
// TYPES & DATA CONTRACTS
// ─────────────────────────────────────────────────────────────
export type IntegrityState = 'HEALTHY' | 'WATCH' | 'SUSPICIOUS' | 'COMPROMISED';
export type AssetMode = 'ALL' | 'BTC' | 'ETH';
export type TimeframeMode = '1m' | '5m' | '15m' | '1H';

// 5 CLEAN MAIN TABS
export type Component3SubTab =
  | 'overview'
  | 'microstructure'
  | 'ml-evidence'
  | 'manipulation-diagnostics'
  | 'explainability-historical';

export type ManipulationPatternType =
  | 'wash-trading'
  | 'pump-and-dump'
  | 'spoofing'
  | 'liquidity-pulls'
  | 'fake-breakouts';

export interface ModelBranchScore {
  autoencoderAnomalyScore: number;
  xgboostPatternRisk: number;
  transformerTemporalScore: number;
  gnnMicrostructureScore: number;
}

export interface ShapFeatureEntry {
  feature: string;
  displayName: string;
  importance: number; // signed: positive = integrity-raising (>0), negative = integrity-degrading (<0)
  direction: 'positive' | 'negative';
}

export interface HistoricalWindow {
  id: string;
  timestamp: string;
  asset: string;
  similarityScore: number;
  historicalState: IntegrityState;
  observedOutcome: string;
  integrityOutcomeScore: number; // Post-event integrity score (0-100)
  vectorDistance: number;
  eventPattern: string;
  conditions: {
    spread: string;
    obImbalance: string;
    rushOrderDensity: string;
    cancelRate: string;
    volumeRatio: string;
    graphCoupling: string;
  };
  notes: string;
  postEventReaction: {
    t15m: string;
    t1h: string;
    t4h: string;
  };
}

export interface TimelinePoint {
  time: string;
  integrityScore: number;
  price: number;
  anomalyFlag: boolean;
  volume: number;
}

export interface KeyDriver {
  label: string;
  value: string;
  status: 'normal' | 'elevated' | 'critical';
}

export interface MarketBanner {
  currentPrice: string;
  volume24h: string;
  spread: string;
  orderBookImbalance: string;
}

export interface IntegrityDataPayload {
  symbol: string;
  integrityScore: number;
  integrityState: IntegrityState;
  unseenAnomalyRate: number;
  knownPatternRisk: number;
  modelConfidence: number;
  conformalCoverage: number;
  modelBranches: ModelBranchScore;
  shapFeatures: ShapFeatureEntry[];
  historicalWindows: HistoricalWindow[];
  timeline: TimelinePoint[];
  keyDrivers: KeyDriver[];
  marketBanner: MarketBanner;
  diagnosticSummary: string;
  decisionNote: string;
  lastUpdated: string;
}

// ─────────────────────────────────────────────────────────────
// ACTIVE QUERY VECTOR CONSTANTS
// ─────────────────────────────────────────────────────────────
export const VECTOR_QUERY_INTEGRITY_STATE = {
  spread: '$1.20 (Optimal)',
  obImbalance: '+4.2% (Bid-Heavy)',
  rushOrderDensity: '2.1% (Low)',
  vpinToxicity: '0.14 (Clean)',
  cancelVelocity: '1.8% (Normal)',
  graphCoupling: '0.06 (Stable)',
};

// ─────────────────────────────────────────────────────────────
// MOCK DATA GENERATOR
// ─────────────────────────────────────────────────────────────
function buildTimeline(baseScore: number, basePrice: number): TimelinePoint[] {
  const pts: TimelinePoint[] = [];
  let score = baseScore;
  let price = basePrice;
  const labels = [
    '09:00', '09:05', '09:10', '09:15', '09:20', '09:25',
    '09:30', '09:35', '09:40', '09:45', '09:50', '09:55',
    '10:00', '10:05', '10:10', '10:15', '10:20', '10:25',
    '10:30', '10:35'
  ];
  for (let i = 0; i < labels.length; i++) {
    const drift = (Math.random() - 0.46) * 3.5;
    const priceDrift = (Math.random() - 0.48) * basePrice * 0.0025;
    score = Math.min(99, Math.max(25, score + drift));
    price = Math.max(basePrice * 0.98, price + priceDrift);
    const vol = Math.round(1200 + Math.random() * 800);
    pts.push({
      time: labels[i],
      integrityScore: Math.round(score),
      price: Math.round(price),
      anomalyFlag: score < 65,
      volume: vol,
    });
  }
  return pts;
}

export async function fetchIntegrityData(
  asset: AssetMode,
  _timeframe: TimeframeMode
): Promise<IntegrityDataPayload> {
  await new Promise((r) => setTimeout(r, 200));
  const isBtc = asset === 'BTC' || asset === 'ALL';
  const baseScore = isBtc ? 92 : 84;
  const basePrice = asset === 'ETH' ? 3842 : 67842;
  const state: IntegrityState = isBtc ? 'HEALTHY' : 'WATCH';

  return {
    symbol: asset === 'ALL' ? 'BTC & ETH Multi-Asset' : `${asset}/USDT`,
    integrityScore: baseScore,
    integrityState: state,
    unseenAnomalyRate: isBtc ? 0.08 : 0.22,
    knownPatternRisk: isBtc ? 0.04 : 0.18,
    modelConfidence: isBtc ? 95.0 : 88.5,
    conformalCoverage: isBtc ? 95.0 : 90.0,
    modelBranches: {
      autoencoderAnomalyScore: isBtc ? 0.08 : 0.22,
      xgboostPatternRisk: isBtc ? 0.04 : 0.18,
      transformerTemporalScore: isBtc ? 0.09 : 0.24,
      gnnMicrostructureScore: isBtc ? 0.06 : 0.21,
    },
    shapFeatures: [
      { feature: 'bid_ask_spread', displayName: 'Bid-Ask Spread', importance: isBtc ? 0.34 : 0.28, direction: 'positive' },
      { feature: 'rush_order_ratio', displayName: 'Rush Order Ratio', importance: isBtc ? -0.28 : -0.37, direction: 'negative' },
      { feature: 'volume_spike_index', displayName: 'Volume Spike Index', importance: isBtc ? -0.21 : -0.29, direction: 'negative' },
      { feature: 'trade_imbalance', displayName: 'Trade Imbalance', importance: isBtc ? 0.18 : 0.14, direction: 'positive' },
      { feature: 'cancellation_rate', displayName: 'Cancellation Rate', importance: isBtc ? -0.14 : -0.19, direction: 'negative' },
    ],
    historicalWindows: [
      {
        id: 'hist-1',
        timestamp: '2024-03-12 14:22 UTC',
        asset: 'BTC/USDT',
        similarityScore: 94.2,
        historicalState: 'HEALTHY',
        eventPattern: 'Organic Liquidity Clustering',
        integrityOutcomeScore: 94,
        observedOutcome: 'High microstructure integrity sustained; zero manipulative patterns detected across subsequent window.',
        vectorDistance: 0.042,
        conditions: {
          spread: '$1.18',
          obImbalance: '+3.9%',
          rushOrderDensity: '1.9%',
          cancelRate: '1.4%',
          volumeRatio: '1.05×',
          graphCoupling: '0.05',
        },
        notes: 'Microstructure vector exhibited strong topological alignment with balanced institutional maker quoting. All 4 ML branches confirmed low reconstruction errors and absence of toxicity.',
        postEventReaction: {
          t15m: 'Tight spread maintained at $1.16; zero toxic order flow observed.',
          t1h: 'Integrity score remained robust at 94/100 with balanced depth.',
          t4h: 'Market concluded rolling window in fully verified healthy regime.',
        },
      },
      {
        id: 'hist-2',
        timestamp: '2024-01-08 09:11 UTC',
        asset: 'ETH/USDT',
        similarityScore: 88.6,
        historicalState: 'WATCH',
        eventPattern: 'Transient Rush Order Spike',
        integrityOutcomeScore: 88,
        observedOutcome: 'Transient rush-order burst resolved within 45 minutes; microstructure normalized to healthy status.',
        vectorDistance: 0.087,
        conditions: {
          spread: '$0.24',
          obImbalance: '-5.2%',
          rushOrderDensity: '8.4%',
          cancelRate: '4.8%',
          volumeRatio: '2.40×',
          graphCoupling: '0.19',
        },
        notes: 'DEX-CEX arbitrage flow caused rapid localized quoting rebalance. Conformal prediction interval widened by 8% during peak intensity before settling back to standard bounds.',
        postEventReaction: {
          t15m: 'Spread widened momentarily to $0.38 then normalized rapidly.',
          t1h: 'Order book imbalance neutralized to -0.8% with score recovery.',
          t4h: 'Integrity normalized to 88/100 with low residual anomaly error.',
        },
      },
      {
        id: 'hist-3',
        timestamp: '2023-11-20 21:47 UTC',
        asset: 'BTC/USDT',
        similarityScore: 82.1,
        historicalState: 'WATCH',
        eventPattern: 'Order Book Layering Attempt',
        integrityOutcomeScore: 86,
        observedOutcome: 'Mild order-book asymmetry absorbed by passive institutional bids; layering spoofing thwarted.',
        vectorDistance: 0.128,
        conditions: {
          spread: '$1.45',
          obImbalance: '+8.1%',
          rushOrderDensity: '4.2%',
          cancelRate: '6.2%',
          volumeRatio: '1.65×',
          graphCoupling: '0.14',
        },
        notes: 'Rapid spoofing cancellations on ask levels were flagged by the XGBoost supervised classifier (prob 0.44). Passive bids absorbed the pressure without liquidity collapse.',
        postEventReaction: {
          t15m: 'Layering orders pulled within 8 minutes of system detection alert.',
          t1h: 'Integrity score recovered from 74 back to 86/100.',
          t4h: 'Microstructure fully stabilized with normal quote-to-trade ratio.',
        },
      },
      {
        id: 'hist-4',
        timestamp: '2023-08-04 03:30 UTC',
        asset: 'ETH/USDT',
        similarityScore: 77.9,
        historicalState: 'SUSPICIOUS',
        eventPattern: 'Wash Trading Burst & Volume Spike',
        integrityOutcomeScore: 48,
        observedOutcome: 'Spoofing burst detected at T+10min; system flagged 32 cancelled large-lot bids and thinned liquidity.',
        vectorDistance: 0.161,
        conditions: {
          spread: '$0.52',
          obImbalance: '+14.2%',
          rushOrderDensity: '18.6%',
          cancelRate: '19.4%',
          volumeRatio: '4.80×',
          graphCoupling: '0.42',
        },
        notes: 'Coordinated non-economic circular trades triggered the Autoencoder anomaly score (0.58) and T-GNN edge distortion alarm. Integrity score dropped to 48.',
        postEventReaction: {
          t15m: 'Aggressive wash volume halted after regulatory and exchange alerts.',
          t1h: 'Order book liquidity thinned by 42% as artificial bid walls dropped.',
          t4h: 'Protracted recovery period required before integrity normalization.',
        },
      },
      {
        id: 'hist-5',
        timestamp: '2023-05-18 16:45 UTC',
        asset: 'BTC/USDT',
        similarityScore: 73.4,
        historicalState: 'COMPROMISED',
        eventPattern: 'Coordinated Pump-and-Dump Cascade',
        integrityOutcomeScore: 32,
        observedOutcome: 'High-frequency spoofing followed by immediate liquidity withdrawal and severe integrity breach.',
        vectorDistance: 0.198,
        conditions: {
          spread: '$3.80',
          obImbalance: '+22.5%',
          rushOrderDensity: '34.1%',
          cancelRate: '38.2%',
          volumeRatio: '7.20×',
          graphCoupling: '0.68',
        },
        notes: 'Severe multi-branch model divergence. Both Transformer temporal attention and T-GNN relational shift flags triggered maximum emergency alerts.',
        postEventReaction: {
          t15m: 'Artificial buy frenzy accompanied by massive cancel bursts.',
          t1h: 'Total liquidity withdrawal causing sharp microstructure degradation.',
          t4h: 'Integrity score plunged to 32/100, requiring manual intervention.',
        },
      },
      {
        id: 'hist-6',
        timestamp: '2023-02-14 11:20 UTC',
        asset: 'ETH/USDT',
        similarityScore: 71.2,
        historicalState: 'HEALTHY',
        eventPattern: 'High-Frequency Arbitrage Flow',
        integrityOutcomeScore: 91,
        observedOutcome: 'Clean cross-venue price convergence with zero spoofing signals and tight spreads.',
        vectorDistance: 0.215,
        conditions: {
          spread: '$0.19',
          obImbalance: '+1.8%',
          rushOrderDensity: '3.1%',
          cancelRate: '2.1%',
          volumeRatio: '1.20×',
          graphCoupling: '0.07',
        },
        notes: 'Standard market maker rebalancing in response to global macro data release. High conformal coverage maintained throughout.',
        postEventReaction: {
          t15m: 'Slight increase in trade frequency with tight spread maintained.',
          t1h: 'Liquidity depth replenished without localized distortion.',
          t4h: 'Robust organic market conditions persisted with 91/100 score.',
        },
      },
    ],
    timeline: buildTimeline(baseScore, basePrice),
    keyDrivers: [
      { label: 'Spread Stability', value: 'Normal', status: 'normal' },
      { label: 'Wash Volume Ratio', value: '0.02%', status: 'normal' },
      { label: 'Rush Order Density', value: 'Low', status: 'normal' },
      { label: 'Graph Coupling', value: 'Stable', status: 'normal' },
    ],
    marketBanner: {
      currentPrice: isBtc ? '$67,842' : '$3,842',
      volume24h: isBtc ? '$38.4B' : '$14.2B',
      spread: isBtc ? '$1.20' : '$0.18',
      orderBookImbalance: isBtc ? '+4.2%' : '−2.1%',
    },
    diagnosticSummary:
      'BTC and ETH market structures currently display high market integrity. Order-book imbalances remain within normal bounds (+4.2%), and sell-side cancellation rates show no signs of spoofing or liquidity withdrawal. All 4 ML branches confirm low risk environment.',
    decisionNote:
      'Integrity metrics indicate robust organic price discovery. Risk constraints can operate with standard confidence bands.',
    lastUpdated: new Date().toISOString(),
  };
}

// ─────────────────────────────────────────────────────────────
// STATE STYLING CONFIG
// ─────────────────────────────────────────────────────────────
function getStateConfig(state: IntegrityState) {
  switch (state) {
    case 'HEALTHY':
      return { color: '#22c55e', hex: 'emerald', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400', glow: 'shadow-[0_0_20px_rgba(34,197,94,0.2)]', Icon: ShieldCheck };
    case 'WATCH':
      return { color: '#eab308', hex: 'yellow', bg: 'bg-yellow-500/10', border: 'border-yellow-500/30', text: 'text-yellow-400', glow: 'shadow-[0_0_20px_rgba(234,179,8,0.2)]', Icon: Eye };
    case 'SUSPICIOUS':
      return { color: '#f97316', hex: 'orange', bg: 'bg-orange-500/10', border: 'border-orange-500/30', text: 'text-orange-400', glow: 'shadow-[0_0_20px_rgba(249,115,22,0.2)]', Icon: AlertTriangle };
    case 'COMPROMISED':
      return { color: '#ef4444', hex: 'red', bg: 'bg-red-500/10', border: 'border-red-500/30', text: 'text-red-400', glow: 'shadow-[0_0_20px_rgba(239,68,68,0.2)]', Icon: XCircle };
  }
}

function getRiskColor(s: number) {
  return s < 0.15 ? '#22c55e' : s < 0.35 ? '#eab308' : s < 0.6 ? '#f97316' : '#ef4444';
}
function getRiskLabel(s: number) {
  return s < 0.15 ? 'LOW RISK' : s < 0.35 ? 'MODERATE' : s < 0.6 ? 'ELEVATED' : 'HIGH RISK';
}

// ─────────────────────────────────────────────────────────────
// REUSABLE MICRO-COMPONENTS
// ─────────────────────────────────────────────────────────────

const StateBadge: React.FC<{ state: IntegrityState; size?: 'sm' | 'md' }> = ({ state, size = 'md' }) => {
  const cfg = getStateConfig(state);
  const Icon = cfg.Icon;
  return (
    <div
      className={`inline-flex items-center gap-1.5 rounded-lg border font-bold font-mono tracking-wider ${cfg.bg} ${cfg.border} ${cfg.glow} ${
        size === 'md' ? 'px-3.5 py-1.5 text-xs' : 'px-2.5 py-1 text-[11px]'
      }`}
      style={{ color: cfg.color }}
    >
      <Icon className={size === 'md' ? 'w-4 h-4' : 'w-3 h-3'} />
      {state}
    </div>
  );
};

interface MetricCardProps {
  title: string;
  value: string | number;
  subtext: string;
  status?: string;
  icon: React.ReactNode;
  badge?: React.ReactNode;
}

const C2StyleMetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtext,
  status,
  icon,
  badge,
}) => {
  return (
    <div className="bg-[#12141c] border border-[#1e2330] p-4 rounded-xl shadow-md space-y-2 hover:border-[#2a3050] transition-all relative overflow-hidden">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{title}</span>
        <div className="p-2 rounded-lg bg-[#0d0f17] border border-[#1e2330]">{icon}</div>
      </div>
      <div className="flex items-baseline justify-between gap-2">
        <div className="text-2xl font-black font-mono text-white tracking-tight">{value}</div>
        {badge}
      </div>
      <div className="flex items-center justify-between text-[11px] pt-1 border-t border-[#1e2330]">
        <span className="text-gray-400">{subtext}</span>
        {status && (
          <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-bold font-mono text-[10px]">
            {status}
          </span>
        )}
      </div>
    </div>
  );
};

// 4-Branch Model Card
interface ModelBranchCardProps {
  primaryTitle: string;
  techBadge: string;
  description: string;
  score: number;
  Icon: React.FC<{ className?: string }>;
  accentColor: string;
  showParams: boolean;
  paramDetails: { label: string; value: string }[];
}

const ModelBranchCard: React.FC<ModelBranchCardProps> = ({
  primaryTitle,
  techBadge,
  description,
  score,
  Icon,
  accentColor,
  showParams,
  paramDetails,
}) => {
  const riskColor = getRiskColor(score);
  const riskLabel = getRiskLabel(score);
  const pct = Math.round(score * 100);
  return (
    <div className="bg-[#12141c] p-5 rounded-xl border border-[#1e2330] hover:border-[#2a3050] transition-all space-y-4">
      <div className="flex items-start gap-3">
        <div className="p-2 rounded-lg flex-shrink-0" style={{ backgroundColor: `${accentColor}15`, border: `1px solid ${accentColor}30` }}>
          <Icon className="w-4 h-4" style={{ color: accentColor } as React.CSSProperties} />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-bold text-white leading-tight">{primaryTitle}</p>
          <span className="inline-block mt-1 text-[10px] font-bold px-2 py-0.5 rounded font-mono" style={{ backgroundColor: `${accentColor}12`, color: accentColor, border: `1px solid ${accentColor}25` }}>
            {techBadge}
          </span>
        </div>
      </div>
      <p className="text-[11px] text-gray-400 leading-relaxed">{description}</p>
      <div>
        <div className="flex justify-between items-center mb-1.5">
          <span className="text-[10px] text-gray-500 font-mono uppercase">Risk Score</span>
          <span className="text-sm font-extrabold font-mono" style={{ color: riskColor }}>{pct}%</span>
        </div>
        <div className="h-2 w-full bg-[#1e2330] rounded-full overflow-hidden">
          <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, backgroundColor: riskColor }} />
        </div>
        <div className="mt-2 flex items-center justify-between">
          <span className="text-[9px] font-bold px-1.5 py-0.5 rounded uppercase tracking-wider" style={{ backgroundColor: `${riskColor}15`, color: riskColor, border: `1px solid ${riskColor}30` }}>{riskLabel}</span>
          <div className="flex gap-2 text-[10px] text-gray-500 font-mono">
            <span>Threshold: <span className="text-gray-300">0.35</span></span>
            <span>|</span>
            <span>Raw: <span className="text-gray-300">{score.toFixed(3)}</span></span>
          </div>
        </div>
      </div>
      {showParams && (
        <div className="pt-2 border-t border-[#1e2330] grid grid-cols-2 gap-2">
          {paramDetails.map((p) => (
            <div key={p.label} className="bg-[#0b0c10] rounded-lg p-2 border border-[#1e2330]">
              <div className="text-[9px] text-gray-500 uppercase font-mono">{p.label}</div>
              <div className="text-[11px] text-gray-200 font-mono font-bold mt-0.5">{p.value}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// SHAP Diverging Tooltip
const ShapTip: React.FC<{ active?: boolean; payload?: Array<{ value: number; payload: ShapFeatureEntry }> }> = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0];
  const isPositive = d.value >= 0;
  return (
    <div className="bg-[#0d0f17] border border-[#1e2330] rounded-lg px-3 py-2 text-xs shadow-xl space-y-1">
      <p className="font-bold text-white font-mono">{d.payload.displayName}</p>
      <p className="text-gray-400 font-mono">
        Attribution Value: <span className={`font-bold ${isPositive ? 'text-emerald-400' : 'text-rose-400'}`}>{isPositive ? `+${d.value.toFixed(3)}` : d.value.toFixed(3)}</span>
      </p>
      <p className={`text-[10px] font-bold ${isPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
        {isPositive ? '▲ Positive attribution (Raises market integrity score)' : '▼ Negative attribution (Signals manipulation risk / stress)'}
      </p>
    </div>
  );
};

const TimelineTip: React.FC<{ active?: boolean; payload?: Array<{ dataKey: string; value: number; color: string }>; label?: string }> = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#0d0f17] border border-[#1e2330] rounded-lg px-3 py-2.5 text-xs shadow-xl space-y-1">
      <p className="text-gray-400 font-mono font-bold">{label}</p>
      {payload.map((p) => (
        <p key={p.dataKey} style={{ color: p.color }} className="font-mono">
          {p.dataKey === 'integrityScore' ? 'Integrity: ' : p.dataKey === 'price' ? 'Price: ' : 'Volume: '}
          <span className="font-bold">{p.dataKey === 'price' ? `$${p.value.toLocaleString()}` : p.value}</span>
        </p>
      ))}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────
// HISTORICAL INTEGRITY EVIDENCE MODAL
// ─────────────────────────────────────────────────────────────
interface HistoricalEvidenceModalProps {
  item: HistoricalWindow | null;
  onClose: () => void;
}

const HistoricalIntegrityEvidenceModal: React.FC<HistoricalEvidenceModalProps> = ({ item, onClose }) => {
  if (!item) return null;
  const isHealthy = item.integrityOutcomeScore >= 80;
  const stateCfg = getStateConfig(item.historicalState);
  const StateIcon = stateCfg.Icon;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-[#0e1017] border border-[#232838] rounded-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto shadow-2xl shadow-indigo-500/10 p-6 space-y-6">
        {/* Header */}
        <div className="flex items-start justify-between border-b border-[#1e2330] pb-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="p-1.5 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                <GitBranch className="w-5 h-5" />
              </span>
              <div>
                <h3 className="text-lg font-bold text-white tracking-tight">
                  Microstructure Vector Similarity Evidence — {item.timestamp}
                </h3>
                <p className="text-xs text-gray-400">
                  Vector similarity analog match ({item.asset}) via FAISS Database
                </p>
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg bg-[#181b26] text-gray-400 hover:text-white hover:bg-[#252a3b] transition-all cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Similarity Score & Historical State Banner */}
        <div className="p-4 rounded-xl bg-[#121522] border border-[#23293c] flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="text-xs text-gray-400 font-medium">Vector Cosine Similarity</div>
            <div className="text-3xl font-extrabold text-indigo-400 font-mono mt-0.5">{item.similarityScore}% Match</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-400 font-medium mb-1">Historical State</div>
            <span className={`inline-flex items-center gap-1 text-xs font-bold px-3 py-1 rounded-lg ${stateCfg.bg} ${stateCfg.border} border`} style={{ color: stateCfg.color }}>
              <StateIcon className="w-3.5 h-3.5" />
              {item.historicalState}
            </span>
          </div>
          <div className="text-right">
            <div className="text-xs text-gray-400 font-medium">Historical Outcome (+1H Horizon)</div>
            <div className={`text-xl font-bold font-mono mt-0.5 flex items-center justify-end gap-1 ${isHealthy ? 'text-emerald-400' : 'text-rose-400'}`}>
              <ShieldCheck className="w-5 h-5" />
              <span>Score: {item.integrityOutcomeScore}/100</span>
            </div>
          </div>
        </div>

        {/* Feature Vector Match Breakdown Table */}
        <div className="space-y-3">
          <h4 className="text-xs font-bold uppercase tracking-wider text-gray-400">
            Microstructure Feature Vector Alignment Breakdown
          </h4>
          <div className="rounded-xl border border-[#1e2330] overflow-hidden bg-[#0a0c12]">
            <table className="w-full text-xs text-left">
              <thead className="bg-[#121522] text-gray-400 border-b border-[#1e2330]">
                <tr>
                  <th className="p-3 font-semibold">Feature Dimension</th>
                  <th className="p-3 font-semibold">Current State</th>
                  <th className="p-3 font-semibold">Historical Match</th>
                  <th className="p-3 font-semibold text-right">Alignment</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1e2330] text-gray-300 font-mono text-[11px]">
                <tr>
                  <td className="p-3 font-medium text-white font-sans">Bid-Ask Spread Deviation</td>
                  <td className="p-3 text-emerald-400">{VECTOR_QUERY_INTEGRITY_STATE.spread}</td>
                  <td className="p-3 text-emerald-400">{item.conditions.spread}</td>
                  <td className="p-3 text-right text-emerald-400 font-bold">96%</td>
                </tr>
                <tr>
                  <td className="p-3 font-medium text-white font-sans">Order Book Imbalance</td>
                  <td className="p-3 text-emerald-400">{VECTOR_QUERY_INTEGRITY_STATE.obImbalance}</td>
                  <td className="p-3 text-emerald-400">{item.conditions.obImbalance}</td>
                  <td className="p-3 text-right text-emerald-400 font-bold">94%</td>
                </tr>
                <tr>
                  <td className="p-3 font-medium text-white font-sans">Rush Order Density</td>
                  <td className="p-3 text-indigo-400">{VECTOR_QUERY_INTEGRITY_STATE.rushOrderDensity}</td>
                  <td className="p-3 text-indigo-400">{item.conditions.rushOrderDensity}</td>
                  <td className="p-3 text-right text-indigo-400 font-bold">91%</td>
                </tr>
                <tr>
                  <td className="p-3 font-medium text-white font-sans">Volume Spike Index</td>
                  <td className="p-3 text-blue-400">{VECTOR_QUERY_INTEGRITY_STATE.vpinToxicity}</td>
                  <td className="p-3 text-blue-400">{item.conditions.volumeRatio}</td>
                  <td className="p-3 text-right text-emerald-400 font-bold">93%</td>
                </tr>
                <tr>
                  <td className="p-3 font-medium text-white font-sans">Cancellation Rate (Spoofing)</td>
                  <td className="p-3 text-amber-400">{VECTOR_QUERY_INTEGRITY_STATE.cancelVelocity}</td>
                  <td className="p-3 text-amber-400">{item.conditions.cancelRate}</td>
                  <td className="p-3 text-right text-amber-400 font-bold">89%</td>
                </tr>
                <tr>
                  <td className="p-3 font-medium text-white font-sans">T-GNN Relational Coupling</td>
                  <td className="p-3 text-purple-400">{VECTOR_QUERY_INTEGRITY_STATE.graphCoupling}</td>
                  <td className="p-3 text-purple-400">{item.conditions.graphCoupling}</td>
                  <td className="p-3 text-right text-purple-400 font-bold">95%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Post-Event Reaction Path */}
        <div className="space-y-3">
          <h4 className="text-xs font-bold uppercase tracking-wider text-gray-400">
            Post-Event Microstructure Evolution &amp; Reaction Path
          </h4>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="p-3.5 rounded-lg bg-[#0d0f17] border border-[#1e2330] flex flex-col justify-between min-h-[90px]">
              <span className="text-[10px] text-indigo-400 uppercase font-mono font-bold block mb-1">T+15 Minutes</span>
              <p className="text-[11px] text-gray-300 leading-relaxed break-words whitespace-normal">{item.postEventReaction.t15m}</p>
            </div>
            <div className="p-3.5 rounded-lg bg-[#0d0f17] border border-[#1e2330] flex flex-col justify-between min-h-[90px]">
              <span className="text-[10px] text-purple-400 uppercase font-mono font-bold block mb-1">T+1 Hour</span>
              <p className="text-[11px] text-gray-300 leading-relaxed break-words whitespace-normal">{item.postEventReaction.t1h}</p>
            </div>
            <div className="p-3.5 rounded-lg bg-[#0d0f17] border border-[#1e2330] flex flex-col justify-between min-h-[90px]">
              <span className="text-[10px] text-emerald-400 uppercase font-mono font-bold block mb-1">T+4 Hours</span>
              <p className="text-[11px] text-gray-300 leading-relaxed break-words whitespace-normal">{item.postEventReaction.t4h}</p>
            </div>
          </div>
        </div>

        {/* Narrative Synthesis */}
        <div className="p-4 rounded-xl bg-[#121520] border border-[#1e2330] space-y-2">
          <div className="flex items-center gap-2 text-xs font-bold text-indigo-400">
            <Cpu className="w-4 h-4" />
            <span>Agentic Synthesis &amp; Microstructure Forensics</span>
          </div>
          <p className="text-xs text-gray-300 leading-relaxed break-words whitespace-normal">{item.notes}</p>
          <div className="text-[11px] text-gray-400 font-mono pt-1">
            Embedding Euclidean distance (L2): <span className="text-indigo-300 font-bold">d = {item.vectorDistance.toFixed(3)}</span> • Event Classification: <span className="text-white font-bold">{item.eventPattern}</span>
          </div>
        </div>

        {/* Mandatory Research Notice */}
        <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-start gap-3">
          <ShieldAlert className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
          <p className="text-xs text-amber-300/90 leading-normal font-medium break-words whitespace-normal">
            <strong>Important Research Notice:</strong> Historical microstructure similarity provides analog context for the agentic reasoning pipeline but does not guarantee execution outcomes. All embeddings are mapped in 47-dimensional latent feature space.
          </p>
        </div>

        {/* Modal Action Footer */}
        <div className="flex justify-end pt-2">
          <button
            onClick={onClose}
            className="px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold transition-all shadow-md cursor-pointer"
          >
            Close Evidence View
          </button>
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────
// 5 CONSOLIDATED SUB-TAB VIEWS
// ─────────────────────────────────────────────────────────────

// TAB 1: OVERVIEW (Hero KPIs + AI Box + Live Threat Monitors + Ensemble Banner + Timeline/Gauge)
const OverviewSubTab: React.FC<{ data: IntegrityDataPayload; selectedAsset: AssetMode; selectedTimeframe: TimeframeMode }> = ({
  data,
  selectedAsset,
  selectedTimeframe,
}) => {
  return (
    <div className="space-y-6">
      {/* 4-Column Hero KPI Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <C2StyleMetricCard
          title="Composite Integrity Score"
          value={`${data.integrityScore}/100`}
          subtext="Ensemble Weighted Assessment"
          icon={<ShieldCheck className="w-4 h-4 text-emerald-400" />}
          badge={<StateBadge state={data.integrityState} size="sm" />}
        />

        <C2StyleMetricCard
          title="Unseen Anomaly Rate"
          value={data.unseenAnomalyRate.toFixed(2)}
          status="Reconstruction OK"
          subtext="Autoencoder Residual Threshold"
          icon={<Activity className="w-4 h-4 text-blue-400" />}
        />

        <C2StyleMetricCard
          title="Known Pattern Risk"
          value={`${(data.knownPatternRisk * 100).toFixed(2)}%`}
          status="Risk Low"
          subtext="Wash Trading / P&D Classifier"
          icon={<TrendingUp className="w-4 h-4 text-amber-400" />}
        />

        <C2StyleMetricCard
          title="Model Confidence (Conformal)"
          value={`${data.modelConfidence.toFixed(1)}%`}
          status="Coverage CP-95"
          subtext="Non-Conformity Interval Set"
          icon={<Target className="w-4 h-4 text-indigo-400" />}
        />
      </div>

      {/* Agentic AI Integrity Reasoning Synthesis Box */}
      <div className="glass-panel p-6 rounded-xl border border-indigo-500/30 glow-indigo relative overflow-hidden bg-gradient-to-r from-[#121626] via-[#12141c] to-[#151224] space-y-4">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white tracking-tight">
              Agentic Market Integrity Reasoning Synthesis
            </h2>
            <span className="text-[11px] text-indigo-400 font-mono">
              Model Ensemble: Autoencoder + XGBoost + Transformer + T-GNN
            </span>
          </div>
        </div>

        <p className="text-sm text-gray-200 leading-relaxed font-normal bg-[#0a0c14]/50 p-4 rounded-xl border border-[#1e2330]">
          "{data.diagnosticSummary}"
        </p>

        <div className="pt-3 border-t border-[#1e2330]">
          <span className="text-xs font-bold text-gray-400 uppercase tracking-wider block mb-2.5">
            Key Microstructure Drivers
          </span>
          <div className="flex flex-wrap items-center gap-2.5">
            {data.keyDrivers.map((kd) => (
              <div
                key={kd.label}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#0d0f17] border border-[#1e2330] text-xs"
              >
                <span className="font-semibold text-gray-300">{kd.label}:</span>
                <span className="font-bold px-2 py-0.5 rounded text-[11px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  {kd.value}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── 1. LIVE MICROSTRUCTURE THREAT MONITORS (4-Card Grid) ─── */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-indigo-400" />
            <h3 className="text-xs font-bold text-white uppercase tracking-wider font-mono">
              Live Microstructure Threat Monitors &amp; Threshold Radar
            </h3>
          </div>
          <span className="text-[10px] text-indigo-400 font-mono font-bold px-2 py-0.5 rounded bg-indigo-500/10 border border-indigo-500/20">
            Real-Time Anomaly Interceptor
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-[#12141c] border border-[#1e2330] p-4 rounded-xl shadow-md space-y-2 hover:border-[#2a3050] transition-all">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">Wash Trading &amp; Self-Flow</span>
              <div className="p-1.5 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
                <Flame className="w-3.5 h-3.5 text-emerald-400" />
              </div>
            </div>
            <div className="flex items-baseline justify-between gap-2">
              <div className="text-2xl font-black font-mono text-white tracking-tight">0.02%</div>
              <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-bold font-mono text-[10px]">
                CLEAN FLOW
              </span>
            </div>
            <div className="text-[11px] pt-1 border-t border-[#1e2330] text-gray-400 leading-snug">
              V-PIN Toxicity: <span className="text-white font-mono font-bold">0.14</span> | Low self-matched trade probability
            </div>
          </div>

          <div className="bg-[#12141c] border border-[#1e2330] p-4 rounded-xl shadow-md space-y-2 hover:border-[#2a3050] transition-all">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">Spoofing &amp; Cancel Velocity</span>
              <div className="p-1.5 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
                <Layers className="w-3.5 h-3.5 text-blue-400" />
              </div>
            </div>
            <div className="flex items-baseline justify-between gap-2">
              <div className="text-2xl font-black font-mono text-white tracking-tight">1.8%</div>
              <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-bold font-mono text-[10px]">
                NORMAL DEPTH
              </span>
            </div>
            <div className="text-[11px] pt-1 border-t border-[#1e2330] text-gray-400 leading-snug">
              Phantom liquidity ratio &lt; 0.05 | No rapid order layering detected
            </div>
          </div>

          <div className="bg-[#12141c] border border-[#1e2330] p-4 rounded-xl shadow-md space-y-2 hover:border-[#2a3050] transition-all">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">Pump &amp; Dump Momentum</span>
              <div className="p-1.5 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
                <TrendingUp className="w-3.5 h-3.5 text-amber-400" />
              </div>
            </div>
            <div className="flex items-baseline justify-between gap-2">
              <div className="text-2xl font-black font-mono text-white tracking-tight">1.04×</div>
              <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-bold font-mono text-[10px]">
                NO SPIKE
              </span>
            </div>
            <div className="text-[11px] pt-1 border-t border-[#1e2330] text-gray-400 leading-snug">
              Volume-to-spread divergence within normal variance
            </div>
          </div>

          <div className="bg-[#12141c] border border-[#1e2330] p-4 rounded-xl shadow-md space-y-2 hover:border-[#2a3050] transition-all">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">Liquidity Withdrawal Risk</span>
              <div className="p-1.5 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
                <ShieldCheck className="w-3.5 h-3.5 text-indigo-400" />
              </div>
            </div>
            <div className="flex items-baseline justify-between gap-2">
              <div className="text-2xl font-black font-mono text-emerald-400 tracking-tight">LOW RISK</div>
              <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-bold font-mono text-[10px]">
                BID-ASK STABLE
              </span>
            </div>
            <div className="text-[11px] pt-1 border-t border-[#1e2330] text-gray-400 leading-snug">
              Top-5 order book depth intact across BTC/USDT &amp; ETH/USDT
            </div>
          </div>
        </div>

        {/* ── 2. ENSEMBLE MODEL CONSENSUS BANNER ─── */}
        <div className="p-3.5 rounded-xl bg-[#0d0f17] border border-indigo-500/20 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Zap className="w-4 h-4 text-indigo-400" />
            <span className="text-xs font-bold text-white font-mono tracking-wide">
              4-BRANCH ENSEMBLE CONSENSUS: <span className="text-emerald-400">96.8% AGREEMENT</span>
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-3 text-xs font-mono">
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#12141c] border border-[#1e2330]">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              <span className="text-gray-300">Autoencoder:</span>
              <span className="text-emerald-400 font-bold">OK (0.08)</span>
            </div>
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#12141c] border border-[#1e2330]">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              <span className="text-gray-300">XGBoost:</span>
              <span className="text-emerald-400 font-bold">OK (0.04)</span>
            </div>
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#12141c] border border-[#1e2330]">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              <span className="text-gray-300">Transformer:</span>
              <span className="text-emerald-400 font-bold">OK (0.09)</span>
            </div>
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#12141c] border border-[#1e2330]">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              <span className="text-gray-300">T-GNN:</span>
              <span className="text-emerald-400 font-bold">OK (0.06)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Dual-Axis Chart & Market Integrity State Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 glass-panel p-5 rounded-xl border border-[#1e2330] space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Activity className="w-4 h-4 text-indigo-400" />
                Real-Time Integrity Score &amp; Price Timeline ({selectedAsset} • {selectedTimeframe})
              </h3>
              <span className="text-[10px] text-gray-400 font-mono">Dual-Axis • Anomaly Bands Highlighted</span>
            </div>
            <span className="px-2.5 py-1 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 text-xs font-mono font-bold">
              1-Min Polling
            </span>
          </div>

          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={data.timeline} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2330" />
                <XAxis dataKey="time" tick={{ fill: '#6b7280', fontSize: 10, fontFamily: 'monospace' }} axisLine={{ stroke: '#1e2330' }} tickLine={false} />
                <YAxis yAxisId="integrity" domain={[0, 100]} tick={{ fill: '#818cf8', fontSize: 10, fontFamily: 'monospace' }} axisLine={false} tickLine={false} width={35} />
                <YAxis yAxisId="price" orientation="right" domain={['auto', 'auto']} tick={{ fill: '#60a5fa', fontSize: 10, fontFamily: 'monospace' }} axisLine={false} tickLine={false} width={55} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
                <ReferenceLine yAxisId="integrity" y={65} stroke="#ef4444" strokeDasharray="4 2" strokeWidth={1} label={{ value: 'Watch Threshold (65)', position: 'insideTopRight', fill: '#ef4444', fontSize: 9 }} />
                <ReTooltip content={<TimelineTip />} />
                <Area yAxisId="integrity" type="monotone" dataKey="integrityScore" stroke="#818cf8" fill="#818cf825" strokeWidth={2} dot={false} />
                <Line yAxisId="price" type="monotone" dataKey="price" stroke="#60a5fa" strokeWidth={1.5} dot={false} strokeDasharray="3 2" />
                <Legend formatter={(value) => <span className="text-[10px] text-gray-400 font-mono">{value === 'integrityScore' ? 'Integrity Score (0-100)' : 'Asset Price ($)'}</span>} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Right Side State Gauge & Microstructure Quick Panel */}
        <div className="glass-panel p-5 rounded-xl border border-[#1e2330] flex flex-col justify-between space-y-4">
          <div>
            <div className="flex items-center justify-between border-b border-[#1e2330] pb-3 mb-4">
              <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                Integrity State
              </h3>
              <StateBadge state={data.integrityState} size="sm" />
            </div>

            <div className="p-4 rounded-xl bg-[#0d0f17] border border-[#1e2330] text-center mb-4">
              <span className="text-xs text-gray-400 block uppercase font-medium">Confidence Score</span>
              <div className="text-3xl font-black text-emerald-400 font-mono mt-1">
                {data.modelConfidence}%
              </div>
              <div className="w-full bg-[#1e2330] h-1.5 rounded-full mt-3 overflow-hidden">
                <div
                  className="bg-emerald-400 h-full rounded-full"
                  style={{ width: `${data.modelConfidence}%` }}
                />
              </div>
            </div>

            <div className="space-y-2">
              <span className="text-xs font-bold text-gray-400 uppercase tracking-wider block">
                Microstructure Diagnostics
              </span>
              {[
                { label: 'Order Book Imbalance', val: data.marketBanner.orderBookImbalance, ok: true },
                { label: 'Bid-Ask Spread', val: data.marketBanner.spread, ok: true },
                { label: 'Wash Trade Ratio', val: '0.02%', ok: true },
                { label: 'Spoofing Cancel Burst', val: '0 / min', ok: true },
              ].map((f) => (
                <div key={f.label} className="flex items-center justify-between p-2 rounded-lg bg-[#0d0f17] border border-[#1e2330] text-xs">
                  <span className="text-gray-300 font-medium">{f.label}</span>
                  <span className="font-bold text-emerald-400 font-mono">{f.val}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="p-3 rounded-lg bg-[#12141c] border border-[#1e2330] text-[11px] text-gray-400 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>Integrity validated by 4 independent mathematical branches.</span>
          </div>
        </div>
      </div>
    </div>
  );
};

// TAB 2: MICROSTRUCTURE INTELLIGENCE (MERGED BTC & ETH WITH INTERNAL SWITCHER)
const MicrostructureIntelligenceSubTab: React.FC = () => {
  const [activeAsset, setActiveAsset] = useState<'BTC' | 'ETH'>('BTC');

  // BTC Data
  const obLevels = [
    { level: 'L1', bidVol: 14.2, askVol: 12.8 },
    { level: 'L2', bidVol: 28.5, askVol: 24.1 },
    { level: 'L3', bidVol: 42.1, askVol: 39.8 },
    { level: 'L4', bidVol: 56.4, askVol: 51.2 },
    { level: 'L5', bidVol: 78.9, askVol: 70.4 },
  ];

  // ETH Data
  const ethRushData = [
    { time: '10:00', rushOrders: 3, cancelRate: 1.2 },
    { time: '10:05', rushOrders: 4, cancelRate: 1.5 },
    { time: '10:10', rushOrders: 2, cancelRate: 1.1 },
    { time: '10:15', rushOrders: 6, cancelRate: 2.1 },
    { time: '10:20', rushOrders: 3, cancelRate: 1.4 },
    { time: '10:25', rushOrders: 2, cancelRate: 1.2 },
  ];

  return (
    <div className="space-y-6">
      {/* Asset Switcher Pill Bar */}
      <div className="glass-panel p-4 rounded-xl border border-indigo-500/30 bg-gradient-to-r from-[#0f1428] via-[#0d0f1c] to-[#141226] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-indigo-400" />
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">
              Microstructure Depth &amp; Order Flow Intelligence
            </h3>
            <p className="text-[11px] text-gray-400">
              L2 order book imbalances, cancellation velocities, and VPIN toxicity analysis
            </p>
          </div>
        </div>

        {/* Toggle between BTC & ETH */}
        <div className="flex items-center bg-[#0b0c10] border border-[#1f2330] rounded-lg p-1 text-xs">
          <button
            onClick={() => setActiveAsset('BTC')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-semibold transition-all cursor-pointer ${
              activeAsset === 'BTC' ? 'bg-amber-600 text-white shadow-sm' : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <Coins className="w-3.5 h-3.5" />
            BTC/USDT Microstructure
          </button>
          <button
            onClick={() => setActiveAsset('ETH')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-semibold transition-all cursor-pointer ${
              activeAsset === 'ETH' ? 'bg-indigo-600 text-white shadow-sm' : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            ETH/USDT Microstructure
          </button>
        </div>
      </div>

      {activeAsset === 'BTC' ? (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <C2StyleMetricCard title="BTC Bid-Ask Spread" value="$1.20" status="Normal (+0.04)" subtext="Optimal Liquidity Spread" icon={<Coins className="w-4 h-4 text-amber-400" />} />
            <C2StyleMetricCard title="BTC Order Book Imbalance" value="+4.2%" status="Bid-Heavy" subtext="Depth Imbalance Ratio" icon={<Activity className="w-4 h-4 text-emerald-400" />} />
            <C2StyleMetricCard title="Volume Toxicity (VPIN)" value="0.14" status="Low Toxicity" subtext="Order Flow Asymmetry" icon={<Flame className="w-4 h-4 text-blue-400" />} />
            <C2StyleMetricCard title="Spoofing Index" value="0.02%" status="0 Alerts" subtext="Rapid Cancellation Rate" icon={<ShieldCheck className="w-4 h-4 text-indigo-400" />} />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="glass-panel p-5 rounded-xl border border-[#1e2330] space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                  <BarChart2 className="w-4 h-4 text-amber-400" />
                  BTC Order Book Depth (Bid vs Ask Levels L1-L5)
                </h3>
                <span className="text-[10px] text-gray-400 font-mono">Volume in BTC</span>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={obLevels} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e2330" />
                    <XAxis dataKey="level" tick={{ fill: '#6b7280', fontSize: 10, fontFamily: 'monospace' }} axisLine={{ stroke: '#1e2330' }} tickLine={false} />
                    <YAxis tick={{ fill: '#6b7280', fontSize: 10, fontFamily: 'monospace' }} axisLine={false} tickLine={false} />
                    <ReTooltip contentStyle={{ backgroundColor: '#0d0f17', borderColor: '#1e2330', borderRadius: '8px', fontSize: '11px' }} />
                    <Bar dataKey="bidVol" name="Bid Depth (BTC)" fill="#22c55e" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="askVol" name="Ask Depth (BTC)" fill="#ef4444" radius={[4, 4, 0, 0]} />
                    <Legend formatter={(v) => <span className="text-[11px] text-gray-300 font-mono">{v}</span>} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="glass-panel p-5 rounded-xl border border-[#1e2330] space-y-4">
              <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                BTC Microstructure Integrity Assessment
              </h3>
              <div className="space-y-3">
                {[
                  { title: 'Quote-to-Trade Ratio', desc: 'Ratio of placed limit quotes to executed trades. Values > 50 flag potential layering.', val: '14.2 (Normal)' },
                  { title: 'Effective vs Quoted Spread', desc: 'Execution slippage comparison against top-of-book quotes.', val: '$1.22 vs $1.20' },
                  { title: 'Large Lot Wash Identifier', desc: 'Cross-exchange synchronized non-economic roundtrip transactions.', val: '0 Detected' },
                  { title: 'Liquidity Withdrawal Pulse', desc: 'Sudden pull of bid depth within 200ms prior to major market orders.', val: 'Clean (0.00%)' },
                ].map((sig) => (
                  <div key={sig.title} className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330] flex items-center justify-between">
                    <div>
                      <span className="text-xs font-bold text-gray-200">{sig.title}</span>
                      <p className="text-[10px] text-gray-400 mt-0.5">{sig.desc}</p>
                    </div>
                    <span className="px-2.5 py-1 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs font-mono font-bold whitespace-nowrap ml-4">
                      {sig.val}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <C2StyleMetricCard title="ETH Rush Order Density" value="2.1%" status="Normal (Low)" subtext="Clustered Market Order Bursts" icon={<Zap className="w-4 h-4 text-purple-400" />} />
            <C2StyleMetricCard title="ETH Cancellation Rate" value="1.8%" status="Clean Flow" subtext="Order Lifespan < 500ms" icon={<Activity className="w-4 h-4 text-emerald-400" />} />
            <C2StyleMetricCard title="DEX / CEX Flow Imbalance" value="-1.4%" status="Organic Arb" subtext="Cross-Venue Toxic Flow" icon={<Layers className="w-4 h-4 text-indigo-400" />} />
            <C2StyleMetricCard title="Gas-Correlated Stress" value="18 Gwei" status="Low Gas Stress" subtext="Mev / Priority Fee Spikes" icon={<Flame className="w-4 h-4 text-amber-400" />} />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="glass-panel p-5 rounded-xl border border-[#1e2330] space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                  <Activity className="w-4 h-4 text-purple-400" />
                  ETH Rush Orders &amp; Cancellation Dynamics
                </h3>
                <span className="text-[10px] text-gray-400 font-mono">1m Rolling Windows</span>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={ethRushData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e2330" />
                    <XAxis dataKey="time" tick={{ fill: '#6b7280', fontSize: 10, fontFamily: 'monospace' }} axisLine={{ stroke: '#1e2330' }} tickLine={false} />
                    <YAxis tick={{ fill: '#6b7280', fontSize: 10, fontFamily: 'monospace' }} axisLine={false} tickLine={false} />
                    <ReTooltip contentStyle={{ backgroundColor: '#0d0f17', borderColor: '#1e2330', borderRadius: '8px', fontSize: '11px' }} />
                    <Line type="monotone" dataKey="rushOrders" name="Rush Order Bursts" stroke="#a78bfa" strokeWidth={2} dot={{ r: 3 }} />
                    <Line type="monotone" dataKey="cancelRate" name="Cancel Rate (%)" stroke="#34d399" strokeWidth={2} dot={{ r: 3 }} />
                    <Legend formatter={(v) => <span className="text-[11px] text-gray-300 font-mono">{v}</span>} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="glass-panel p-5 rounded-xl border border-[#1e2330] space-y-4">
              <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Layers className="w-4 h-4 text-indigo-400" />
                ETH Microstructure Anomaly Metrics
              </h3>
              <div className="space-y-3">
                {[
                  { title: 'MEV Sandwich Extraction Index', desc: 'Blocks with suspected sandwiching or priority gas auction frontrunning.', val: '0.00% (Clean)' },
                  { title: 'DEX Pool Liquidity Imbalance', desc: 'Uniswap v3 ETH/USDT pool tick concentration vs CEX top-of-book.', val: 'Aligned (0.08% spread)' },
                  { title: 'Rapid Re-quote Frequency', desc: 'Market maker spread adjustment velocity per 100ms interval.', val: '42 / sec (Optimal)' },
                  { title: 'Cross-Asset Lead-Lag Skew', desc: 'Price discovery lead latency of ETH/USDT relative to BTC/USDT.', val: '12ms Lead (Normal)' },
                ].map((sig) => (
                  <div key={sig.title} className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330] flex items-center justify-between">
                    <div>
                      <span className="text-xs font-bold text-gray-200">{sig.title}</span>
                      <p className="text-[10px] text-gray-400 mt-0.5">{sig.desc}</p>
                    </div>
                    <span className="px-2.5 py-1 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs font-mono font-bold whitespace-nowrap ml-4">
                      {sig.val}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// TAB 3: ML MODEL EVIDENCE HUB (COMBINES 4-BRANCH, T-GNN GRAPH, AND HORIZON PREDICTIONS)
const MLEvidenceHubSubTab: React.FC<{ data: IntegrityDataPayload }> = ({ data }) => {
  const [hubView, setHubView] = useState<'4-branch' | 'tgnn-graph' | 'predictions'>('4-branch');
  const [showParams, setShowParams] = useState(false);

  // GNN state
  const [selectedNode, setSelectedNode] = useState<string>('order-book');
  const nodes = [
    { id: 'order-book', label: 'Order Book Imbalance', x: 50, y: 15, color: '#818cf8', weight: '0.34', degree: '4', desc: 'Depth ratio between bid and ask sides' },
    { id: 'liquidity', label: 'Liquidity Depth', x: 85, y: 42, color: '#34d399', weight: '0.28', degree: '3', desc: 'Aggregate volume within 1% of mid-price' },
    { id: 'trade-flow', label: 'Trade Flow', x: 70, y: 78, color: '#f59e0b', weight: '0.22', degree: '3', desc: 'Taker buy vs taker sell execution volume' },
    { id: 'volume', label: 'Volume Acceleration', x: 30, y: 78, color: '#f97316', weight: '0.19', degree: '3', desc: 'Exponential moving volume surge index' },
    { id: 'price', label: 'Asset Price', x: 15, y: 42, color: '#60a5fa', weight: '0.25', degree: '4', desc: 'High-frequency mid-price return series' },
    { id: 'spread', label: 'Bid-Ask Spread', x: 50, y: 50, color: '#c084fc', weight: '0.31', degree: '4', desc: 'Top-of-book quoted spread width in USD' },
  ];
  const edges = [
    { from: 'order-book', to: 'liquidity', weight: 0.88, status: 'Normal Coupling' },
    { from: 'order-book', to: 'spread', weight: 0.92, status: 'Normal Coupling' },
    { from: 'order-book', to: 'price', weight: 0.84, status: 'Normal Coupling' },
    { from: 'liquidity', to: 'trade-flow', weight: 0.79, status: 'Normal Coupling' },
    { from: 'trade-flow', to: 'volume', weight: 0.86, status: 'Normal Coupling' },
    { from: 'volume', to: 'price', weight: 0.81, status: 'Normal Coupling' },
    { from: 'price', to: 'spread', weight: 0.89, status: 'Normal Coupling' },
    { from: 'spread', to: 'liquidity', weight: 0.91, status: 'Normal Coupling' },
  ];
  const activeNodeData = nodes.find((n) => n.id === selectedNode) || nodes[0];

  // Predictions state
  const [horizon, setHorizon] = useState<'1m' | '5m' | '15m' | '1H'>('5m');
  const horizonPredictions = [
    { window: 'T+1m', anomalyProb: 0.04, integrityForecast: 92, riskLevel: 'LOW', confidence: 96 },
    { window: 'T+5m', anomalyProb: 0.06, integrityForecast: 91, riskLevel: 'LOW', confidence: 94 },
    { window: 'T+15m', anomalyProb: 0.09, integrityForecast: 89, riskLevel: 'LOW', confidence: 91 },
    { window: 'T+30m', anomalyProb: 0.12, integrityForecast: 87, riskLevel: 'LOW', confidence: 88 },
    { window: 'T+1H', anomalyProb: 0.15, integrityForecast: 85, riskLevel: 'WATCH', confidence: 85 },
  ];
  const forecastCurve = [
    { step: 'T+0', predictedProb: 4 },
    { step: 'T+5m', predictedProb: 6 },
    { step: 'T+10m', predictedProb: 7 },
    { step: 'T+15m', predictedProb: 9 },
    { step: 'T+20m', predictedProb: 11 },
    { step: 'T+30m', predictedProb: 12 },
    { step: 'T+45m', predictedProb: 14 },
    { step: 'T+60m', predictedProb: 15 },
  ];

  const branches = [
    {
      primaryTitle: 'Unseen Abnormal Behavior Evidence',
      techBadge: 'Model Branch: Autoencoder (Reconstruction Error)',
      description: 'Measures how far current market dynamics diverge from learned normal market patterns. A high reconstruction error signals potential unseen manipulation not matching known attack templates.',
      score: data.modelBranches.autoencoderAnomalyScore,
      Icon: Activity,
      accentColor: '#818cf8',
      params: [
        { label: 'Architecture', value: 'LSTM-AE' },
        { label: 'Latent Dim', value: '32' },
        { label: 'Threshold', value: '0.35' },
        { label: 'Window', value: '60 bars' }
      ],
    },
    {
      primaryTitle: 'Known Manipulation Pattern Evidence',
      techBadge: 'Model Branch: XGBoost (Supervised Classifier)',
      description: 'Evaluates matching probabilities for known wash trading and pump-and-dump scenarios using 47 engineered microstructure features. Trained on labeled manipulation event archives.',
      score: data.modelBranches.xgboostPatternRisk,
      Icon: TrendingUp,
      accentColor: '#f59e0b',
      params: [
        { label: 'Estimators', value: '300' },
        { label: 'Max Depth', value: '6' },
        { label: 'Features', value: '47' },
        { label: 'Classes', value: 'WashTrade, P&D, Normal' }
      ],
    },
    {
      primaryTitle: 'Temporal Sequence Abnormality Evidence',
      techBadge: 'Model Branch: Transformer (Multi-Window Sequence)',
      description: 'Detects unusual temporal behavior evolution across consecutive 1-minute time windows. Self-attention heads identify coordinated sequential order-flow patterns typical of manipulation.',
      score: data.modelBranches.transformerTemporalScore,
      Icon: Cpu,
      accentColor: '#a78bfa',
      params: [
        { label: 'Heads', value: '8' },
        { label: 'Layers', value: '4' },
        { label: 'Seq Len', value: '30 bars' },
        { label: 'd_model', value: '128' }
      ],
    },
    {
      primaryTitle: 'Relational Microstructure Network Evidence',
      techBadge: 'Model Branch: Temporal GNN (Within-Market Graph)',
      description: 'Analyzes structural shifts in order-book, volume, spread, and trade-flow relationships using a graph neural network. Detects anomalous edge-weight changes between market microstructure nodes.',
      score: data.modelBranches.gnnMicrostructureScore,
      Icon: Network,
      accentColor: '#34d399',
      params: [
        { label: 'Graph Type', value: 'Dynamic' },
        { label: 'GNN Layers', value: '3' },
        { label: 'Nodes', value: '6' },
        { label: 'Update', value: 'Per 1-min bar' }
      ],
    },
  ];

  return (
    <div className="space-y-6">
      {/* Internal ML Hub Switcher */}
      <div className="glass-panel p-4 rounded-xl border border-indigo-500/30 bg-gradient-to-r from-[#0f1428] via-[#0d0f1c] to-[#141226] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Cpu className="w-5 h-5 text-indigo-400" />
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">
              Machine Learning Evidence Hub
            </h3>
            <p className="text-[11px] text-gray-400">
              4-Branch Architecture, T-GNN Relational Network, and Rolling Horizon Predictions
            </p>
          </div>
        </div>

        <div className="flex items-center bg-[#0b0c10] border border-[#1f2330] rounded-lg p-1 text-xs">
          <button
            onClick={() => setHubView('4-branch')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-semibold transition-all cursor-pointer ${
              hubView === '4-branch' ? 'bg-indigo-600 text-white shadow-sm' : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <GitMerge className="w-3.5 h-3.5" />
            4-Branch Grid
          </button>
          <button
            onClick={() => setHubView('tgnn-graph')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-semibold transition-all cursor-pointer ${
              hubView === 'tgnn-graph' ? 'bg-indigo-600 text-white shadow-sm' : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <Network className="w-3.5 h-3.5" />
            T-GNN Graph Canvas
          </button>
          <button
            onClick={() => setHubView('predictions')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-semibold transition-all cursor-pointer ${
              hubView === 'predictions' ? 'bg-indigo-600 text-white shadow-sm' : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <Target className="w-3.5 h-3.5" />
            Horizon Predictions
          </button>
        </div>
      </div>

      {/* 1. 4-BRANCH MODEL GRID VIEW */}
      {hubView === '4-branch' && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-bold text-white uppercase tracking-wider font-mono flex items-center gap-2">
              <Network className="w-4 h-4 text-indigo-400" />
              4-Branch Model Evidence Grid
            </h4>
            <button
              onClick={() => setShowParams(!showParams)}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-bold border transition-all cursor-pointer ${
                showParams ? 'bg-indigo-600 text-white border-indigo-400/40' : 'bg-[#0d0f17] text-gray-400 border-[#1e2330] hover:text-gray-200'
              }`}
            >
              <Code2 className="w-3.5 h-3.5" />
              {showParams ? 'Hide' : 'Show'} Developer Model Parameters
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {branches.map((b) => (
              <ModelBranchCard
                key={b.primaryTitle}
                primaryTitle={b.primaryTitle}
                techBadge={b.techBadge}
                description={b.description}
                score={b.score}
                Icon={b.Icon}
                accentColor={b.accentColor}
                showParams={showParams}
                paramDetails={b.params}
              />
            ))}
          </div>
        </div>
      )}

      {/* 2. T-GNN GRAPH CANVAS VIEW */}
      {hubView === 'tgnn-graph' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <C2StyleMetricCard title="Graph Density Score" value="0.64" status="NORMAL TOPOLOGY" subtext="6 Microstructure Nodes • 8 Dynamic Edges" icon={<Network className="w-4 h-4 text-indigo-400" />} />
            <C2StyleMetricCard title="Relational Coupling Risk" value="0.06" status="LOW EDGE DISTORTION" subtext="GNN Anomaly Edge Shift Matrix" icon={<ShieldCheck className="w-4 h-4 text-emerald-400" />} />
            <C2StyleMetricCard title="Message-Passing Architecture" value="3-Layer GAT" status="TEMPORAL ATTENTION" subtext="Self-Attention Microstructure Embeddings" icon={<Cpu className="w-4 h-4 text-purple-400" />} />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 glass-panel p-5 rounded-xl border border-[#1e2330] space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Share2 className="w-4 h-4 text-indigo-400" />
                  <h4 className="text-xs font-bold text-white uppercase font-mono tracking-wider">
                    Interactive Microstructure Node-Edge Graph (T-GNN)
                  </h4>
                </div>
                <span className="text-[10px] text-gray-400 font-mono flex items-center gap-1">
                  <MousePointerClick className="w-3 h-3 text-indigo-400" /> Click node to inspect
                </span>
              </div>

              <div className="relative w-full bg-[#0b0c10] rounded-xl border border-[#1e2330] p-4" style={{ paddingBottom: '55%' }}>
                <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 90">
                  <defs>
                    <marker id="arrow-act" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
                      <path d="M 0 0 L 10 5 L 0 10 z" fill="#818cf8" />
                    </marker>
                    <marker id="arrow-def" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
                      <path d="M 0 0 L 10 5 L 0 10 z" fill="#374151" />
                    </marker>
                  </defs>

                  {edges.map((e, i) => {
                    const na = nodes.find((n) => n.id === e.from)!;
                    const nb = nodes.find((n) => n.id === e.to)!;
                    const isConnected = e.from === selectedNode || e.to === selectedNode;
                    return (
                      <line
                        key={i}
                        x1={na.x}
                        y1={na.y}
                        x2={nb.x}
                        y2={nb.y}
                        stroke={isConnected ? '#818cf8' : '#374151'}
                        strokeWidth={isConnected ? '1.5' : '0.8'}
                        strokeDasharray={isConnected ? 'none' : '2 1.5'}
                        markerEnd={isConnected ? 'url(#arrow-act)' : 'url(#arrow-def)'}
                        opacity={isConnected ? 1 : 0.6}
                      />
                    );
                  })}

                  {nodes.map((n) => {
                    const isSelected = n.id === selectedNode;
                    return (
                      <g key={n.id} onClick={() => setSelectedNode(n.id)} className="cursor-pointer transition-transform hover:scale-110">
                        {isSelected && (
                          <circle cx={n.x} cy={n.y} r="9" fill="none" stroke="#818cf8" strokeWidth="1.2" strokeDasharray="2 1" className="animate-spin" />
                        )}
                        <circle cx={n.x} cy={n.y} r={isSelected ? '7' : '6'} fill={`${n.color}25`} stroke={n.color} strokeWidth={isSelected ? '1.8' : '1'} />
                        <text x={n.x} y={n.y + 0.4} textAnchor="middle" dominantBaseline="middle" fontSize="2.4" fontWeight="bold" fill={n.color} fontFamily="monospace">
                          {n.label.split(' ')[0]}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>

              <div className="flex flex-wrap gap-2 pt-1">
                {nodes.map((n) => (
                  <button
                    key={n.id}
                    onClick={() => setSelectedNode(n.id)}
                    className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-mono font-bold transition-all cursor-pointer ${
                      selectedNode === n.id ? 'bg-indigo-600 text-white shadow-sm' : 'bg-[#0d0f17] text-gray-400 hover:text-gray-200 border border-[#1e2330]'
                    }`}
                  >
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: n.color }} />
                    {n.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="glass-panel p-5 rounded-xl border border-[#1e2330] space-y-4">
              <div className="flex items-center justify-between border-b border-[#1e2330] pb-3">
                <div>
                  <span className="text-[10px] text-gray-400 uppercase font-mono font-bold block">Selected Node Telemetry</span>
                  <h4 className="text-base font-bold text-white tracking-tight mt-0.5">{activeNodeData.label}</h4>
                </div>
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: activeNodeData.color }} />
              </div>

              <p className="text-xs text-gray-300 leading-relaxed">{activeNodeData.desc}</p>

              <div className="space-y-2.5">
                <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330] flex justify-between items-center text-xs">
                  <span className="text-gray-400 font-medium">Node Attention Weight:</span>
                  <span className="font-mono font-bold text-indigo-400">{activeNodeData.weight}</span>
                </div>
                <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330] flex justify-between items-center text-xs">
                  <span className="text-gray-400 font-medium">Degree Centrality:</span>
                  <span className="font-mono font-bold text-emerald-400">{activeNodeData.degree} Connected Edges</span>
                </div>
                <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330] flex justify-between items-center text-xs">
                  <span className="text-gray-400 font-medium">Coupling Status:</span>
                  <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono text-[10px] font-bold">
                    STABLE COUPLING
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 3. FORWARD HORIZON PREDICTIONS VIEW */}
      {hubView === 'predictions' && (
        <div className="space-y-6">
          <div className="flex items-center justify-between bg-[#12141c] p-4 rounded-xl border border-[#1e2330]">
            <div>
              <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Target className="w-4 h-4 text-indigo-400" />
                Integrity Risk Horizon Forecast Matrix
              </h3>
              <p className="text-[11px] text-gray-400 mt-0.5">Projected anomaly probabilities across forward micro-horizons</p>
            </div>
            <div className="flex items-center bg-[#0b0c10] border border-[#1f2330] rounded-lg p-1 text-xs">
              {(['1m', '5m', '15m', '1H'] as const).map((h) => (
                <button
                  key={h}
                  onClick={() => setHorizon(h)}
                  className={`px-3 py-1 rounded-md font-semibold transition-all cursor-pointer ${
                    horizon === h ? 'bg-indigo-600 text-white shadow-sm' : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  +{h} Horizon
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 glass-panel p-5 rounded-xl border border-[#1e2330] space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-bold text-gray-200 uppercase font-mono tracking-wider">
                  Projected Anomaly Probability Curve (%)
                </h4>
                <span className="text-[10px] text-emerald-400 font-mono font-bold bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                  Low Risk Regime
                </span>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={forecastCurve} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e2330" />
                    <XAxis dataKey="step" tick={{ fill: '#6b7280', fontSize: 10, fontFamily: 'monospace' }} axisLine={{ stroke: '#1e2330' }} tickLine={false} />
                    <YAxis domain={[0, 40]} tick={{ fill: '#6b7280', fontSize: 10, fontFamily: 'monospace' }} axisLine={false} tickLine={false} />
                    <ReferenceLine y={25} stroke="#ef4444" strokeDasharray="3 3" label={{ value: 'Elevated Risk Line (25%)', fill: '#ef4444', fontSize: 9 }} />
                    <ReTooltip contentStyle={{ backgroundColor: '#0d0f17', borderColor: '#1e2330', borderRadius: '8px', fontSize: '11px' }} />
                    <Line type="monotone" dataKey="predictedProb" name="Predicted Anomaly Risk (%)" stroke="#818cf8" strokeWidth={2} dot={{ r: 4, fill: '#818cf8' }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="glass-panel p-5 rounded-xl border border-[#1e2330] space-y-4">
              <h4 className="text-xs font-bold text-gray-200 uppercase font-mono tracking-wider">
                Early Warning Risk Thresholds
              </h4>
              <div className="space-y-3">
                {[
                  { level: 'Normal Baseline', prob: '< 15%', action: 'Standard monitoring interval (1m)' },
                  { level: 'Watch Warning', prob: '15% - 35%', action: 'Tighten execution slippage guards' },
                  { level: 'Suspicious State', prob: '35% - 60%', action: 'Halt aggressive maker quoting' },
                  { level: 'Compromised', prob: '> 60%', action: 'Route orders to alternative venues' },
                ].map((t) => (
                  <div key={t.level} className="p-2.5 rounded-lg bg-[#0d0f17] border border-[#1e2330] text-xs">
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-bold text-white">{t.level}</span>
                      <span className="font-mono text-indigo-400 font-bold">{t.prob}</span>
                    </div>
                    <p className="text-[10px] text-gray-400">{t.action}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// TAB 4: MANIPULATION PATTERN DIAGNOSTICS (5-IN-1 VIEW)
const ManipulationDiagnosticsSubTab: React.FC = () => {
  const [activePattern, setActivePattern] = useState<ManipulationPatternType>('wash-trading');

  const patternDefinitions = [
    { id: 'wash-trading' as const, label: 'Wash Trading & Rush Flow', icon: ShieldAlert, subtitle: 'Self-trading match rate & circular volume' },
    { id: 'pump-and-dump' as const, label: 'Pump & Dump Detection', icon: TrendingUp, subtitle: 'Price velocity & post-pump liquidity cliff' },
    { id: 'spoofing' as const, label: 'Spoofing & Order Layering', icon: Layers, subtitle: 'Phantom bids & cancellation frequency' },
    { id: 'liquidity-pulls' as const, label: 'Liquidity Withdrawal / Pulls', icon: Zap, subtitle: 'Sudden depth drain & spread expansion' },
    { id: 'fake-breakouts' as const, label: 'Fake Breakouts & Traps', icon: Target, subtitle: 'Price-volume divergence & bull/bear traps' },
  ];

  const patternCharts: Record<ManipulationPatternType, { data: any[]; dataKeys: { key: string; name: string; color: string; type?: 'bar' | 'line' | 'area' }[] }> = {
    'wash-trading': {
      data: [
        { time: '10:00', selfTradeRate: 0.02, vpinToxicity: 0.12, organicRatio: 99.8 },
        { time: '10:05', selfTradeRate: 0.03, vpinToxicity: 0.14, organicRatio: 99.7 },
        { time: '10:10', selfTradeRate: 0.02, vpinToxicity: 0.11, organicRatio: 99.8 },
        { time: '10:15', selfTradeRate: 0.05, vpinToxicity: 0.16, organicRatio: 99.5 },
        { time: '10:20', selfTradeRate: 0.02, vpinToxicity: 0.13, organicRatio: 99.8 },
        { time: '10:25', selfTradeRate: 0.01, vpinToxicity: 0.10, organicRatio: 99.9 },
      ],
      dataKeys: [
        { key: 'vpinToxicity', name: 'VPIN Toxicity (0-1)', color: '#60a5fa', type: 'line' },
        { key: 'selfTradeRate', name: 'Self-Trade Ratio (%)', color: '#f59e0b', type: 'bar' },
      ],
    },
    'pump-and-dump': {
      data: [
        { time: '10:00', volumeSpike: 1.05, priceVelocity: 0.12, decayRisk: 2 },
        { time: '10:05', volumeSpike: 1.12, priceVelocity: 0.18, decayRisk: 3 },
        { time: '10:10', volumeSpike: 1.08, priceVelocity: 0.09, decayRisk: 2 },
        { time: '10:15', volumeSpike: 1.35, priceVelocity: 0.42, decayRisk: 5 },
        { time: '10:20', volumeSpike: 1.15, priceVelocity: -0.15, decayRisk: 4 },
        { time: '10:25', volumeSpike: 1.02, priceVelocity: -0.04, decayRisk: 2 },
      ],
      dataKeys: [
        { key: 'volumeSpike', name: 'Volume Spike Index (×)', color: '#818cf8', type: 'bar' },
        { key: 'priceVelocity', name: 'Price Pump Velocity (%/m)', color: '#34d399', type: 'line' },
      ],
    },
    'spoofing': {
      data: [
        { time: '10:00', cancelSpeed: 1.4, phantomDepth: 2.1, tradeFillRatio: 92.4 },
        { time: '10:05', cancelSpeed: 1.8, phantomDepth: 2.8, tradeFillRatio: 91.8 },
        { time: '10:10', cancelSpeed: 1.2, phantomDepth: 1.9, tradeFillRatio: 93.1 },
        { time: '10:15', cancelSpeed: 2.9, phantomDepth: 4.2, tradeFillRatio: 88.5 },
        { time: '10:20', cancelSpeed: 1.6, phantomDepth: 2.4, tradeFillRatio: 92.0 },
        { time: '10:25', cancelSpeed: 1.1, phantomDepth: 1.8, tradeFillRatio: 94.2 },
      ],
      dataKeys: [
        { key: 'cancelSpeed', name: 'Cancellation Velocity (%/m)', color: '#f43f5e', type: 'line' },
        { key: 'phantomDepth', name: 'Phantom Bid Depth (BTC)', color: '#c084fc', type: 'area' },
      ],
    },
    'liquidity-pulls': {
      data: [
        { time: '10:00', depthResilience: 98.4, spreadExpansion: 1.18, pullVelocity: 0.0 },
        { time: '10:05', depthResilience: 97.9, spreadExpansion: 1.20, pullVelocity: 0.0 },
        { time: '10:10', depthResilience: 99.1, spreadExpansion: 1.19, pullVelocity: 0.0 },
        { time: '10:15', depthResilience: 94.2, spreadExpansion: 1.34, pullVelocity: 0.2 },
        { time: '10:20', depthResilience: 98.0, spreadExpansion: 1.21, pullVelocity: 0.0 },
        { time: '10:25', depthResilience: 99.4, spreadExpansion: 1.18, pullVelocity: 0.0 },
      ],
      dataKeys: [
        { key: 'depthResilience', name: 'L1-L5 Depth Resilience (%)', color: '#10b981', type: 'line' },
        { key: 'spreadExpansion', name: 'Spread Width ($)', color: '#fbbf24', type: 'line' },
      ],
    },
    'fake-breakouts': {
      data: [
        { time: '10:00', momentumDelta: 0.14, volumeDivergence: 0.02, trapScore: 4 },
        { time: '10:05', momentumDelta: 0.19, volumeDivergence: 0.03, trapScore: 5 },
        { time: '10:10', momentumDelta: 0.08, volumeDivergence: 0.01, trapScore: 3 },
        { time: '10:15', momentumDelta: 0.38, volumeDivergence: 0.14, trapScore: 9 },
        { time: '10:20', momentumDelta: -0.12, volumeDivergence: 0.05, trapScore: 6 },
        { time: '10:25', momentumDelta: 0.04, volumeDivergence: 0.02, trapScore: 3 },
      ],
      dataKeys: [
        { key: 'momentumDelta', name: 'Price Momentum Delta', color: '#38bdf8', type: 'line' },
        { key: 'volumeDivergence', name: 'Order Flow Divergence', color: '#ec4899', type: 'bar' },
      ],
    },
  };

  const patternMetrics: Record<ManipulationPatternType, { risk: string; confidence: string; frequency: string; indicators: { title: string; desc: string; val: string }[] }> = {
    'wash-trading': {
      risk: '4.2%',
      confidence: '96.4%',
      frequency: '0 / min',
      indicators: [
        { title: 'Cyclic Wash Transfer Match', desc: 'Identifies matching buyer-seller addresses with identical micro-quantities.', val: 'Clean (0.00%)' },
        { title: 'Burst Timestamp Clustering', desc: 'Execution timestamp proximity within single 10ms network packets.', val: 'Organic Flow' },
        { title: 'VPIN Toxic Volume Asymmetry', desc: 'Volume-synchronized probability of toxicity measuring uninformed vs toxic volume.', val: '0.14 (Optimal)' },
      ],
    },
    'pump-and-dump': {
      risk: '3.8%',
      confidence: '94.8%',
      frequency: '0 / min',
      indicators: [
        { title: 'Volume Spike Acceleration', desc: 'Volume relative to 20-bar exponential baseline.', val: '1.05× (Stable)' },
        { title: 'Price Velocity Gradient', desc: 'Momentum rate of change per second across top order book levels.', val: '+0.12% / min' },
        { title: 'Post-Pump Liquidity Cliff', desc: 'Abrupt evaporation of bid volume following upward price excursions.', val: '0% Evaporation' },
      ],
    },
    'spoofing': {
      risk: '2.1%',
      confidence: '97.2%',
      frequency: '0 / min',
      indicators: [
        { title: 'Flash Cancellation Rate (<200ms)', desc: 'Large limit orders submitted and cancelled before execution.', val: '1.8% (Normal)' },
        { title: 'Quote-to-Trade Ratio', desc: 'Ratio of submitted quote updates to filled taker orders.', val: '14.2 / Trade' },
        { title: 'Deep Book Layering Thickness', desc: 'Asymmetric non-executable depth added across L3-L10 levels.', val: 'Symmetric Depth' },
      ],
    },
    'liquidity-pulls': {
      risk: '1.4%',
      confidence: '95.9%',
      frequency: '0 / min',
      indicators: [
        { title: 'Sudden Depth Drain Velocity', desc: 'Top-of-book bid depth reduction during aggressive selling.', val: '0.00% (Solid)' },
        { title: 'Spread Expansion Multiplier', desc: 'Dynamic widening of bid-ask spread during liquidity withdrawal.', val: '1.02× Baseline' },
        { title: 'Market Maker Retreat Metric', desc: 'Simultaneous cancellation of primary passive liquidity providers.', val: 'Active Quoting' },
      ],
    },
    'fake-breakouts': {
      risk: '5.1%',
      confidence: '91.5%',
      frequency: '0 / min',
      indicators: [
        { title: 'Price-Volume Flow Divergence', desc: 'Price new highs without corresponding organic buy-side volume influx.', val: '0.08 Divergence' },
        { title: 'Passive Absorption Wall', desc: 'Aggressive buyers absorbed by concealed icebergs causing reversal.', val: 'No Absorption' },
        { title: 'Bull / Bear Trap Signal', desc: 'Rapid stop-loss hunt pattern triggering cascading retail liquidations.', val: 'Normal Flow' },
      ],
    },
  };

  const currentPatternConfig = patternMetrics[activePattern];
  const currentChartConfig = patternCharts[activePattern];

  return (
    <div className="space-y-6">
      <div className="glass-panel p-4 rounded-xl border border-indigo-500/30 bg-gradient-to-r from-[#0f1428] via-[#0d0f1c] to-[#141226] space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-5 h-5 text-indigo-400" />
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">
              Manipulation Pattern Diagnostic Engine
            </h3>
          </div>
          <span className="text-[10px] text-indigo-400 font-mono font-bold px-2.5 py-1 rounded bg-indigo-500/10 border border-indigo-500/20">
            5-in-1 Pattern Classifier
          </span>
        </div>

        <div className="flex items-center gap-2 overflow-x-auto pb-1">
          {patternDefinitions.map((p) => {
            const Icon = p.icon;
            const isActive = activePattern === p.id;
            return (
              <button
                key={p.id}
                onClick={() => setActivePattern(p.id)}
                className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-bold transition-all whitespace-nowrap cursor-pointer ${
                  isActive
                    ? 'bg-indigo-600 text-white shadow-md border border-indigo-400/40'
                    : 'bg-[#0d0f17] text-gray-400 hover:text-gray-200 border border-[#1e2330] hover:bg-[#151824]'
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-white' : 'text-gray-400'}`} />
                <span>{p.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <C2StyleMetricCard title="Active Pattern Risk Score" value={currentPatternConfig.risk} status="LOW RISK" subtext="Supervised XGBoost + AE Score" icon={<ShieldCheck className="w-4 h-4 text-emerald-400" />} />
        <C2StyleMetricCard title="Detection Confidence" value={currentPatternConfig.confidence} status="CONFORMAL CP-95" subtext="Non-Conformity Interval Set" icon={<Target className="w-4 h-4 text-indigo-400" />} />
        <C2StyleMetricCard title="Observed Burst Frequency" value={currentPatternConfig.frequency} status="ZERO DETECTED" subtext="Rolling 1-Minute Evaluation Window" icon={<Activity className="w-4 h-4 text-blue-400" />} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 glass-panel p-5 rounded-xl border border-[#1e2330] space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-xs font-bold text-gray-200 uppercase font-mono tracking-wider">
                {patternDefinitions.find((p) => p.id === activePattern)?.label} — Microstructure Dynamics
              </h4>
              <p className="text-[10px] text-gray-400 mt-0.5">{patternDefinitions.find((p) => p.id === activePattern)?.subtitle}</p>
            </div>
            <span className="text-[10px] text-emerald-400 font-mono font-bold bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
              Pattern Inactive
            </span>
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={currentChartConfig.data} margin={{ top: 10, right: 15, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2330" />
                <XAxis dataKey="time" tick={{ fill: '#6b7280', fontSize: 10, fontFamily: 'monospace' }} axisLine={{ stroke: '#1e2330' }} tickLine={false} />
                <YAxis tick={{ fill: '#6b7280', fontSize: 10, fontFamily: 'monospace' }} axisLine={false} tickLine={false} />
                <ReTooltip contentStyle={{ backgroundColor: '#0d0f17', borderColor: '#1e2330', borderRadius: '8px', fontSize: '11px' }} />
                {currentChartConfig.dataKeys.map((dk) => {
                  if (dk.type === 'bar') {
                    return <Bar key={dk.key} dataKey={dk.key} name={dk.name} fill={dk.color} radius={[4, 4, 0, 0]} />;
                  }
                  if (dk.type === 'area') {
                    return <Area key={dk.key} type="monotone" dataKey={dk.key} name={dk.name} stroke={dk.color} fill={`${dk.color}25`} strokeWidth={2} />;
                  }
                  return <Line key={dk.key} type="monotone" dataKey={dk.key} name={dk.name} stroke={dk.color} strokeWidth={2} dot={{ r: 3 }} />;
                })}
                <Legend formatter={(v) => <span className="text-[11px] text-gray-300 font-mono">{v}</span>} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="glass-panel p-5 rounded-xl border border-[#1e2330] space-y-4">
          <h4 className="text-xs font-bold text-gray-200 uppercase font-mono tracking-wider flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            Pattern Verification Indicators
          </h4>
          <div className="space-y-3">
            {currentPatternConfig.indicators.map((ind) => (
              <div key={ind.title} className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-gray-200">{ind.title}</span>
                  <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono text-[10px] font-bold">
                    {ind.val}
                  </span>
                </div>
                <p className="text-[10px] text-gray-400 mt-1">{ind.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

// TAB 5: EXPLAINABILITY & HISTORICAL CONTEXT (COMBINES SHAP & VECTOR SIMILARITY MATCHING)
const ExplainabilityAndHistoricalSubTab: React.FC<{ data: IntegrityDataPayload }> = ({ data }) => {
  const [selectedItem, setSelectedItem] = useState<HistoricalWindow | null>(null);

  return (
    <div className="space-y-8">
      {/* ── SECTION 1: SHAP FEATURE ATTRIBUTION ─── */}
      <div className="space-y-6">
        <div className="glass-panel p-5 rounded-xl border border-indigo-500/30 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Eye className="w-4 h-4 text-purple-400" />
                SHAP Feature Attribution (Diverging Centered Decomposition)
              </h3>
              <p className="text-[11px] text-gray-400 mt-0.5">
                Microstructure contributions: Emerald Green (Raises Integrity &gt; 0) vs Rose Red (Degrades Integrity &lt; 0)
              </p>
            </div>
            <div className="flex items-center gap-3 text-[10px] font-bold">
              <span className="flex items-center gap-1 text-emerald-400">
                <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500 inline-block" /> Positive (+Right)
              </span>
              <span className="flex items-center gap-1 text-rose-400">
                <span className="w-2.5 h-2.5 rounded-sm bg-rose-500 inline-block" /> Negative (-Left)
              </span>
            </div>
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.shapFeatures} layout="vertical" margin={{ top: 10, right: 30, left: 30, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2330" horizontal={false} />
                <XAxis type="number" domain={[-0.5, 0.5]} ticks={[-0.4, -0.2, 0, 0.2, 0.4]} tick={{ fill: '#6b7280', fontSize: 10, fontFamily: 'monospace' }} axisLine={{ stroke: '#1e2330' }} tickLine={false} />
                <YAxis type="category" dataKey="displayName" width={140} tick={{ fill: '#9ca3af', fontSize: 11, fontFamily: 'monospace' }} axisLine={false} tickLine={false} />
                <ReferenceLine x={0} stroke="#64748b" strokeWidth={1.5} />
                <ReTooltip content={<ShapTip />} />
                <Bar dataKey="importance" radius={[4, 4, 4, 4]}>
                  {data.shapFeatures.map((entry, idx) => (
                    <Cell key={`shap-${idx}`} fill={entry.importance >= 0 ? '#22c55e' : '#ef4444'} fillOpacity={0.88} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Feature Detail Table */}
        <div className="glass-panel rounded-xl border border-[#1e2330] overflow-hidden">
          <div className="p-4 border-b border-[#1e2330] bg-[#08090d]">
            <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <BarChart2 className="w-4 h-4 text-purple-400" />
              Feature Impact Breakdown &amp; Attribution Weights
            </h3>
          </div>
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-[#1e2330] bg-[#08090d] text-gray-400 font-mono text-[10px] uppercase">
                <th className="px-4 py-3">Feature Name</th>
                <th className="px-4 py-3">SHAP Value</th>
                <th className="px-4 py-3">Direction</th>
                <th className="px-4 py-3">Microstructure Effect</th>
                <th className="px-4 py-3">Feature Category</th>
              </tr>
            </thead>
            <tbody>
              {data.shapFeatures.map((f, idx) => (
                <tr key={f.feature} className={`border-b border-[#1a1d2b] hover:bg-[#10121a] ${idx % 2 === 0 ? 'bg-[#0d0f17]' : 'bg-[#0b0c10]'}`}>
                  <td className="px-4 py-3 text-[11px] font-mono text-gray-200 font-bold">{f.feature}</td>
                  <td className="px-4 py-3 text-[11px] font-mono text-white">{f.importance >= 0 ? `+${f.importance.toFixed(3)}` : f.importance.toFixed(3)}</td>
                  <td className="px-4 py-3">
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${f.importance >= 0 ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'}`}>
                      {f.importance >= 0 ? '▲ Positive' : '▼ Negative'}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-[11px] text-gray-400">{f.importance >= 0 ? 'Supports normal price formation' : 'Signals order book stress/clustering'}</td>
                  <td className="px-4 py-3 text-[11px] text-gray-500 font-mono">L2 Microstructure</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── SECTION 2: HISTORICAL VECTOR SIMILARITY MATCHER ─── */}
      <div className="space-y-6 pt-4 border-t border-[#1e2330]">
        <div className="glass-panel p-6 rounded-xl border border-indigo-500/30 bg-gradient-to-r from-[#0f1428] via-[#0d0f1c] to-[#141226] space-y-3">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-indigo-500/15 text-indigo-400 border border-indigo-500/30 shadow-[0_0_12px_rgba(99,102,241,0.2)]">
              <History className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white tracking-tight">
                Historical Microstructure Vector Similarity Matcher
              </h2>
              <p className="text-xs text-gray-300 mt-0.5">
                Nearest-neighbor retrieval of microstructure &amp; manipulation feature vectors via Qdrant / FAISS Vector DB.
              </p>
            </div>
          </div>

          <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 shrink-0" />
            <span>Notice: Historical microstructure patterns provide baseline analog context for anomaly assessment.</span>
          </div>
        </div>

        {/* Query Vector Card */}
        <div className="glass-panel p-5 rounded-xl border border-indigo-500/30 space-y-3">
          <div className="flex items-center justify-between border-b border-[#1e2330] pb-3">
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">
              Active Market Integrity Query Vector (47-D Latent State)
            </h3>
            <span className="text-xs font-mono text-indigo-400 bg-indigo-500/10 px-2.5 py-1 rounded border border-indigo-500/20">
              Active Query Vector
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 text-center">
            <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
              <span className="text-[10px] text-gray-400 uppercase font-medium">Spread Deviation</span>
              <div className="text-sm font-extrabold text-emerald-400 font-mono mt-0.5">{VECTOR_QUERY_INTEGRITY_STATE.spread}</div>
            </div>
            <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
              <span className="text-[10px] text-gray-400 uppercase font-medium">Order Book Imbalance</span>
              <div className="text-sm font-extrabold text-emerald-400 font-mono mt-0.5">{VECTOR_QUERY_INTEGRITY_STATE.obImbalance}</div>
            </div>
            <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
              <span className="text-[10px] text-gray-400 uppercase font-medium">Rush Order Density</span>
              <div className="text-sm font-extrabold text-indigo-400 font-mono mt-0.5">{VECTOR_QUERY_INTEGRITY_STATE.rushOrderDensity}</div>
            </div>
            <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
              <span className="text-[10px] text-gray-400 uppercase font-medium">Volume Toxicity</span>
              <div className="text-sm font-extrabold text-blue-400 font-mono mt-0.5">{VECTOR_QUERY_INTEGRITY_STATE.vpinToxicity}</div>
            </div>
            <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
              <span className="text-[10px] text-gray-400 uppercase font-medium">Cancel Velocity</span>
              <div className="text-sm font-extrabold text-amber-400 font-mono mt-0.5">{VECTOR_QUERY_INTEGRITY_STATE.cancelVelocity}</div>
            </div>
            <div className="p-3 rounded-lg bg-[#0d0f17] border border-[#1e2330]">
              <span className="text-[10px] text-gray-400 uppercase font-medium">Graph Coupling</span>
              <div className="text-sm font-extrabold text-purple-400 font-mono mt-0.5">{VECTOR_QUERY_INTEGRITY_STATE.graphCoupling}</div>
            </div>
          </div>
        </div>

        {/* Top Similarity Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {data.historicalWindows.map((item) => {
            const isHealthy = item.integrityOutcomeScore >= 80;
            const stateCfg = getStateConfig(item.historicalState);
            const StateIcon = stateCfg.Icon;
            return (
              <div key={item.id} className="glass-panel p-5 rounded-xl border border-[#1e2330] hover:border-indigo-500/40 transition-all flex flex-col justify-between space-y-4 bg-[#10131f]">
                <div>
                  <div className="flex items-center justify-between border-b border-[#1e2330] pb-3 mb-3">
                    <div>
                      <span className="text-sm font-bold text-white font-mono">{item.timestamp}</span>
                      <span className="text-[10px] text-gray-400 font-mono block">{item.asset}</span>
                    </div>
                    <span className="px-2.5 py-1 rounded bg-indigo-500/10 text-indigo-400 font-mono font-bold text-xs border border-indigo-500/20 block">
                      {item.similarityScore}% Match
                    </span>
                  </div>

                  <div className="space-y-2 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] text-gray-400 font-medium">Historical State:</span>
                      <span className={`inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded ${stateCfg.bg} ${stateCfg.border} border`} style={{ color: stateCfg.color }}>
                        <StateIcon className="w-3 h-3" />
                        {item.historicalState}
                      </span>
                    </div>

                    <div className="p-2.5 rounded-lg bg-[#0d0f17] border border-[#1e2330] space-y-1">
                      <div className="text-[10px] text-gray-400 font-medium uppercase">Microstructure Pattern:</div>
                      <div className="text-indigo-300 font-mono text-[11px] font-bold">{item.eventPattern}</div>
                      <div className="text-gray-400 font-mono text-[10px] pt-1">
                        Spread: {item.conditions.spread} | OB: {item.conditions.obImbalance} | Rush: {item.conditions.rushOrderDensity}
                      </div>
                    </div>

                    <div className="p-3 rounded-lg bg-[#121520] border border-[#1e2330] flex items-center justify-between">
                      <span className="text-gray-400 font-medium text-[11px]">Integrity Outcome (+1H):</span>
                      <span className={`font-mono font-bold text-xs flex items-center gap-1 ${isHealthy ? 'text-emerald-400' : 'text-rose-400'}`}>
                        <ShieldCheck className="w-3.5 h-3.5" />
                        Score: {item.integrityOutcomeScore}/100
                      </span>
                    </div>
                  </div>
                </div>

                <button
                  onClick={() => setSelectedItem(item)}
                  className="w-full py-2.5 rounded-lg bg-[#181c2b] hover:bg-indigo-600 text-indigo-400 hover:text-white border border-indigo-500/30 text-xs font-bold transition-all flex items-center justify-center gap-2 cursor-pointer shadow-sm"
                >
                  <span>View Evidence &amp; Vector Breakdown</span>
                  <ExternalLink className="w-3.5 h-3.5" />
                </button>
              </div>
            );
          })}
        </div>
      </div>

      {/* Interactive Modal */}
      <HistoricalIntegrityEvidenceModal item={selectedItem} onClose={() => setSelectedItem(null)} />
    </div>
  );
};

// ─────────────────────────────────────────────────────────────
// MASTER COMPONENT (5 CLEAN SUB-TABS)
// ─────────────────────────────────────────────────────────────
export const MarketIntegrityTab: React.FC = () => {
  const [selectedAsset, setSelectedAsset] = useState<AssetMode>('ALL');
  const [selectedTimeframe, setSelectedTimeframe] = useState<TimeframeMode>('1m');
  const [activeSubTab, setActiveSubTab] = useState<Component3SubTab>('overview');
  const [data, setData] = useState<IntegrityDataPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const loadData = useCallback(async (asset: AssetMode, tf: TimeframeMode, isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    else setLoading(true);
    try {
      const result = await fetchIntegrityData(asset, tf);
      setData(result);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadData(selectedAsset, selectedTimeframe);
  }, [selectedAsset, selectedTimeframe, loadData]);

  // 5 CLEAN CONSOLIDATED SUB-NAV ITEMS
  const subNavItems: { id: Component3SubTab; label: string; Icon: React.FC<{ className?: string }> }[] = [
    { id: 'overview', label: 'Overview', Icon: LayoutDashboard },
    { id: 'microstructure', label: 'Microstructure Intelligence', Icon: Activity },
    { id: 'ml-evidence', label: 'ML Model Evidence', Icon: Cpu },
    { id: 'manipulation-diagnostics', label: 'Manipulation Diagnostics', Icon: ShieldAlert },
    { id: 'explainability-historical', label: 'Explainability & Historical Context', Icon: FileSearch },
  ];

  const renderSubContent = () => {
    if (!data) return null;
    switch (activeSubTab) {
      case 'overview':
        return <OverviewSubTab data={data} selectedAsset={selectedAsset} selectedTimeframe={selectedTimeframe} />;
      case 'microstructure':
        return <MicrostructureIntelligenceSubTab />;
      case 'ml-evidence':
        return <MLEvidenceHubSubTab data={data} />;
      case 'manipulation-diagnostics':
        return <ManipulationDiagnosticsSubTab />;
      case 'explainability-historical':
        return <ExplainabilityAndHistoricalSubTab data={data} />;
      default:
        return <OverviewSubTab data={data} selectedAsset={selectedAsset} selectedTimeframe={selectedTimeframe} />;
    }
  };

  return (
    <div className="space-y-6">
      {/* ── TOP HEADER & GLOBAL CONTROLS BAR (GLASS PANEL + GLOW GRADIENT) ─── */}
      <div className="glass-panel p-6 rounded-xl border border-indigo-500/30 glow-indigo bg-gradient-to-r from-[#0f1428] via-[#0d0f1c] to-[#111226] space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#1e2330] pb-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 font-mono text-xs font-bold border border-indigo-500/30">
                COMPONENT 3 CORE MODULE
              </span>
              <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 text-[10px] font-bold uppercase tracking-wider">
                <Sparkles className="w-3 h-3" />
                PROTOTYPE • MOCK DATA ENGINE
              </span>
            </div>
            <h2 className="text-xl sm:text-2xl font-extrabold text-white tracking-tight mt-1">
              AI-Based Market Integrity Intelligence Engine
            </h2>
            <p className="text-xs text-gray-300 mt-0.5">
              Real-time microstructure integrity assessment, anomaly detection, and explainable risk scoring for BTC &amp; ETH.
            </p>
          </div>

          {/* Controls Cluster */}
          <div className="flex flex-wrap items-center gap-2">
            {/* Asset Selector */}
            <div className="flex items-center bg-[#0d0f17] border border-[#1f2330] rounded-lg p-1 text-xs">
              {(['ALL', 'BTC', 'ETH'] as AssetMode[]).map((asset) => (
                <button
                  key={asset}
                  onClick={() => setSelectedAsset(asset)}
                  className={`px-3 py-1 rounded-md font-semibold transition-all cursor-pointer ${
                    selectedAsset === asset
                      ? 'bg-indigo-600 text-white shadow-sm'
                      : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  {asset === 'ALL' ? 'Multi-Asset' : `${asset}/USDT`}
                </button>
              ))}
            </div>

            {/* Timeframe Selector (Strict Micro-Horizons) */}
            <div className="flex items-center bg-[#0d0f17] border border-[#1f2330] rounded-lg p-1 text-xs">
              <span className="px-2 text-gray-400 font-medium text-[11px] flex items-center gap-1">
                <Clock className="w-3 h-3 text-gray-400" />
                TF:
              </span>
              {(['1m', '5m', '15m', '1H'] as TimeframeMode[]).map((tf) => (
                <button
                  key={tf}
                  onClick={() => setSelectedTimeframe(tf)}
                  className={`px-2.5 py-1 rounded-md font-semibold transition-all cursor-pointer ${
                    selectedTimeframe === tf
                      ? 'bg-purple-600 text-white shadow-sm'
                      : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  {tf}
                </button>
              ))}
            </div>

            {/* Live Status & Refresh */}
            <div className="flex items-center gap-2 pl-1">
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-500/10 border border-emerald-500/20">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-[11px] font-bold text-emerald-400 font-mono">LIVE WEBSOCKET</span>
              </div>
              <button
                onClick={() => loadData(selectedAsset, selectedTimeframe, true)}
                disabled={refreshing || loading}
                className="p-1.5 rounded-lg bg-[#0d0f17] border border-[#1e2330] text-gray-400 hover:text-white hover:border-indigo-500/40 transition-all cursor-pointer"
                title="Refresh Stream"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin' : ''}`} />
              </button>
            </div>
          </div>
        </div>

        {/* ── 5 CLEAN CONSOLIDATED SUB-NAVIGATION PILLS ─── */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1">
          {subNavItems.map((item) => {
            const Icon = item.Icon;
            const isActive = activeSubTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveSubTab(item.id)}
                className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-bold transition-all whitespace-nowrap cursor-pointer ${
                  isActive
                    ? 'bg-indigo-600 text-white shadow-md border border-indigo-400/40'
                    : 'bg-[#0d0f17] text-gray-400 hover:text-gray-200 border border-[#1e2330] hover:bg-[#151824]'
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-white' : 'text-gray-400'}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Render Sub-Content */}
      {loading ? (
        <div className="flex items-center justify-center h-80 space-x-3">
          <RefreshCw className="w-6 h-6 text-indigo-400 animate-spin" />
          <span className="text-gray-400 text-sm font-mono">Loading Market Integrity Stream...</span>
        </div>
      ) : (
        renderSubContent()
      )}
    </div>
  );
};

export default MarketIntegrityTab;
