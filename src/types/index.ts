export type MainTab = 'home' | 'component-1' | 'component-2' | 'component-3' | 'component-4';

export type Component2SubTab = 
  | 'overview'
  | 'btc-intelligence'
  | 'eth-intelligence'
  | 'cross-market'
  | 'capital-flow'
  | 'news-sentiment'
  | 'prediction'
  | 'historical-evidence'
  | 'explainability';

export type AssetType = 'BTC' | 'ETH' | 'ALL';
export type TimeRange = '1H' | '4H' | '1D' | '1W';
export type ForecastHorizon = '1H' | '2H' | '3H' | '4H';
export type SentimentFilter = 'ALL' | 'Positive' | 'Neutral' | 'Negative';

export interface PricePoint {
  time: string;
  price: number;
  predicted?: number;
  lowerBound?: number;
  upperBound?: number;
  volume: number;
}

export interface MetricCardData {
  title: string;
  value: string;
  change?: string;
  isPositive?: boolean;
  status?: string;
  subtext?: string;
}

export interface DriverImpactData {
  name: string;
  impactScore: number;
  label: string;
  type: 'bearish' | 'bullish' | 'neutral';
  description?: string;
}

export interface InternalSignalData {
  name: string;
  value: string;
  trend: 'up' | 'down' | 'neutral';
  status: 'bullish' | 'bearish' | 'neutral';
  change24h: string;
}

export interface CorrelationItem {
  asset1: string;
  asset2: string;
  value: number;
}

export interface WhaleTransaction {
  id: string;
  timestamp: string;
  asset: 'BTC' | 'ETH';
  amount: string;
  valueUsd: string;
  from: string;
  to: string;
  type: 'Inflow' | 'Outflow' | 'Transfer';
  impact: 'High' | 'Medium' | 'Low';
}

export interface NewsArticle {
  id: string;
  headline: string;
  source: string;
  time: string;
  asset: 'BTC' | 'ETH' | 'ALL';
  sentiment: 'Positive' | 'Neutral' | 'Negative';
  impact: 'High' | 'Medium' | 'Low';
  confidence: number;
  summary: string;
  finbertScores: {
    positive: number;
    neutral: number;
    negative: number;
  };
}

export interface ForecastItem {
  horizon: string;
  direction: 'Bullish' | 'Bearish' | 'Neutral-Bullish' | 'Neutral-Bearish';
  expectedReturn: string;
  confidence: number;
  targetPrice: number;
  description: string;
}

export interface HistoricalEvidenceItem {
  id: string;
  date: string;
  similarity: number;
  conditions: {
    btcMomentum: string;
    dxy: string;
    sp500: string;
    newsSentiment: string;
    fearGreed: number;
  };
  outcome: string;
  outcomeReturn: number;
  notes: string;
  vectorDistance: number;
}

export interface ShapFeature {
  feature: string;
  importance: number;
  impact: 'positive' | 'negative';
  category: 'cross-market' | 'internal' | 'sentiment';
}
