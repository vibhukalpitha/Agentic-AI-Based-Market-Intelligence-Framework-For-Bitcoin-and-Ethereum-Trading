import React from 'react';
import { TrendingUp, TrendingDown, MinusCircle, AlertCircle } from 'lucide-react';

/**
 * Reusable SignalBadge Component
 * Implements strict Meta-Learner confidence threshold logic:
 * 
 * - If Confidence > 55%:
 *     Label: BUY
 *     UI Styling: Neon Green (#00FF00) text/border to indicate a strong bullish signal.
 * 
 * - If Confidence is between 45% and 55% (inclusive):
 *     Label: NEUTRAL
 *     UI Styling: Muted Yellow or Gray to indicate a weak/risky signal that the user should ignore.
 * 
 * - If Confidence < 45%:
 *     Label: SELL
 *     UI Styling: Neon Red (#FF0000) text/border to indicate a bearish signal.
 */
export default function SignalBadge({ 
  confidence, 
  showPercentage = false, 
  className = '',
  size = 'md' 
}) {
  const conf = typeof confidence === 'number' ? confidence : parseFloat(confidence);

  if (isNaN(conf)) {
    return (
      <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold border border-slate-700 text-slate-400 bg-slate-800/40 ${className}`}>
        <MinusCircle size={12} />
        <span>N/A</span>
      </span>
    );
  }

  let label = 'NEUTRAL';
  let tailwindClass = '';
  let inlineStyles = {};
  let Icon = AlertCircle;

  if (conf > 55) {
    label = 'BUY';
    Icon = TrendingUp;
    // Neon Green (#00FF00) text/border with glowing backdrop
    tailwindClass = 'signal-badge-buy text-[#00FF00] border-[#00FF00] bg-[#00FF00]/10 shadow-[0_0_12px_rgba(0,255,0,0.35)]';
    inlineStyles = {
      color: '#00FF00',
      borderColor: '#00FF00',
      backgroundColor: 'rgba(0, 255, 0, 0.12)',
      boxShadow: '0 0 10px rgba(0, 255, 0, 0.35)',
      textShadow: '0 0 4px rgba(0, 255, 0, 0.5)'
    };
  } else if (conf >= 45 && conf <= 55) {
    label = 'NEUTRAL';
    Icon = MinusCircle;
    // Muted Yellow / Gray to indicate weak/risky signal
    tailwindClass = 'signal-badge-neutral text-yellow-400 border-yellow-500/50 bg-yellow-500/10 text-amber-300 border-amber-500/40';
    inlineStyles = {
      color: '#eab308',
      borderColor: 'rgba(234, 179, 8, 0.5)',
      backgroundColor: 'rgba(234, 179, 8, 0.12)',
      boxShadow: '0 0 6px rgba(234, 179, 8, 0.2)'
    };
  } else {
    label = 'SELL';
    Icon = TrendingDown;
    // Neon Red (#FF0000) text/border with glowing backdrop
    tailwindClass = 'signal-badge-sell text-[#FF0000] border-[#FF0000] bg-[#FF0000]/10 shadow-[0_0_12px_rgba(255,0,0,0.35)]';
    inlineStyles = {
      color: '#FF0000',
      borderColor: '#FF0000',
      backgroundColor: 'rgba(255, 0, 0, 0.12)',
      boxShadow: '0 0 10px rgba(255, 0, 0, 0.35)',
      textShadow: '0 0 4px rgba(255, 0, 0, 0.5)'
    };
  }

  const isSmall = size === 'sm';

  return (
    <span
      className={`signal-badge inline-flex items-center gap-1.5 font-bold uppercase tracking-wider rounded-full border transition-all duration-200 select-none ${tailwindClass} ${className}`}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: isSmall ? '4px' : '6px',
        padding: isSmall ? '2px 8px' : '3px 10px',
        borderRadius: '9999px',
        fontSize: isSmall ? '10px' : '11px',
        fontWeight: 700,
        textTransform: 'uppercase',
        letterSpacing: '0.06em',
        borderWidth: '1.5px',
        borderStyle: 'solid',
        ...inlineStyles
      }}
      title={`Confidence: ${conf.toFixed(1)}% -> Signal: ${label}`}
    >
      <Icon size={isSmall ? 11 : 13} strokeWidth={2.5} />
      <span>{label}</span>
      {showPercentage && (
        <span style={{ opacity: 0.85, fontSize: isSmall ? '9px' : '10px', fontWeight: 600 }}>
          {conf.toFixed(1)}%
        </span>
      )}
    </span>
  );
}
