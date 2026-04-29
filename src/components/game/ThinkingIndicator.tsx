import type { FC } from 'react';
import { Loader2 } from 'lucide-react';
import { useReducedMotion } from 'framer-motion';

export interface ThinkingIndicatorProps {
  /** Elapsed thinking time in milliseconds. */
  thinkingMs?: number;
  /** Optional difficulty (1-10) shown as a small badge. */
  difficulty?: number;
  /** Optional override label (default: "CPU 思考中…"). */
  label?: string;
  /** Optional className for the wrapper. */
  className?: string;
}

/**
 * Inline indicator for "CPU is thinking…" with elapsed time and an optional
 * difficulty badge. Honors `prefers-reduced-motion` (no spin when reduced).
 */
export const ThinkingIndicator: FC<ThinkingIndicatorProps> = ({
  thinkingMs,
  difficulty,
  label = 'CPU 思考中…',
  className,
}) => {
  const reducedMotion = useReducedMotion();
  const seconds = thinkingMs !== undefined ? (thinkingMs / 1000).toFixed(1) : null;

  const wrapperClasses = [
    'inline-flex items-center gap-2 rounded-md border border-border bg-bg px-3 py-1.5 text-sm text-fg',
    className ?? '',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div role="status" aria-live="polite" className={wrapperClasses}>
      <Loader2
        aria-hidden="true"
        className={['h-4 w-4 text-accent', reducedMotion ? '' : 'motion-safe:animate-spin']
          .filter(Boolean)
          .join(' ')}
      />
      <span>{label}</span>
      {seconds !== null && (
        <span data-testid="thinking-elapsed" className="tabular-nums text-muted">
          {seconds}s
        </span>
      )}
      {difficulty !== undefined && (
        <span
          data-testid="thinking-difficulty"
          className="ml-1 rounded-full bg-accent px-2 py-0.5 text-xs font-medium text-bg"
          aria-label={`難易度 ${difficulty}`}
        >
          Lv.{difficulty}
        </span>
      )}
    </div>
  );
};
