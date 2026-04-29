import type { FC } from 'react';
import { Loader2 } from 'lucide-react';

export type LoadingSpinnerSize = 'sm' | 'md' | 'lg';

export interface LoadingSpinnerProps {
  size?: LoadingSpinnerSize;
  className?: string;
  'aria-label'?: string;
}

const sizeMap: Record<LoadingSpinnerSize, string> = {
  sm: 'h-4 w-4',
  md: 'h-6 w-6',
  lg: 'h-8 w-8',
};

/**
 * Accessible loading spinner. Honors `prefers-reduced-motion` (no spin when reduced).
 */
export const LoadingSpinner: FC<LoadingSpinnerProps> = ({
  size = 'md',
  className,
  'aria-label': ariaLabel = 'Loading...',
}) => {
  return (
    <span
      role="status"
      aria-live="polite"
      aria-label={ariaLabel}
      className={['inline-flex items-center justify-center text-accent', className]
        .filter(Boolean)
        .join(' ')}
    >
      <Loader2
        aria-hidden="true"
        className={[sizeMap[size], 'motion-safe:animate-spin'].join(' ')}
      />
    </span>
  );
};
