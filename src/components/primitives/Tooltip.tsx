import {
  Children,
  cloneElement,
  isValidElement,
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
} from 'react';
import type { FocusEvent, MouseEvent, ReactElement, ReactNode } from 'react';

export type TooltipPlacement = 'top' | 'right' | 'bottom' | 'left';

export interface TooltipProps {
  content: ReactNode;
  placement?: TooltipPlacement;
  /** Show delay in ms (default 150). */
  delay?: number;
  /** Single React element child. */
  children: ReactElement;
  /** Disable the tooltip entirely. */
  disabled?: boolean;
  className?: string;
}

const placementClasses: Record<TooltipPlacement, string> = {
  top: 'bottom-full left-1/2 -translate-x-1/2 mb-1.5',
  bottom: 'top-full left-1/2 -translate-x-1/2 mt-1.5',
  left: 'right-full top-1/2 -translate-y-1/2 mr-1.5',
  right: 'left-full top-1/2 -translate-y-1/2 ml-1.5',
};

interface TriggerProps {
  onMouseEnter?: (e: MouseEvent<HTMLElement>) => void;
  onMouseLeave?: (e: MouseEvent<HTMLElement>) => void;
  onFocus?: (e: FocusEvent<HTMLElement>) => void;
  onBlur?: (e: FocusEvent<HTMLElement>) => void;
  'aria-describedby'?: string | undefined;
}

/**
 * Hover/focus tooltip that wraps a single child.
 * - Adds `aria-describedby` to the child pointing at the tooltip.
 * - Show on hover or keyboard focus; hide on unhover or blur.
 * - Honors `delay` for show; hides immediately.
 */
export const Tooltip = ({
  content,
  placement = 'top',
  delay = 150,
  disabled = false,
  className,
  children,
}: TooltipProps) => {
  const reactId = useId();
  const tooltipId = `tooltip-${reactId}`;
  const [open, setOpen] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clear = useCallback(() => {
    if (timer.current) {
      clearTimeout(timer.current);
      timer.current = null;
    }
  }, []);

  useEffect(() => () => clear(), [clear]);

  const show = useCallback(() => {
    if (disabled) return;
    clear();
    if (delay <= 0) {
      setOpen(true);
      return;
    }
    timer.current = setTimeout(() => setOpen(true), delay);
  }, [clear, delay, disabled]);

  const hide = useCallback(() => {
    clear();
    setOpen(false);
  }, [clear]);

  // Only one child supported; clone to attach handlers.
  const child = Children.only(children);
  if (!isValidElement<TriggerProps>(child)) {
    return children;
  }

  const existing = child.props;

  const triggerProps: TriggerProps = {
    onMouseEnter: (e) => {
      existing.onMouseEnter?.(e);
      show();
    },
    onMouseLeave: (e) => {
      existing.onMouseLeave?.(e);
      hide();
    },
    onFocus: (e) => {
      existing.onFocus?.(e);
      show();
    },
    onBlur: (e) => {
      existing.onBlur?.(e);
      hide();
    },
    'aria-describedby': open
      ? [existing['aria-describedby'], tooltipId].filter(Boolean).join(' ')
      : existing['aria-describedby'],
  };

  const cloned = cloneElement(child, triggerProps);

  return (
    <span className="relative inline-flex">
      {cloned}
      <span
        id={tooltipId}
        role="tooltip"
        data-state={open ? 'open' : 'closed'}
        className={[
          'pointer-events-none absolute z-40 whitespace-nowrap rounded-md border border-border',
          'bg-bg px-2 py-1 text-xs text-fg shadow-md',
          'transition-opacity',
          open ? 'opacity-100' : 'opacity-0',
          placementClasses[placement],
          className,
        ]
          .filter(Boolean)
          .join(' ')}
      >
        {content}
      </span>
    </span>
  );
};
