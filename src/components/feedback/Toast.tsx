import type { ReactNode } from 'react';
import { AlertCircle, CheckCircle2, Info, XCircle } from 'lucide-react';

export type ToastVariant = 'info' | 'success' | 'warning' | 'error';

export interface ToastProps {
  title?: ReactNode;
  children: ReactNode;
  variant?: ToastVariant;
  className?: string;
}

const variantClasses: Record<ToastVariant, string> = {
  info: 'border-accent/40 bg-bg text-fg',
  success: 'border-emerald-500/50 bg-bg text-fg',
  warning: 'border-amber-500/60 bg-bg text-fg',
  error: 'border-p1/60 bg-bg text-fg',
};

const iconClasses: Record<ToastVariant, string> = {
  info: 'text-accent',
  success: 'text-emerald-500',
  warning: 'text-amber-500',
  error: 'text-p1',
};

const icons: Record<ToastVariant, ReactNode> = {
  info: <Info className="h-5 w-5" />,
  success: <CheckCircle2 className="h-5 w-5" />,
  warning: <AlertCircle className="h-5 w-5" />,
  error: <XCircle className="h-5 w-5" />,
};

export const Toast = ({ title, children, variant = 'info', className }: ToastProps) => {
  const isAssertive = variant === 'error' || variant === 'warning';

  return (
    <div
      role={isAssertive ? 'alert' : 'status'}
      aria-live={isAssertive ? 'assertive' : 'polite'}
      className={[
        'flex w-full max-w-md items-start gap-3 rounded-md border px-4 py-3 text-sm shadow-sm',
        variantClasses[variant],
        className,
      ]
        .filter(Boolean)
        .join(' ')}
      data-variant={variant}
    >
      <span aria-hidden="true" className={['mt-0.5 shrink-0', iconClasses[variant]].join(' ')}>
        {icons[variant]}
      </span>
      <div className="min-w-0">
        {title ? <div className="font-semibold text-fg">{title}</div> : null}
        <div className={title ? 'mt-1 text-muted' : 'text-fg'}>{children}</div>
      </div>
    </div>
  );
};
