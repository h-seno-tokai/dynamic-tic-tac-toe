import { forwardRef } from 'react';
import type { ButtonHTMLAttributes, ReactNode } from 'react';
import { LoadingSpinner } from './LoadingSpinner';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';
export type ButtonSize = 'sm' | 'md' | 'lg';

export type ButtonProps = {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
  iconLeft?: ReactNode;
  iconRight?: ReactNode;
} & ButtonHTMLAttributes<HTMLButtonElement>;

const variantClasses: Record<ButtonVariant, string> = {
  primary:
    'bg-accent text-bg border border-accent hover:opacity-90 active:opacity-80 disabled:opacity-50',
  secondary:
    'bg-bg text-fg border border-border hover:bg-[color:var(--color-cell-light)] disabled:opacity-50',
  ghost:
    'bg-transparent text-fg border border-transparent hover:bg-[color:var(--color-cell-light)] disabled:opacity-50',
  danger: 'bg-p1 text-bg border border-p1 hover:opacity-90 active:opacity-80 disabled:opacity-50',
};

const sizeClasses: Record<ButtonSize, string> = {
  sm: 'h-8 px-3 text-sm gap-1.5',
  md: 'h-10 px-4 text-base gap-2',
  lg: 'h-12 px-6 text-lg gap-2.5',
};

const spinnerSize: Record<ButtonSize, 'sm' | 'md' | 'lg'> = {
  sm: 'sm',
  md: 'sm',
  lg: 'md',
};

/**
 * Themed accessible button.
 * - Variants: primary | secondary | ghost | danger
 * - Sizes: sm | md | lg
 * - `loading` shows a spinner and disables the button.
 */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    variant = 'primary',
    size = 'md',
    loading = false,
    iconLeft,
    iconRight,
    disabled,
    className,
    children,
    type,
    ...rest
  },
  ref,
) {
  const isDisabled = disabled ?? loading;
  const composed = [
    'inline-flex items-center justify-center font-medium rounded-md transition-colors',
    'disabled:cursor-not-allowed select-none whitespace-nowrap',
    variantClasses[variant],
    sizeClasses[size],
    className,
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <button
      ref={ref}
      type={type ?? 'button'}
      className={composed}
      disabled={isDisabled}
      aria-busy={loading || undefined}
      data-variant={variant}
      data-size={size}
      {...rest}
    >
      {loading ? (
        <LoadingSpinner size={spinnerSize[size]} aria-label="Loading" />
      ) : (
        iconLeft && (
          <span aria-hidden="true" className="inline-flex shrink-0">
            {iconLeft}
          </span>
        )
      )}
      {children != null && <span>{children}</span>}
      {!loading && iconRight && (
        <span aria-hidden="true" className="inline-flex shrink-0">
          {iconRight}
        </span>
      )}
    </button>
  );
});
