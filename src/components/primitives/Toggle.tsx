import { forwardRef, useId } from 'react';
import type { KeyboardEvent } from 'react';

export interface ToggleProps {
  checked: boolean;
  onCheckedChange: (next: boolean) => void;
  /** Visible label rendered next to the switch. If omitted, `aria-label` is used. */
  label?: string;
  /** Accessible label when no visible label is rendered. */
  'aria-label'?: string;
  disabled?: boolean;
  id?: string;
  className?: string;
  name?: string;
}

/**
 * Accessible switch-style toggle (WAI-ARIA `switch` role).
 * - Controlled: `checked` + `onCheckedChange`
 * - Space toggles. Enter does NOT toggle (per ARIA switch pattern).
 */
export const Toggle = forwardRef<HTMLButtonElement, ToggleProps>(function Toggle(
  {
    checked,
    onCheckedChange,
    label,
    'aria-label': ariaLabel,
    disabled = false,
    id,
    className,
    name,
  },
  ref,
) {
  const reactId = useId();
  const switchId = id ?? `toggle-${reactId}`;
  const labelId = label ? `${switchId}-label` : undefined;

  const handleClick = () => {
    if (disabled) return;
    onCheckedChange(!checked);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLButtonElement>) => {
    if (disabled) return;
    if (e.key === ' ' || e.key === 'Spacebar') {
      e.preventDefault();
      onCheckedChange(!checked);
    }
  };

  const switchEl = (
    <button
      ref={ref}
      id={switchId}
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label ? undefined : ariaLabel}
      aria-labelledby={labelId}
      aria-disabled={disabled || undefined}
      disabled={disabled}
      data-state={checked ? 'on' : 'off'}
      name={name}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      className={[
        'relative inline-flex h-6 w-11 shrink-0 cursor-pointer items-center rounded-full',
        'border border-border transition-colors',
        checked ? 'bg-accent' : 'bg-[color:var(--color-cell-dark)]',
        'disabled:cursor-not-allowed disabled:opacity-50',
      ].join(' ')}
    >
      <span
        aria-hidden="true"
        className={[
          'pointer-events-none inline-block h-5 w-5 transform rounded-full bg-bg shadow ring-0',
          'transition-transform',
          checked ? 'translate-x-5' : 'translate-x-0.5',
        ].join(' ')}
      />
    </button>
  );

  if (label) {
    return (
      <span className={['inline-flex items-center gap-3', className].filter(Boolean).join(' ')}>
        {switchEl}
        <label id={labelId} htmlFor={switchId} className="cursor-pointer select-none text-fg">
          {label}
        </label>
      </span>
    );
  }

  return <span className={className}>{switchEl}</span>;
});
