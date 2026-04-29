import { forwardRef, useId } from 'react';
import type { ChangeEvent } from 'react';

export interface SliderProps {
  value: number;
  onChange: (next: number) => void;
  min: number;
  max: number;
  step?: number;
  label: string;
  /** Optional formatter for displayed value (e.g. "60%"). */
  formatValue?: (v: number) => string;
  disabled?: boolean;
  id?: string;
  className?: string;
}

/**
 * Accessible range slider built on a native `<input type="range">` for keyboard support.
 * - Arrow keys / Home / End / PageUp / PageDown handled natively.
 * - `label` is required and rendered visibly; current value is also shown.
 */
export const Slider = forwardRef<HTMLInputElement, SliderProps>(function Slider(
  { value, onChange, min, max, step = 1, label, formatValue, disabled = false, id, className },
  ref,
) {
  const reactId = useId();
  const inputId = id ?? `slider-${reactId}`;
  const valueId = `${inputId}-value`;
  const display = formatValue ? formatValue(value) : String(value);

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const next = Number(e.target.value);
    if (Number.isFinite(next)) onChange(next);
  };

  return (
    <div className={['flex flex-col gap-1', className].filter(Boolean).join(' ')}>
      <div className="flex items-center justify-between text-fg">
        <label htmlFor={inputId} className="text-sm font-medium">
          {label}
        </label>
        <output id={valueId} htmlFor={inputId} className="text-sm tabular-nums text-muted">
          {display}
        </output>
      </div>
      <input
        ref={ref}
        id={inputId}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={handleChange}
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={value}
        aria-valuetext={display}
        aria-describedby={valueId}
        className={[
          'h-2 w-full cursor-pointer appearance-none rounded-full bg-[color:var(--color-cell-dark)]',
          'accent-accent disabled:cursor-not-allowed disabled:opacity-50',
        ].join(' ')}
      />
    </div>
  );
});
