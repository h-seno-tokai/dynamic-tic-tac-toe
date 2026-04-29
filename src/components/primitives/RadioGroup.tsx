import { useCallback, useId, useRef } from 'react';
import type { KeyboardEvent, ReactNode } from 'react';

export interface RadioOption<T extends string> {
  value: T;
  label: ReactNode;
  description?: ReactNode;
  disabled?: boolean;
}

export interface RadioGroupProps<T extends string> {
  value: T;
  onChange: (next: T) => void;
  options: RadioOption<T>[];
  /** Group accessible label. If omitted, provide `aria-labelledby`. */
  label?: string;
  'aria-label'?: string;
  'aria-labelledby'?: string;
  orientation?: 'horizontal' | 'vertical';
  name?: string;
  className?: string;
  disabled?: boolean;
}

/**
 * WAI-ARIA `radiogroup` with roving tabindex.
 * - Single tab stop (the currently selected option, or first enabled if no selection).
 * - Arrow keys (Left/Up = prev, Right/Down = next) wrap around.
 * - Home / End jump to first / last enabled option.
 * - Space / Enter select the focused option.
 */
export const RadioGroup = <T extends string>({
  value,
  onChange,
  options,
  label,
  'aria-label': ariaLabel,
  'aria-labelledby': ariaLabelledBy,
  orientation = 'vertical',
  name,
  className,
  disabled: groupDisabled = false,
}: RadioGroupProps<T>) => {
  const reactId = useId();
  const groupName = name ?? `radiogroup-${reactId}`;
  const labelId = label ? `${groupName}-label` : undefined;
  const itemRefs = useRef(new Map<T, HTMLButtonElement | null>());

  const enabledValues = options.filter((o) => !o.disabled).map((o) => o.value);

  const focusValue = useCallback((v: T) => {
    const el = itemRefs.current.get(v);
    if (el) el.focus();
  }, []);

  const moveFocus = useCallback(
    (current: T, dir: 1 | -1) => {
      if (enabledValues.length === 0) return;
      const idx = enabledValues.indexOf(current);
      const start = idx === -1 ? 0 : idx;
      const next = (start + dir + enabledValues.length) % enabledValues.length;
      const target = enabledValues[next];
      if (target !== undefined) {
        onChange(target);
        focusValue(target);
      }
    },
    [enabledValues, onChange, focusValue],
  );

  const handleKeyDown = (e: KeyboardEvent<HTMLButtonElement>, optionValue: T) => {
    if (groupDisabled) return;
    const isHorizontal = orientation === 'horizontal';
    const nextKey = isHorizontal ? 'ArrowRight' : 'ArrowDown';
    const prevKey = isHorizontal ? 'ArrowLeft' : 'ArrowUp';

    if (
      e.key === nextKey ||
      (!isHorizontal && e.key === 'ArrowRight') ||
      (isHorizontal && e.key === 'ArrowDown')
    ) {
      e.preventDefault();
      moveFocus(optionValue, 1);
    } else if (
      e.key === prevKey ||
      (!isHorizontal && e.key === 'ArrowLeft') ||
      (isHorizontal && e.key === 'ArrowUp')
    ) {
      e.preventDefault();
      moveFocus(optionValue, -1);
    } else if (e.key === 'Home') {
      e.preventDefault();
      const first = enabledValues[0];
      if (first !== undefined) {
        onChange(first);
        focusValue(first);
      }
    } else if (e.key === 'End') {
      e.preventDefault();
      const last = enabledValues[enabledValues.length - 1];
      if (last !== undefined) {
        onChange(last);
        focusValue(last);
      }
    } else if (e.key === ' ' || e.key === 'Enter') {
      e.preventDefault();
      onChange(optionValue);
    }
  };

  // Determine which option owns the tab stop (roving tabindex):
  // selected if present and enabled, else first enabled option.
  const selectedEnabled = options.find((o) => o.value === value && !o.disabled) !== undefined;
  const tabStopValue: T | undefined = selectedEnabled ? value : enabledValues[0];

  return (
    <div
      role="radiogroup"
      aria-label={label ? undefined : ariaLabel}
      aria-labelledby={ariaLabelledBy ?? labelId}
      aria-orientation={orientation}
      aria-disabled={groupDisabled || undefined}
      className={className}
    >
      {label && (
        <div id={labelId} className="mb-2 text-sm font-medium text-fg">
          {label}
        </div>
      )}
      <div
        className={[
          'flex',
          orientation === 'horizontal' ? 'flex-row gap-3' : 'flex-col gap-2',
        ].join(' ')}
      >
        {options.map((opt) => {
          const isChecked = opt.value === value;
          const isDisabled = groupDisabled || opt.disabled === true;
          const isTabStop = tabStopValue === opt.value && !isDisabled;
          return (
            <button
              key={opt.value}
              ref={(el) => {
                itemRefs.current.set(opt.value, el);
              }}
              type="button"
              role="radio"
              aria-checked={isChecked}
              aria-disabled={isDisabled || undefined}
              disabled={isDisabled}
              tabIndex={isTabStop ? 0 : -1}
              data-state={isChecked ? 'checked' : 'unchecked'}
              onClick={() => {
                if (!isDisabled) onChange(opt.value);
              }}
              onKeyDown={(e) => handleKeyDown(e, opt.value)}
              className={[
                'flex flex-col items-start gap-0.5 rounded-md border px-3 py-2 text-left',
                'transition-colors',
                isChecked
                  ? 'border-accent bg-[color:var(--color-cell-light)] text-fg'
                  : 'border-border bg-bg text-fg hover:bg-[color:var(--color-cell-light)]',
                'disabled:cursor-not-allowed disabled:opacity-50',
              ].join(' ')}
            >
              <span className="text-sm font-medium">{opt.label}</span>
              {opt.description != null && (
                <span className="text-xs text-muted">{opt.description}</span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
};
