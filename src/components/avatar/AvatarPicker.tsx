import { UserAvatar } from './UserAvatar';

export const DEFAULT_AVATAR_SEEDS = [
  'haru',
  'aoi',
  'ren',
  'mio',
  'sora',
  'yui',
  'kai',
  'nana',
  'riku',
  'ema',
  'toma',
  'hina',
] as const;

export interface AvatarPickerProps {
  value: string;
  onChange: (seed: string) => void;
  seeds?: readonly string[];
  size?: number;
  label?: string;
  className?: string;
  getSeedLabel?: (seed: string, index: number) => string;
}

export function AvatarPicker({
  value,
  onChange,
  seeds = DEFAULT_AVATAR_SEEDS,
  size = 48,
  label = 'Choose avatar',
  className,
  getSeedLabel = (seed) => `Avatar ${seed}`,
}: AvatarPickerProps) {
  return (
    <div role="group" aria-label={label} className={className}>
      <div className="grid grid-cols-4 gap-2 sm:grid-cols-6">
        {seeds.map((seed, index) => {
          const selected = seed === value;
          const seedLabel = getSeedLabel(seed, index);

          return (
            <button
              key={seed}
              type="button"
              aria-label={seedLabel}
              aria-pressed={selected}
              data-seed={seed}
              data-selected={selected ? 'true' : 'false'}
              onClick={() => onChange(seed)}
              className={[
                'inline-flex items-center justify-center rounded-md border bg-bg p-1 transition',
                'hover:bg-[color:var(--color-cell-light)] focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg',
                selected ? 'border-accent' : 'border-border',
              ].join(' ')}
            >
              <UserAvatar seed={seed} size={size} selected={selected} />
            </button>
          );
        })}
      </div>
    </div>
  );
}
