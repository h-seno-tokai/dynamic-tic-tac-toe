import type { SupportedLanguage } from '../../infra';

export interface PlayerSetupFormProps {
  playerLabel: string;
  name: string;
  onNameChange: (next: string) => void;
  avatarSeed: string;
  avatarSeeds: readonly string[];
  onAvatarSeedChange: (next: string) => void;
  language: SupportedLanguage;
  className?: string;
}

const text = {
  ja: {
    name: '名前',
    avatar: 'アバター',
    chooseAvatar: (seed: string) => `アバター ${seed} を選択`,
  },
  en: {
    name: 'Name',
    avatar: 'Avatar',
    chooseAvatar: (seed: string) => `Choose avatar ${seed}`,
  },
} as const;

const getAvatarInitial = (seed: string) => seed.trim().slice(0, 2).toUpperCase() || '?';

export const PlayerSetupForm = ({
  playerLabel,
  name,
  onNameChange,
  avatarSeed,
  avatarSeeds,
  onAvatarSeedChange,
  language,
  className,
}: PlayerSetupFormProps) => {
  const copy = text[language];
  const nameId = `${playerLabel.replace(/\s+/g, '-').toLowerCase()}-name`;

  return (
    <fieldset className={['flex flex-col gap-4', className].filter(Boolean).join(' ')}>
      <legend className="text-sm font-semibold text-fg">{playerLabel}</legend>
      <div className="flex flex-col gap-1">
        <label htmlFor={nameId} className="text-sm font-medium text-fg">
          {copy.name}
        </label>
        <input
          id={nameId}
          type="text"
          value={name}
          onChange={(event) => onNameChange(event.target.value)}
          autoComplete="nickname"
          className={[
            'h-10 rounded-md border border-border bg-bg px-3 text-sm text-fg',
            'outline-none transition-colors placeholder:text-muted',
            'focus:ring-accent/30 focus:border-accent focus:ring-2',
          ].join(' ')}
        />
      </div>
      <fieldset className="flex flex-col gap-2">
        <legend className="text-sm font-medium text-fg">{copy.avatar}</legend>
        <div className="flex flex-wrap gap-2" role="list">
          {avatarSeeds.map((seed) => {
            const selected = seed === avatarSeed;
            return (
              <button
                key={seed}
                type="button"
                aria-label={copy.chooseAvatar(seed)}
                aria-pressed={selected}
                title={seed}
                onClick={() => onAvatarSeedChange(seed)}
                className={[
                  'flex h-10 w-10 items-center justify-center rounded-md border text-xs font-semibold',
                  'focus:ring-accent/40 transition-colors focus:outline-none focus:ring-2',
                  selected
                    ? 'border-accent bg-[color:var(--color-cell-light)] text-fg'
                    : 'border-border bg-bg text-muted hover:bg-[color:var(--color-cell-light)] hover:text-fg',
                ].join(' ')}
              >
                {getAvatarInitial(seed)}
              </button>
            );
          })}
        </div>
      </fieldset>
    </fieldset>
  );
};
