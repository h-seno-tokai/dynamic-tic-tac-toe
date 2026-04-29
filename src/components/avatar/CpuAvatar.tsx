import { Bot } from 'lucide-react';

export type CpuAvatarTier = 'bronze' | 'silver' | 'gold' | 'master';

export interface CpuAvatarProps {
  difficulty: number;
  size?: number;
  label?: string;
  selected?: boolean;
  className?: string;
}

export function normalizeCpuDifficulty(difficulty: number): number {
  if (!Number.isFinite(difficulty)) return 1;
  return Math.min(10, Math.max(1, Math.trunc(difficulty)));
}

export function getCpuAvatarTier(difficulty: number): CpuAvatarTier {
  const level = normalizeCpuDifficulty(difficulty);

  if (level === 10) return 'master';
  if (level >= 7) return 'gold';
  if (level >= 4) return 'silver';
  return 'bronze';
}

const tierClasses: Record<CpuAvatarTier, string> = {
  bronze: 'cpu-avatar-tier-bronze border-[#CD7F32] bg-[#CD7F32]/15 text-[#8A4F1D]',
  silver:
    'cpu-avatar-tier-silver border-[#C0C0C0] bg-[#C0C0C0]/20 text-[#6B7280] outline outline-1 outline-[#C0C0C0]/60',
  gold: 'cpu-avatar-tier-gold border-[#FFD700] bg-[#FFD700]/20 text-[#9A6B00] shadow-[0_0_14px_rgba(255,215,0,0.35)]',
  master:
    'cpu-avatar-tier-master animate-pulse border-slate-950 bg-slate-100 text-slate-950 shadow-[0_0_18px_rgba(15,23,42,0.35)]',
};

export function CpuAvatar({
  difficulty,
  size = 56,
  label,
  selected = false,
  className,
}: CpuAvatarProps) {
  const normalizedDifficulty = normalizeCpuDifficulty(difficulty);
  const tier = getCpuAvatarTier(normalizedDifficulty);
  const accessibleLabel = label ?? `CPU difficulty ${normalizedDifficulty}`;
  const iconSize = Math.max(18, Math.round(size * 0.44));

  return (
    <div
      role="img"
      aria-label={accessibleLabel}
      data-difficulty={normalizedDifficulty}
      data-tier={tier}
      data-selected={selected ? 'true' : 'false'}
      style={{ width: size, height: size }}
      className={[
        'inline-flex shrink-0 flex-col items-center justify-center gap-0.5 rounded-md border text-xs font-semibold transition',
        selected ? 'ring-2 ring-accent ring-offset-2 ring-offset-bg' : '',
        tierClasses[tier],
        className,
      ]
        .filter(Boolean)
        .join(' ')}
    >
      <Bot aria-hidden="true" size={iconSize} strokeWidth={2.25} />
      <span aria-hidden="true" className="leading-none">
        LV.{normalizedDifficulty}
      </span>
    </div>
  );
}
