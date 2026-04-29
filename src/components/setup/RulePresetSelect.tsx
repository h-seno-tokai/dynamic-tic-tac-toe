import type { ReactNode } from 'react';
import { RadioGroup } from '../primitives';
import { RULE_PRESETS } from '../../domain/rules/presets';
import type { RulePresetId } from '../../domain/rules/presets';
import type { GameRules, WinCondition } from '../../domain';
import type { SupportedLanguage } from '../../infra';

type RulePreset = (typeof RULE_PRESETS)[number];

export interface RulePresetSelectProps {
  value: RulePresetId;
  onChange: (next: RulePresetId) => void;
  language: SupportedLanguage;
  presets?: readonly RulePreset[];
  label?: string;
  className?: string;
}

const text = {
  ja: {
    label: 'ルール',
    lineOfN: (n: number) => `${n}目並べ`,
    customWin: 'カスタム勝利条件',
    board: (size: number) => `${size} x ${size}`,
    pieces: (count: number) => `各プレイヤー ${count} 個`,
  },
  en: {
    label: 'Rules',
    lineOfN: (n: number) => `Line of ${n}`,
    customWin: 'Custom win condition',
    board: (size: number) => `${size} x ${size}`,
    pieces: (count: number) => `${count} pieces per player`,
  },
} as const;

const formatWinCondition = (condition: WinCondition, language: SupportedLanguage) => {
  if (condition.kind === 'lineOfN') return text[language].lineOfN(condition.n);
  return text[language].customWin;
};

const countPiecesPerPlayer = (rules: GameRules) =>
  rules.piecesPerSize.reduce((total, count) => total + count, 0);

const describePreset = (rules: GameRules, language: SupportedLanguage): ReactNode => {
  const copy = text[language];
  return [
    copy.board(rules.boardSize),
    formatWinCondition(rules.winCondition, language),
    copy.pieces(countPiecesPerPlayer(rules)),
  ].join(' / ');
};

export const RulePresetSelect = ({
  value,
  onChange,
  language,
  presets = RULE_PRESETS,
  label,
  className,
}: RulePresetSelectProps) => (
  <RadioGroup
    value={value}
    onChange={onChange}
    label={label ?? text[language].label}
    {...(className !== undefined ? { className } : {})}
    options={presets.map((preset) => ({
      value: preset.id,
      label: preset.label[language],
      description: describePreset(preset.rules, language),
    }))}
  />
);
