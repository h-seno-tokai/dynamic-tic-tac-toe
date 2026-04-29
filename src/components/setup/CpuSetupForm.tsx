import { RadioGroup, Slider } from '../primitives';
import { RulePresetSelect } from './RulePresetSelect';
import type { RulePresetId } from '../../domain/rules/presets';
import type { Player } from '../../domain';
import type { SupportedLanguage } from '../../infra';

export interface CpuSetupFormProps {
  difficulty: number;
  onDifficultyChange: (next: number) => void;
  humanSide: Player;
  onHumanSideChange: (next: Player) => void;
  rulePresetId: RulePresetId;
  onRulePresetChange: (next: RulePresetId) => void;
  language: SupportedLanguage;
  className?: string;
}

const text = {
  ja: {
    difficulty: 'CPUの強さ',
    difficultyValue: (level: number) => `レベル ${level}`,
    humanSide: 'あなたの手番',
    p1: '先手',
    p2: '後手',
  },
  en: {
    difficulty: 'CPU difficulty',
    difficultyValue: (level: number) => `Level ${level}`,
    humanSide: 'Your side',
    p1: 'First player',
    p2: 'Second player',
  },
} as const;

export const CpuSetupForm = ({
  difficulty,
  onDifficultyChange,
  humanSide,
  onHumanSideChange,
  rulePresetId,
  onRulePresetChange,
  language,
  className,
}: CpuSetupFormProps) => {
  const copy = text[language];

  return (
    <div className={['flex flex-col gap-5', className].filter(Boolean).join(' ')}>
      <Slider
        value={difficulty}
        onChange={onDifficultyChange}
        min={1}
        max={10}
        step={1}
        label={copy.difficulty}
        formatValue={copy.difficultyValue}
      />
      <RadioGroup<Player>
        value={humanSide}
        onChange={onHumanSideChange}
        label={copy.humanSide}
        orientation="horizontal"
        options={[
          { value: 'P1', label: copy.p1 },
          { value: 'P2', label: copy.p2 },
        ]}
      />
      <RulePresetSelect value={rulePresetId} onChange={onRulePresetChange} language={language} />
    </div>
  );
};
