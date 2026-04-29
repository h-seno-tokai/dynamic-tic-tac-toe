// Public surface of the domain layer.

export type {
  Board,
  Cell,
  GameEngine,
  GameRules,
  GameState,
  Move,
  MoveOnBoardMove,
  Piece,
  PieceSize,
  PlaceFromReserveMove,
  Player,
  Position,
  Reserve,
  WinCondition,
} from './types';

export {
  AI_LIMITS,
  PRESET_3X3,
  PRESET_4X4_XL,
  RULE_PRESETS,
  isRuleSupportedByAI,
} from './rules/presets';
export type { RulePresetId } from './rules/presets';

export { engine } from './engine/engine';
export { hashState } from './engine/hash';
