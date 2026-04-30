// Public surface of the AI layer.

export {
  MAX_BOARD,
  MAX_PIECE_SIZES,
  MAX_PIECES_PER_SIZE,
  MOVE_ACTIONS,
  NUM_CHANNELS,
  PLACE_ACTIONS,
  TENSOR_LENGTH,
  TOTAL_ACTIONS,
  actionIndexToMove,
  encodeState,
  legalActionMask,
  moveToActionIndex,
} from './encoding';

export { InferenceEngine } from './inference';
export type { InferenceEngineOptions, InferenceOutput } from './inference';

export { MCTS } from './mcts';
export type { MctsOptions } from './mcts';

export { selectFallbackMove } from './fallbackAi';
export type { FallbackRandom } from './fallbackAi';

export {
  DIFFICULTY_PROFILES,
  DIFFICULTY_PROFILES_3X3,
  getProfile,
  getProfile3x3,
} from './difficulty';
export type { Difficulty3x3Profile, Difficulty4x4Profile, DifficultyProfile } from './difficulty';

export { Solver3x3 } from './solver3x3';
export type { Solver3x3Options } from './solver3x3';

export { AiClient } from './workerClient';
export type { AiClientOptions } from './workerClient';

export type { AiRequest, AiResponse } from './protocol';
export { deserializeGameState, serializeGameState } from './serialization';
export type { SerializedGameState } from './serialization';
