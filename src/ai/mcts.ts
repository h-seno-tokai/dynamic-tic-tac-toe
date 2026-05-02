/**
 * AlphaZero-style PUCT MCTS over the 4x4 ResNet (3x3 uses Solver3x3 instead).
 *
 * Tree node contract:
 *   N : visit count
 *   W : cumulative value (from parent's perspective)
 *   P : prior probability (post-mask, post-softmax)
 *   children: lazily expanded, keyed by 320-d action index
 *
 * Apply the legal-action mask BEFORE the prior softmax (graph emits raw
 * logits, so we add `-Infinity` on illegal slots prior to softmax).
 *
 * Dirichlet noise on the root is *off* by default (we ship inference-only).
 */

import type { GameState, Move } from '@/domain';
import { engine } from '@/domain';
import {
  TOTAL_ACTIONS,
  actionIndexToMove,
  encodeState,
  legalActionMask,
  moveToActionIndex,
} from './encoding';
import type { InferenceEngine } from './inference';

const DEFAULT_PUCT_C = 1.5;

interface MctsNode {
  state: GameState;
  /** Player to move at this node (1 = P1, -1 = P2). Used for value sign. */
  toMoveSign: 1 | -1;
  /** Visit count of this node. */
  n: number;
  /** Sum of backed-up values from this node's perspective. */
  w: number;
  /** Map from action index -> child node (undefined if not expanded). */
  children: Map<number, MctsNode>;
  /** Prior probability per action index (length TOTAL_ACTIONS). 0 for illegal. */
  priors: Float32Array | null;
  /** Cached legal action indices (for efficient selection). */
  legalActions: number[];
  /** Terminal state cache. */
  terminalValue: number | null;
  /** Whether this node has been expanded (priors evaluated). */
  expanded: boolean;
}

export interface MctsOptions {
  puctC?: number;
}

export interface MctsSearchOptions {
  signal?: AbortSignal;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw new DOMException('MCTS search aborted', 'AbortError');
  }
}

function toMoveSign(state: GameState): 1 | -1 {
  return state.toMove === 'P1' ? 1 : -1;
}

/** Outcome from `engine.outcome()` translated to a value in [-1, 1] from P1's
 * perspective. Caller flips sign for the side-to-move. */
function terminalValueFromP1(state: GameState): number | null {
  const o = state.outcome;
  if (o === null) return null;
  if (o === 'draw') return 0;
  return o === 'P1' ? 1 : -1;
}

function makeNode(state: GameState): MctsNode {
  const tv = terminalValueFromP1(state);
  return {
    state,
    toMoveSign: toMoveSign(state),
    n: 0,
    w: 0,
    children: new Map(),
    priors: null,
    legalActions: [],
    terminalValue: tv,
    expanded: false,
  };
}

/** Apply mask to logits and softmax in place. Returns new array length 320. */
function maskedSoftmax(logits: Float32Array, mask: Float32Array): Float32Array {
  const out = new Float32Array(logits.length);
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (mask[i] === 1) {
      const v = logits[i] ?? 0;
      if (v > max) max = v;
    }
  }
  if (!Number.isFinite(max)) {
    // No legal actions: leave as zeros.
    return out;
  }
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    if (mask[i] === 1) {
      const v = Math.exp((logits[i] ?? 0) - max);
      out[i] = v;
      sum += v;
    }
  }
  if (sum > 0) {
    for (let i = 0; i < out.length; i++) {
      out[i] = (out[i] ?? 0) / sum;
    }
  }
  return out;
}

export class MCTS {
  private readonly engineRef: InferenceEngine;
  private readonly puctC: number;

  constructor(engineRef: InferenceEngine, options: MctsOptions = {}) {
    this.engineRef = engineRef;
    this.puctC = options.puctC ?? DEFAULT_PUCT_C;
  }

  /**
   * Run `simCount` simulations from `state`. Returns a 320-d distribution
   * proportional to the root visit counts (zeros for illegal actions).
   */
  async search(
    state: GameState,
    simCount: number,
    options: MctsSearchOptions = {},
  ): Promise<Float32Array> {
    const root = makeNode(state);
    throwIfAborted(options.signal);
    await this.expand(root);

    for (let i = 0; i < simCount; i++) {
      throwIfAborted(options.signal);
      await this.simulate(root, options.signal);
    }

    const visits = new Float32Array(TOTAL_ACTIONS);
    let total = 0;
    for (const [actionIdx, child] of root.children) {
      visits[actionIdx] = child.n;
      total += child.n;
    }
    if (total > 0) {
      for (let i = 0; i < visits.length; i++) {
        visits[i] = (visits[i] ?? 0) / total;
      }
    }
    return visits;
  }

  /**
   * Run search, then sample a move via temperature.
   * `temperature === 0` -> argmax (greedy).
   * `temperature > 0`   -> sample from visit-count^(1/temperature).
   */
  async selectMove(
    state: GameState,
    simCount: number,
    temperature: number,
    options: MctsSearchOptions = {},
  ): Promise<Move> {
    const policy = await this.search(state, simCount, options);
    return this.sampleFromPolicy(state, policy, temperature);
  }

  private sampleFromPolicy(state: GameState, policy: Float32Array, temperature: number): Move {
    const legalMoves = engine.legalMoves(state);
    if (legalMoves.length === 0) {
      throw new Error('MCTS.selectMove: no legal moves');
    }

    if (temperature <= 0) {
      let bestIdx = -1;
      let bestVal = -1;
      for (let i = 0; i < policy.length; i++) {
        const v = policy[i] ?? 0;
        if (v > bestVal) {
          bestVal = v;
          bestIdx = i;
        }
      }
      if (bestIdx < 0 || bestVal <= 0) {
        // Fallback: pick the first legal move.
        const fallback = legalMoves[0];
        if (!fallback) throw new Error('MCTS.selectMove: no legal moves');
        return fallback;
      }
      // Find the matching legal move (decoded action may not be unique-key
      // since the same Move object identity differs).
      for (const mv of legalMoves) {
        if (moveToActionIndex(mv, state) === bestIdx) return mv;
      }
      const first = legalMoves[0];
      if (!first) throw new Error('MCTS.selectMove: no legal moves');
      return first;
    }

    // Temperature sampling over legal actions only.
    const exponent = 1 / temperature;
    const weights: number[] = [];
    let sum = 0;
    for (const mv of legalMoves) {
      const idx = moveToActionIndex(mv, state);
      const p = Math.pow(policy[idx] ?? 0, exponent);
      weights.push(p);
      sum += p;
    }
    if (sum <= 0) {
      const first = legalMoves[0];
      if (!first) throw new Error('MCTS.selectMove: no legal moves');
      return first;
    }
    let r = Math.random() * sum;
    for (let i = 0; i < legalMoves.length; i++) {
      r -= weights[i] ?? 0;
      if (r <= 0) {
        const mv = legalMoves[i];
        if (mv) return mv;
      }
    }
    const last = legalMoves[legalMoves.length - 1];
    if (!last) throw new Error('MCTS.selectMove: no legal moves');
    return last;
  }

  /** Single PUCT simulation rooted at `node`. Returns leaf value at `node`'s perspective. */
  private async simulate(node: MctsNode, signal: AbortSignal | undefined): Promise<number> {
    throwIfAborted(signal);

    if (node.terminalValue !== null) {
      // Convert P1-perspective terminal value to node's perspective.
      const v = node.terminalValue * node.toMoveSign;
      node.n += 1;
      node.w += v;
      return v;
    }

    if (!node.expanded) {
      const v = await this.expand(node);
      throwIfAborted(signal);
      node.n += 1;
      node.w += v;
      return v;
    }

    const actionIdx = this.selectAction(node);
    if (actionIdx < 0) {
      // No legal action (shouldn't happen for non-terminal but be safe).
      node.n += 1;
      return 0;
    }

    let child = node.children.get(actionIdx);
    if (!child) {
      const mv = actionIndexToMove(actionIdx, node.state);
      const nextState = engine.applyMove(node.state, mv);
      child = makeNode(nextState);
      node.children.set(actionIdx, child);
    }

    // Child value is from child's perspective; flip for backup.
    const childValue = await this.simulate(child, signal);
    const v = -childValue;
    node.n += 1;
    node.w += v;
    return v;
  }

  /** Expand a node: evaluate priors + value via the network. */
  private async expand(node: MctsNode): Promise<number> {
    const input = encodeState(node.state);
    const { policy, value } = await this.engineRef.forward(input);

    const mask = legalActionMask(node.state);
    const priors = maskedSoftmax(policy, mask);
    node.priors = priors;

    const legal: number[] = [];
    for (let i = 0; i < TOTAL_ACTIONS; i++) {
      if (mask[i] === 1) legal.push(i);
    }
    node.legalActions = legal;
    node.expanded = true;

    // The network's value is from the side-to-move's perspective (tanh'd).
    return value;
  }

  /** Pick the action with the highest PUCT score among the legal set. */
  private selectAction(node: MctsNode): number {
    if (!node.priors || node.legalActions.length === 0) return -1;
    const sqrtN = Math.sqrt(Math.max(1, node.n));
    let bestIdx = -1;
    let bestScore = -Infinity;
    for (const a of node.legalActions) {
      const child = node.children.get(a);
      const childN = child ? child.n : 0;
      const childQ = child && child.n > 0 ? child.w / child.n : 0;
      const prior = node.priors[a] ?? 0;
      const u = this.puctC * prior * (sqrtN / (1 + childN));
      // Child W is stored from the child's side-to-move perspective. The
      // parent is the opponent, so negate Q before comparing actions here.
      const score = -childQ + u;
      if (score > bestScore) {
        bestScore = score;
        bestIdx = a;
      }
    }
    return bestIdx;
  }
}
