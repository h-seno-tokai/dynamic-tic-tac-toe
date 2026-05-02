/**
 * ONNX Runtime Web wrapper for the 4x4 AlphaZero network.
 *
 * Forward signature mirrors `ai-training/src/dttt_train_4x4/network.py`:
 *   input  : Float32Array of length 27*4*4 = 432, shape (1, 27, 4, 4)
 *   output : {
 *     policy : Float32Array(320) raw policy logits  (no mask, no softmax)
 *     wdl    : Float32Array(3)   raw WDL logits     (Win, Draw, Loss)
 *     value  : number  -- scalar in [-1, 1] derived from softmax(wdl):
 *                         value = P(win) - P(loss)
 *   }
 *
 * The legal-action mask is applied in TS by callers (MCTS).
 */

import * as ort from 'onnxruntime-web';
import './ortConfig';
import { MAX_BOARD, NUM_CHANNELS, TENSOR_LENGTH, TOTAL_ACTIONS } from './encoding';

const WDL_OUTPUTS = 3;

export interface InferenceOutput {
  policy: Float32Array;
  wdl: Float32Array;
  value: number;
}

export interface InferenceEngineOptions {
  inputName?: string;
  policyOutputName?: string;
  /** Either 'wdl_logits' (new) or 'value' (legacy scalar). */
  valueOutputName?: string;
  executionProviders?: ort.InferenceSession.ExecutionProviderConfig[];
}

// wasm first: it's universally available; webgpu is best-effort and many
// browsers (Firefox, older Chromium, anything without an enabled flag)
// silently fail to register the provider. Trying wasm first gives the
// fastest path to a working session.
const DEFAULT_PROVIDERS: ort.InferenceSession.ExecutionProviderConfig[] = ['wasm', 'webgpu'];

function softmax3(x: ArrayLike<number>): [number, number, number] {
  const a = x[0] ?? 0;
  const b = x[1] ?? 0;
  const c = x[2] ?? 0;
  const m = Math.max(a, b, c);
  const ea = Math.exp(a - m);
  const eb = Math.exp(b - m);
  const ec = Math.exp(c - m);
  const z = ea + eb + ec;
  return [ea / z, eb / z, ec / z];
}

export class InferenceEngine {
  private session: ort.InferenceSession | null = null;
  private inputName = 'input';
  private policyOutputName = 'policy_logits';
  private valueOutputName = 'wdl_logits';
  private readonly providers: ort.InferenceSession.ExecutionProviderConfig[];

  constructor(options: InferenceEngineOptions = {}) {
    if (options.inputName) this.inputName = options.inputName;
    if (options.policyOutputName) this.policyOutputName = options.policyOutputName;
    if (options.valueOutputName) this.valueOutputName = options.valueOutputName;
    this.providers = options.executionProviders ?? DEFAULT_PROVIDERS;
  }

  async load(modelUrl: string): Promise<void> {
    const errors: string[] = [];
    for (const provider of this.providers) {
      try {
        const session = await ort.InferenceSession.create(modelUrl, {
          executionProviders: [provider],
          graphOptimizationLevel: 'all',
        });
        this.session = session;
        if (!session.inputNames.includes(this.inputName)) {
          const first = session.inputNames[0];
          if (first) this.inputName = first;
        }
        if (!session.outputNames.includes(this.policyOutputName)) {
          // Heuristic: pick the larger output as policy.
          const names = session.outputNames;
          if (names.length >= 2 && names[0] && names[1]) {
            this.policyOutputName = names[0];
            this.valueOutputName = names[1];
          }
        }
        return;
      } catch (err) {
        const providerName = typeof provider === 'string' ? provider : (provider.name ?? 'ep');
        const msg = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
        errors.push(`[${providerName}] ${msg}`);
      }
    }
    throw new Error(`InferenceEngine.load failed for all providers: ${errors.join(' | ')}`);
  }

  isReady(): boolean {
    return this.session !== null;
  }

  async forward(input: Float32Array): Promise<InferenceOutput> {
    if (!this.session) {
      throw new Error('InferenceEngine.forward: model not loaded');
    }
    if (input.length !== TENSOR_LENGTH) {
      throw new Error(
        `InferenceEngine.forward: expected length ${TENSOR_LENGTH}, got ${input.length}`,
      );
    }

    const tensor = new ort.Tensor('float32', input, [1, NUM_CHANNELS, MAX_BOARD, MAX_BOARD]);
    const feeds: Record<string, ort.Tensor> = { [this.inputName]: tensor };
    const results = await this.session.run(feeds);

    const policyT = results[this.policyOutputName];
    const valueT = results[this.valueOutputName];
    if (!policyT || !valueT) {
      throw new Error('InferenceEngine.forward: outputs missing');
    }
    const policyData = policyT.data;
    const valueData = valueT.data;
    if (!(policyData instanceof Float32Array)) {
      throw new Error('InferenceEngine.forward: policy output not float32');
    }
    if (!(valueData instanceof Float32Array)) {
      throw new Error('InferenceEngine.forward: value output not float32');
    }
    if (policyData.length !== TOTAL_ACTIONS) {
      throw new Error(
        `InferenceEngine.forward: policy length ${policyData.length} != ${TOTAL_ACTIONS}`,
      );
    }

    let scalarValue: number;
    let wdl: Float32Array;
    if (valueData.length === WDL_OUTPUTS) {
      // New WDL head: convert (W, D, L) logits -> scalar Q = P(W) - P(L).
      const [pw, , pl] = softmax3(valueData);
      scalarValue = pw - pl;
      wdl = new Float32Array(valueData);
    } else {
      // Legacy scalar value head (tanh) — keep working for old models.
      scalarValue = valueData[0] ?? 0;
      wdl = new Float32Array([
        Math.max(0, scalarValue),
        Math.max(0, 1 - Math.abs(scalarValue)),
        Math.max(0, -scalarValue),
      ]);
    }

    return {
      policy: new Float32Array(policyData),
      wdl,
      value: scalarValue,
    };
  }
}
