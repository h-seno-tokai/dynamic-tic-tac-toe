/**
 * ONNX Runtime Web wrapper for the universal DTTT network.
 *
 * Forward signature mirrors `ai-training/src/dttt_train/network.py`:
 *   input  : Float32Array of length 27*4*4 = 432, shape (1, 27, 4, 4)
 *   output : { policy: Float32Array(320) raw logits, value: scalar in [-1, 1] }
 *
 * The ONNX graph emits **raw policy logits** (no mask, no softmax) and a
 * tanh'd value. The legal-action mask is applied in TS by callers (MCTS).
 */

import * as ort from 'onnxruntime-web';
import { MAX_BOARD, NUM_CHANNELS, TENSOR_LENGTH, TOTAL_ACTIONS } from './encoding';

export interface InferenceOutput {
  policy: Float32Array;
  value: number;
}

/** Names match the conventional torch.onnx.export defaults; configurable. */
export interface InferenceEngineOptions {
  inputName?: string;
  policyOutputName?: string;
  valueOutputName?: string;
  /** Override the executionProviders fallback list. */
  executionProviders?: ort.InferenceSession.ExecutionProviderConfig[];
}

const DEFAULT_PROVIDERS: ort.InferenceSession.ExecutionProviderConfig[] = ['webgpu', 'wasm'];

export class InferenceEngine {
  private session: ort.InferenceSession | null = null;
  private inputName = 'input';
  private policyOutputName = 'policy';
  private valueOutputName = 'value';
  private readonly providers: ort.InferenceSession.ExecutionProviderConfig[];

  constructor(options: InferenceEngineOptions = {}) {
    if (options.inputName) this.inputName = options.inputName;
    if (options.policyOutputName) this.policyOutputName = options.policyOutputName;
    if (options.valueOutputName) this.valueOutputName = options.valueOutputName;
    this.providers = options.executionProviders ?? DEFAULT_PROVIDERS;
  }

  /** Loads the ONNX model. Tries WebGPU first, falls back to WASM. */
  async load(modelUrl: string): Promise<void> {
    let lastError: unknown = null;
    for (const provider of this.providers) {
      try {
        const session = await ort.InferenceSession.create(modelUrl, {
          executionProviders: [provider],
          graphOptimizationLevel: 'all',
        });
        this.session = session;
        // Auto-detect input/output names if defaults don't match.
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
        lastError = err;
      }
    }
    throw new Error(`InferenceEngine.load failed for all providers: ${String(lastError)}`);
  }

  isReady(): boolean {
    return this.session !== null;
  }

  /** Runs a single forward pass. Input MUST be length 432 in CHW order. */
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
    return {
      policy: new Float32Array(policyData),
      value: valueData[0] ?? 0,
    };
  }
}
