# 10. リスク管理

識別したリスクと採用した緩和策の記録。

---

## 1. 技術リスク

### 1.0 ORT-Web wasm パスが Vite の SPA フォールバックで HTML 化される（**実発生・修正済み**）

- **リスク**: ORT-Web は wasm を相対パスで取りに行く。Vite dev server / GitHub Pages の SPA フォールバックは
  存在しない wasm パスに対しても HTML（`index.html`）を返してしまう。結果、ORT-Web は HTML を wasm として
  decode しようとして "Wasm decoding failed: expected magic word" で初期化に失敗する。
- **影響**: 4x4 推論が一切起動できない。3x3 にも波及する設計だと CPU 全機能停止。
- **緩和策**:
  1. `src/ai/ortConfig.ts` で Vite の `?url` import 経由で wasm の **絶対 URL** を取得し、`ort.env.wasm.wasmPaths` に固定
  2. Worker の `init` を即時 `ready` 化 + ONNX を **遅延ロード** にして、ロード失敗が 3x3 に波及しないよう分離（`03_architecture.md` §2.2）
- **結果**: 修正済み。4x4 がロードできない環境でも 3x3 はプレイ可能。

### 1.1 iOS Safari + ONNX Runtime Web

- **リスク**: GitHub Pages では COOP/COEP ヘッダを設定できないため、`SharedArrayBuffer` が使えず WASM マルチスレッドが無効化される可能性大。
- **影響**: 推論速度が単一スレッド WASM になり、目標の `< 5ms / forward pass` を達成できない可能性。
- **緩和策**:
  1. iPhone 実機ベンチを実施 → 問題なし（WASM SIMD で推論速度確保）
  2. WebGPU が利用可能な場合は優先（iOS 17+ で限定的にサポート）
  3. WASM SIMD を有効化（`onnxruntime-web` のオプション）
  4. 不足する場合は MCTS シミュレーション数を動的に調整（モバイルでは 200 → 100 など）

### 1.2 ONNX エクスポート互換性

- **リスク**: PyTorch → ONNX → ONNX Runtime Web で挙動が一致しない（NaN / 棋力大幅劣化）
- **緩和策**:
  1. **opset 17** 固定
  2. `model.eval()` モードで export（BN/Dropout を inference 状態に）
  3. `dynamic_axes=None`（入力 4×4×27 で固定）
  4. **行動マスクは ONNX グラフ外（TS 側）で適用** — `−∞ + softmax` の NaN を回避
  5. **3経路パリティテスト**: PyTorch / ONNX Runtime CPU / ONNX Runtime Web の出力一致を `max abs diff < 1e-4` で検証
- **結果**: 3経路パリティテスト実施済み（`max abs diff < 1e-4` 確認）

### 1.3 Web Worker と UI のレースコンディション

- **リスク**: 「待った」直後に古い AI 応答が到着し、巻き戻した局面に古い手が適用される
- **緩和策**:
  - Worker request に `requestId` (uuid) と `stateHash` を付与
  - 応答時に最新 ID と照合、不一致なら破棄
  - 投了時は Worker に `abort` メッセージを送信
- **詳細**: `03_architecture.md` 2.2 参照

---

## 2. AlphaZero 学習リスク（4x4 専用パイプライン）

3x3 は alpha-beta に切り替えたため、本セクションは 4x4 のみが対象。

### 2.1 自己対戦の無限ループ（**緩和済み**）

- **リスク**: Gobblet は駒を動かし続けられるため「合法手なし」はほぼ発生しない
- **緩和策**: `GameRules.maxPly`（4x4=120）と threefold repetition を組み込み済み（`06_data_model.md`）

### 2.2 学習時間（**実績あり**）

- **実績**: RTX 2080 でユニバーサル試行 (run01..09) 後、4x4 専用 net (run10..) で数日規模の連続学習を継続中
- **観測指標**: policy CE / value loss / 自己対戦勝率
- **run11 実績**: policy 4.1 → 2.5、value 0.85 → 0.01（~23,000 steps、`--no-gating`）

### 2.3 旧「プリセット間の難易度差」リスク（**回避済み**）

- 単一ユニバーサルネットで 3x3/4x4 を 50/50 混合する設計だったが、3x3 を alpha-beta に切替えたことで
  **混合自己対戦そのものが不要に**。リスクは消滅。

### 2.4 自己対戦のスループット（**緩和済み**）

- **リスク**: シングルプロセス自己対戦では GPU 利用率が低い
- **緩和策**:
  - NumPy ベースの int8 board buffer + LUT cover-table に書き換え (`dttt_train_4x4/engine.py`):
    `apply_move` 24×、`_state_hash` 26×、`legal_action_mask` 11×、1 ゲーム自己対戦 3.1×
  - `--num-workers 6`（CPU 自己対戦）を主データ源に、`--num-parallel 64`（GPU メイン）と並走
  - leaf-batched MCTS（virtual loss）でバッチ推論

### 2.5 Apprentice deadlock（**実発生・run10 → run11 で緩和**）

- **リスク**: `--gating` ON だと自己対戦は **凍結された step-2000 best net** を使う。候補ネットがその best を
  模倣する方向に収束し、上回れず gating が通らないループに陥る。
- **症状**: run10 で gating 0–40 / step 4000–8000、loss は微改善するが棋力は伸びず。
- **緩和**: run11 で `run10/ckpt_002000` から **`--no-gating`** で再開。自己対戦は **EMA shadow**（候補を緩く追従）
  を使うため、候補が伸びれば自己対戦相手も伸びる。policy 4.1 → 2.5、value 0.85 → 0.01 に到達。
- **教訓**: 教師（凍結 best）と生徒（候補）が同レベルで握手すると進まない。生徒を緩く追う教師が必要。

### 2.6 ユニバーサルネット試行の打ち切り（**設計変更で対処**）

- run01..09 は (1, 27, 4, 4) 入力で 3x3/4x4 両方を扱う 8×128ch CNN を試したが棋力が伸びなかった。
- 4x4 専用に縮小（6×96ch、~1.0M params、~3.92 MB ONNX）+ 3x3 を alpha-beta に分離して打開。
- run01..09 は再現性のためディスク残置、ai-training 側のレガシー `dttt_train/` パッケージも維持。

---

## 3. スコープ

- フルスコープ（両プリセット）で実装完了
- Must（Domain + 対局 + CPU + 設定 + i18n + テーマ + 戦績）はすべて実装済み
- 3x3 は alpha-beta ソルバ（`src/ai/solver3x3.ts`）で完成
- 4x4 は 4x4 専用 AlphaZero ネット (~3.92 MB ONNX, run11/step25000) を `public/model.onnx` に配信
- run12（`--num-sims 400 --lr 5e-4`）で更なる棋力向上を学習中

---

## 4. 技術検証結果

| 検証項目                            | 結果                                         |
| ----------------------------------- | -------------------------------------------- |
| iPhone 実機 ONNX Runtime Web ベンチ | 問題なし（WASM SIMD で推論速度確保）         |
| PyTorch → ONNX 3経路パリティテスト  | 実施済み（`max abs diff < 1e-4` 確認）       |
| Web Worker `postMessage` レイテンシ | requestId タグ付き実装で問題なし             |
| GitHub Actions ビルド・デプロイ     | pnpm + Vite、GitHub Pages 自動デプロイ稼働中 |
