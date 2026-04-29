# 10. リスク管理

識別したリスクと採用した緩和策の記録。

---

## 1. 技術リスク

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

## 2. AlphaZero 学習リスク

### 2.1 自己対戦の無限ループ

- **リスク**: Gobblet は駒を動かし続けられるため「合法手なし」はほぼ発生しない
- **緩和策**: `GameRules.maxPly`（3x3=60, 4x4=120）と threefold repetition を組み込み済み（`06_data_model.md`）

### 2.2 学習時間

- **見積もり**: 3x3 Gobblet は Connect4 と同等オーダー、4x4 でもチェスより遥かに小さい探索空間
- **実績**: RTX 2080 で数日〜1週間で実用棋力に到達。UI 実装と並行で学習を進め、完成後に組み込み
- **観測指標**: Elo curve（自己対戦リーグ）、Policy KL divergence、Value loss

### 2.3 プリセット間の難易度差

- **リスク**: 50/50 サンプリングだと探索深い 4x4 が伸び悩む可能性
- **緩和策**: Elo 差を観察、必要なら 30/70（4x4 寄り）にシフト。Dirichlet ノイズ α は平均合法手数でスケール

### 2.4 自己対戦のスループット

- **リスク**: シングルプロセス自己対戦では GPU 利用率が低い
- **緩和策**:
  - **leaf-batched MCTS**（virtual loss）でバッチ推論
  - multiprocessing で複数並列に自己対戦実行（CPU コア数依存）

---

## 3. スコープ

- フルスコープ（両プリセット）で実装完了
- Must（Domain + 対局 + CPU + 設定 + i18n + テーマ + 戦績）はすべて実装済み
- AI 学習・UI 実装・デプロイ完了。学習済みモデル（8×128ch ResNet, ~9.6MB）を配信中

---

## 4. 技術検証結果

| 検証項目                            | 結果                                         |
| ----------------------------------- | -------------------------------------------- |
| iPhone 実機 ONNX Runtime Web ベンチ | 問題なし（WASM SIMD で推論速度確保）         |
| PyTorch → ONNX 3経路パリティテスト  | 実施済み（`max abs diff < 1e-4` 確認）       |
| Web Worker `postMessage` レイテンシ | requestId タグ付き実装で問題なし             |
| GitHub Actions ビルド・デプロイ     | pnpm + Vite、GitHub Pages 自動デプロイ稼働中 |
