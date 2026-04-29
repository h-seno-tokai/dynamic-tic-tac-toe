# 10. リスク管理・運用方針

最終更新: 2026-04-29

実装前のレビュー（`.review-feasibility.md` / `.review-portfolio.md` / `.review-doc-consistency.md` を参照）で識別されたリスクと緩和策を一覧化する。実装フェーズで参照しながら検証していく。

---

## 1. 技術リスク

### 1.1 iOS Safari + ONNX Runtime Web
- **リスク**: GitHub Pages では COOP/COEP ヘッダを設定できないため、`SharedArrayBuffer` が使えず WASM マルチスレッドが無効化される可能性大。
- **影響**: 推論速度が単一スレッド WASM になり、目標の `< 5ms / forward pass` を達成できない可能性。
- **緩和策**:
  1. **実装早期に iPhone 実機ベンチを取る**（最初のスケルトン段階で）
  2. WebGPU が利用可能な場合は優先（iOS 17+ で限定的にサポート）
  3. WASM SIMD を有効化（`onnxruntime-web` のオプション）
  4. 不足する場合は MCTS シミュレーション数を動的に調整（モバイルでは 200 → 100 など）
- **観測指標**: 1 forward pass 平均 ms（iPhone Safari）

### 1.2 ONNX エクスポート互換性
- **リスク**: PyTorch → ONNX → ONNX Runtime Web で挙動が一致しない（NaN / 棋力大幅劣化）
- **緩和策**:
  1. **opset 17** 固定
  2. `model.eval()` モードで export（BN/Dropout を inference 状態に）
  3. `dynamic_axes=None`（入力 4×4×27 で固定）
  4. **行動マスクは ONNX グラフ外（TS 側）で適用** — `−∞ + softmax` の NaN を回避
  5. **3経路パリティテスト**: PyTorch / ONNX Runtime CPU / ONNX Runtime Web の出力一致を `max abs diff < 1e-4` で検証
- **検証時期**: 実装スケルトン段階で最小プロトタイプを整備

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
- **見積もり**: 探索空間が小さい（3x3 Gobblet は Connect4 と同等オーダー、4x4 でもチェスより遥かに小）ため、家庭用 GPU で **数日〜1週間程度** で実用棋力に到達する見込み
- **対応方針**: 学習を Day 1 に開始し、UI 構築と並行で走らせる
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

## 3. スコープ・スケジュール

### 3.1 100時間予算
- **方針**: フルスコープ（両プリセット）で進行（ユーザー決定 = Plan B）
- **優先順位**:
  - **Must**: Domain + 2プリセット対局 + ローカル2人 + CPU対戦（強AI込み） + 設定 + i18n + テーマ + 戦績
  - **Should**: フル E2E、アクセシビリティ詳細、Lighthouse スコア
  - **Could**: 量子化、Mermaid 等の追加 polish、技術ブログ記事
- 100時間内で Must を確実に、Should/Could は時間で調整
- **学習時間とのオーバーラップ**: AI 学習は数日〜1週間程度の想定。UI 実装と並行で走らせ、出来上がった時点で組み込む

### 3.2 学習を待つ間のリリース戦略
- 万一 AI 学習が UI より長引いた場合、UI を先に公開し、初期は弱AI（ランダム/ヒューリスティック）でリリース、その後学習済みモデルを差し込む形でアップデート

---

## 4. 実装スケルトン段階の必須アクション

機能実装に入る前に以下を確認:
- [ ] iPhone 実機での ONNX Runtime Web ベンチ（簡単な MLP で十分）
- [ ] PyTorch → ONNX エクスポート + 3経路パリティテストの最小プロトタイプ
- [ ] Web Worker での `postMessage` レイテンシ測定
- [ ] GitHub Actions のビルド・デプロイパイプラインの試運転（空 React アプリで）

これらが通った後で本格的な機能実装に着手する。

---

## 5. ポートフォリオ品質向上のアクション

### 5.1 設計フェーズ完了時点で対応 (Now)
- README に Live Demo プレースホルダ追加 ✅
- README 英語サマリ追加 ✅
- ASCII 図を Mermaid 化 ✅
- 設計ドキュメントへのアクティブな誘導 ✅

### 5.2 実装フェーズで対応 (Later)
- デモ GIF / スクリーンショット（README 用）
- Lighthouse a11y スコア 100 の screenshot
- 学習カーブ・Elo 推移のプロット
- バンドルサイズ badge
- GitHub About / Topics の設定
- Zenn / Qiita 等での技術記事公開

### 5.3 リポジトリ運用
- GitHub remote 作成タイミングはユーザー判断
- リポジトリ名候補: `dttt` / `dynamic-tic-tac-toe`（未決）
