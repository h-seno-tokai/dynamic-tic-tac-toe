# 03. アーキテクチャ設計

## 1. 概要

完全クライアント完結の **SPA（Single Page Application）**。サーバ側ロジックは持たない。
GitHub Pages 上に静的ファイルとしてデプロイし、モデル重みも同一オリジンから配信する。

```
┌──────────────────────────────────────────┐
│  Browser (Client)                        │
│                                          │
│  ┌────────────────────────────────────┐  │
│  │  React UI Layer (Main thread)      │  │
│  │   - 画面・コンポーネント            │  │
│  │   - アニメーション                  │  │
│  └──────────────┬─────────────────────┘  │
│                 │ Zustand store           │
│  ┌──────────────┴─────────────────────┐  │
│  │  Application Layer                 │  │
│  │   - Game flow / 設定 / 戦績        │  │
│  └──────────────┬─────────────────────┘  │
│                 │                         │
│  ┌──────────────┴─────────────────────┐  │
│  │  Domain Layer (Pure TS)            │  │
│  │   - GameRules / Board / Move       │  │
│  │   - Rule-driven Engine (拡張可能)   │  │
│  └────────────────────────────────────┘  │
│                                          │
│  ┌────────────────────────────────────┐  │
│  │  Web Worker (AI thread)            │  │
│  │   - MCTS                           │  │
│  │   - ONNX Runtime Web (推論)        │  │
│  └────────────────────────────────────┘  │
│                                          │
│  ┌────────────────────────────────────┐  │
│  │  Infrastructure                    │  │
│  │   - i18n / Audio / Storage / Theme │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
            │
            │ fetch (model weights)
            ▼
   GitHub Pages (Static)
```

## 2. レイヤ構成

### 2.1 Domain（純粋関数・依存ゼロ）

- `GameRules`（盤面サイズ・駒サイズ・各サイズの個数 等の設定）
- `Board`、`Cell`、`Piece`、`Move` 等のデータ型
- ルールエンジン: `applyMove(state, move)`、`legalMoves(state)`、`isWin(state)` 等
- **ハードコードされた数値（3, 大中小 等）を持たない**。すべて GameRules オブジェクト経由
- React/DOM 非依存。ユニットテストしやすい。

### 2.2 AI

- **Web Worker** で実行。メインスレッドをブロックしない。
- 構成:
  - MCTS 探索ループ（TypeScript 自前実装）
  - ONNX Runtime Web による Policy / Value 推論
  - 難易度パラメータに応じた振る舞い（チェックポイント・シミュレーション数・温度）
- メインスレッドとは `postMessage` で対話：
  - `request: { requestId, state, stateHash, difficulty, timeBudgetMs }`
  - `response: { requestId, move, thinkingMs, principalVariation }`
- 思考の中断 (abort signal) に対応（投了・ウィンドウ閉じ時）
- **レースコンディション対策**: 「待った」で局面が巻き戻った直後に古い思考結果が返る可能性があるため、
  メインスレッド側は応答の `requestId` を現在の最新リクエスト ID と照合し、stale な応答は無条件で破棄する。
  併せて `stateHash` も比較して二重ガードする。

### 2.3 Application（状態管理）

- **Zustand** ストアで構成
- 主要ストア:
  - `gameStore`: 現在の対局状態・履歴・残り駒
  - `settingsStore`: BGM/SFX 音量・テーマ・言語
  - `statsStore`: 戦績（localStorage 永続化）
  - `sessionStore`: ローカル2人対戦のプレイヤー情報（揮発でもよい）
- ストアは Domain のデータ型を保持。Domain ロジックを呼び出す薄い層。

### 2.4 UI（React）

- 関数コンポーネント + Hooks のみ
- 重い計算は Domain / AI に委譲。コンポーネントは表示と入力に集中
- Routing: `react-router-dom` の **HashRouter**（GitHub Pages のディープリンク制限を回避）

### 2.5 Infrastructure

- **i18n**: react-i18next（JA/EN）
- **Audio**: Howler.js（BGM/SFX）
- **Storage**: localStorage 抽象レイヤ（直アクセス禁止、wrapper 経由）
- **Theme**: CSS variables + `data-theme` attribute、Tailwind と統合

## 3. ディレクトリ構成（提案）

```
DTTT/
├── docs/                       # 設計ドキュメント
├── public/
│   ├── models/                 # ONNX 重みファイル
│   ├── audio/                  # BGM/SFX
│   └── avatars/                # アバター画像
├── src/
│   ├── domain/                 # ピュアな型・ルールエンジン
│   │   ├── rules/              # GameRules 型・プリセット
│   │   ├── board.ts
│   │   ├── piece.ts
│   │   ├── move.ts
│   │   └── engine.ts
│   ├── ai/
│   │   ├── worker.ts           # Web Worker エントリ
│   │   ├── mcts.ts
│   │   ├── inference.ts        # ONNX Runtime ラッパ
│   │   └── difficulty.ts       # 難易度プロファイル定義
│   ├── stores/                 # Zustand ストア
│   ├── components/             # React コンポーネント
│   │   ├── board/
│   │   ├── menu/
│   │   ├── settings/
│   │   └── shared/
│   ├── pages/                  # 画面コンポーネント
│   ├── infra/
│   │   ├── i18n/
│   │   ├── audio/
│   │   ├── storage/
│   │   └── theme/
│   ├── hooks/
│   ├── utils/
│   ├── App.tsx
│   └── main.tsx
├── ai-training/                # Python 学習プロジェクト（独立）
│   ├── pyproject.toml
│   ├── selfplay/
│   ├── network/
│   ├── train.py
│   └── export_onnx.py
├── tests/
├── .github/workflows/
│   └── deploy.yml
├── package.json
├── tsconfig.json
├── vite.config.ts
└── README.md
```

**コンポーネント分割の粒度はユーザー側で決定する事項**。本ディレクトリ構成は最小限の枠組みのみ提示。

## 4. ビルド & デプロイ

### 4.1 ビルド

- Vite による静的ビルド
- Tree-shaking と code-splitting で初期ロードを 10MB 以内に
- ONNX 重み・音声・画像はチャンク分離して遅延ロード

### 4.2 デプロイ

- GitHub Actions ワークフロー: push to `main` → ビルド → `gh-pages` ブランチへデプロイ
- PR ごとにビルドのみ実行（プレビューデプロイは GitHub Pages が無料で対応していないので不採用）

### 4.3 学習パイプライン

- `ai-training/` は独立した Python プロジェクト
- 学習は手動実行（CI には載せない、GPU 不要なため）
- 完了後、`export_onnx.py` で ONNX 化 → `public/models/` に配置 → コミット → 自動デプロイ

## 5. 主要な意思決定（ADR 風）

### ADR-001: フロントエンドフレームワーク = React

- 採用: React 18+
- 理由: ユーザー指定。エコシステム・採用実績・i18n/状態管理ライブラリの選択肢が豊富。

### ADR-002: Web Worker による AI 分離

- 採用: AI 計算は専用 Worker で実行
- 理由: MCTS は最大10秒 CPU を占有する。メインスレッドで動かすとアニメーションがカクつき UX が崩壊する。

### ADR-003: ルール駆動の Domain 設計

- 採用: ハードコードを排し、`GameRules` オブジェクトで盤面サイズ・駒構成を駆動
- 理由: ユーザーは 3x3→4x4 や駒サイズ段階の追加（極大等）を要求している。リテラル値の散在は破綻する。

### ADR-004: HashRouter

- 採用: ディープリンクは `/#/game` 形式
- 理由: GitHub Pages は SPA の history routing をサポートしない（404 になる）。HashRouter なら設定ゼロで動く。

### ADR-005: Zustand

- 採用: グローバル状態に Zustand
- 理由: Redux は冗長、Context は再描画のスコープが粗い。Zustand は両者の中間で TS との相性が良い。

### ADR-006: 音声に Howler.js

- 採用: BGM/SFX は Howler.js
- 理由: Web Audio API の差異（特に iOS Safari の自動再生制限）を吸収する。手書きより堅牢。

### ADR-007: アクセシビリティ目標 = WCAG 2.1 AA

- 採用: AA レベルを目標
- 理由: ユーザー指示「適当に」を踏まえ、ポートフォリオ品質として現実的なライン。AAA はコスト過大。

### ADR-008: Python 学習プロジェクトの分離

- 採用: `ai-training/` を Web フロントとは独立したパッケージに
- 理由: 依存（PyTorch・CUDA）がフロントに混入するのを避ける。ビルド・配布の対象外にする。
