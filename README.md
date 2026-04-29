# Dynamic Tic-Tac-Toe

> Gobblet を題材にした拡張可能なボードゲームの Web 実装。AlphaZero 系の強化学習による CPU 対戦を搭載。

[![Status](https://img.shields.io/badge/status-design%20complete-brightgreen)](./docs)
[![License](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![ONNX Runtime Web](https://img.shields.io/badge/ONNX%20Runtime%20Web-005CED)](https://onnxruntime.ai/docs/tutorials/web/)

🚀 **Live Demo**: Coming soon

---

🇬🇧 **English summary** (日本語版は下):
> A web implementation of "Dynamic Tic-Tac-Toe", a Gobblet-inspired strategic board game, featuring local 2-player and 10-tier CPU opponents trained from scratch via **AlphaZero-style self-play reinforcement learning** (PyTorch on RTX 2080). Inference runs **entirely in the browser** via ONNX Runtime Web with a custom MCTS in a TypeScript Web Worker — no backend, no server cost, hosted on GitHub Pages. Frontend: React 18 + Vite + Tailwind + Framer Motion + react-i18next (JA/EN). Currently in **design-complete phase**; the design documents themselves in [`docs/`](./docs/) are part of the deliverable.

---

## 概要

「Dynamic Tic-Tac-Toe」は、ボードゲーム **Gobblet（ゴブレット）** を題材にした Web ゲームです。
ローカル2人対戦に加え、**AlphaZero 系の強化学習で自己対戦から学習させた CPU**（10段階の難易度）と対戦できます。

ML モデルはローカル GPU（RTX 2080）で事前学習し、ブラウザ内で **ONNX Runtime Web** により推論します。サーバ不要、無料の GitHub Pages のみで完結します。

> 💡 **設計フェーズの成果物そのものをポートフォリオの一部として提示しています**。ウォーターフロー方式で要件定義から詳細設計までを段階的に commit し、実装前の最終レビュー（複数エージェントによる整合性・実現可能性・ポートフォリオ品質チェック）も実施済みです。実装に入る前のフェーズの質も評価対象として [`docs/`](./docs/) を参照してください。

## 特徴

- 🎯 **ローカル2人対戦 / CPU対戦**（難易度10段階）
- 🤖 **AlphaZero 系強化学習**：自己対戦のみで学習（完全教師無し）
- 🧠 **ブラウザ内 ML 推論**：ONNX Runtime Web + TypeScript MCTS in Web Worker
- 🪶 **コンパクトモデル**：約 316K params / ~1.2 MB ONNX、モバイルでも高速推論
- ⚙️ **2 プリセットルール**：3x3 クラシック / 4x4 巨大入り
- 🌐 **多言語**：日本語 / 英語
- 🎨 **テーマ**：ダーク / ライト（OS 設定尊重）
- 📱 **モバイル対応**：レスポンシブ + タッチ操作
- ♿ **アクセシビリティ**：WCAG 2.1 AA レベル目標

## アーキテクチャ

```mermaid
flowchart TB
    subgraph Browser["Browser (Client only)"]
        UI["React UI<br/>(main thread)"]
        Store["Zustand store"]
        Domain["Domain<br/>rule engine (pure TS)"]

        UI <--> Store
        Store <--> Domain

        subgraph Worker["Web Worker (AI thread)"]
            MCTS["MCTS<br/>(TypeScript)"]
            Inference["ONNX Runtime Web"]
            MCTS <--> Inference
        end

        UI <-.->|requestId-tagged<br/>postMessage| Worker
    end

    Inference -.->|fetch| Model["Trained model<br/>(.onnx, ~1.2 MB)"]

    style Browser fill:#f9f9ff,stroke:#5a5a8a
    style Worker fill:#fff5e6,stroke:#aa6e0e
    style Model fill:#e8f5e9,stroke:#2e7d32
```

| レイヤ | 技術 |
|---|---|
| フロントエンド | React 18 + TypeScript + Vite |
| 状態管理 | Zustand |
| スタイル | Tailwind CSS + CSS variables |
| アニメーション | Framer Motion |
| i18n | react-i18next |
| 音声 | Howler.js |
| ML 推論 | ONNX Runtime Web |
| ML 学習 | PyTorch（オフライン、RTX 2080） |
| ホスティング | GitHub Pages |
| CI/CD | GitHub Actions |

詳細は [docs/](docs/) 参照。

## 開発状況

**設計フェーズ完了**。実装フェーズ開始前。ウォーターフロー方式で進行中。

| フェーズ | ステータス |
|---|---|
| 要件定義 | ✅ 完了 |
| 基本設計 | ✅ 完了 |
| 詳細設計 | ✅ 完了 |
| 実装前レビュー（マルチエージェント） | ✅ 完了 |
| 実装 | 🚧 次のステップ |
| 学習・チューニング | ⏳ 未着手 |
| デプロイ | ⏳ 未着手 |

## 設計ドキュメント

設計フェーズの成果物として、以下を `docs/` に集約しています。**ポートフォリオ評価対象**です。

- [📑 設計ドキュメント インデックス](docs/README.md)
- [01. 要件定義](docs/01_requirements.md) — 機能・非機能要件・スコープ
- [02. 未決事項リスト](docs/02_open_questions.md)
- [03. アーキテクチャ](docs/03_architecture.md) — 8 個の ADR を含む
- [04. 技術スタック](docs/04_tech_stack.md)
- [05. UI 設計](docs/05_ui_design.md)
- [06. データモデル](docs/06_data_model.md) — ルール駆動の拡張性設計
- [07. CPU / 強化学習設計](docs/07_ai_design.md) — **AlphaZero ネットワーク詳細（4×4×27 入力 / ResNet 4×64 / 320 行動空間 / 10段階難易度）**
- [08. アセット計画](docs/08_assets.md)
- [09. コーディング規約](docs/09_conventions.md) — 汎用性最大化のコンポーネント設計指針
- [10. リスク管理](docs/10_risks.md) — 実装前レビューの指摘と緩和策

## ライセンス

[MIT License](LICENSE)
