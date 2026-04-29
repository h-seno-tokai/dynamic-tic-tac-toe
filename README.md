# Dynamic Tic-Tac-Toe

> Gobblet を題材にした拡張可能なボードゲームの Web 実装。AlphaZero 系の強化学習による CPU 対戦を搭載予定。

[![Status](https://img.shields.io/badge/status-design%20phase-yellow)](./docs)
[![License](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)

---

## 概要

「Dynamic Tic-Tac-Toe」は、ボードゲーム **Gobblet（ゴブレット）** を題材にした Web ゲームです。
ローカル2人対戦に加え、**AlphaZero 系の強化学習で自己対戦から学習させた CPU**（10段階の難易度）と対戦できます。

ML モデルはローカル GPU（RTX 2080）で事前学習し、ブラウザ内で **ONNX Runtime Web** により推論します。サーバ不要、無料の GitHub Pages のみで完結します。

## 特徴

- 🎯 **ローカル2人対戦 / CPU対戦**（難易度10段階）
- 🤖 **AlphaZero 系強化学習**：自己対戦のみで学習（完全教師無し）
- 🧠 **ブラウザ内 ML 推論**：ONNX Runtime Web による Policy/Value ネット推論 + TypeScript 製 MCTS
- ⚙️ **カスタマイズ可能なルール**：盤面サイズ・駒サイズ段階・各サイズの駒数を変更可
- 🌐 **多言語**：日本語 / 英語
- 🎨 **テーマ**：ダーク / ライト（OS設定尊重）
- 📱 **モバイル対応**：レスポンシブ + タッチ操作
- ♿ **アクセシビリティ**：WCAG 2.1 AA レベル目標

## アーキテクチャ

```
Browser (Client only)
├─ React UI  ←→  Zustand store  ←→  Domain (rule engine)
│
└─ Web Worker
   └─ MCTS  ←→  ONNX Runtime Web  ←→  trained model (.onnx)
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

**設計フェーズ**（実装未着手）。ウォーターフロー方式で進行中。

| フェーズ | ステータス |
|---|---|
| 要件定義 | ✅ 完了 |
| 基本設計（アーキテクチャ・技術選定） | ✅ 完了 |
| 詳細設計 | 🚧 進行中 |
| 実装 | ⏳ 未着手 |
| 学習・チューニング | ⏳ 未着手 |
| デプロイ | ⏳ 未着手 |

## ドキュメント

設計ドキュメントは `docs/` に集約：

- [📑 設計ドキュメント インデックス](docs/README.md)
- [01. 要件定義](docs/01_requirements.md)
- [02. 未決事項リスト](docs/02_open_questions.md)
- [03. アーキテクチャ](docs/03_architecture.md)
- [04. 技術スタック](docs/04_tech_stack.md)
- [05. UI 設計](docs/05_ui_design.md)
- [06. データモデル](docs/06_data_model.md)
- [07. CPU / 強化学習設計](docs/07_ai_design.md)
- [08. アセット計画](docs/08_assets.md)

## ライセンス

[MIT License](LICENSE)
