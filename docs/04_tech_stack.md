# 04. 技術スタック選定

最終更新: 2026-04-29

ユーザー指定: フロントエンドは React。それ以外は本ドキュメントで決定する。

## 1. フロントエンド

| カテゴリ | 採用 | バージョン目安 | 理由 |
|---|---|---|---|
| 言語 | TypeScript | 5.x | 型安全。AlphaZero周りの複雑な型を扱いきるため必須。 |
| UI フレームワーク | React | 18+ | ユーザー指定。 |
| ビルドツール | Vite | 5.x | 高速。GitHub Pages デプロイの設定が最小。Web Worker サポートが綺麗。 |
| ルーティング | react-router-dom (HashRouter) | 6.x | GitHub Pages とコンフリクトしない。 |
| 状態管理 | Zustand | 4.x | 軽量・boilerplate 最小・TS 親和性高。 |
| スタイリング | Tailwind CSS | 3.x | ユーティリティで高速開発、テーマ切替が CSS 変数と相性良い。 |
| アニメーション | Framer Motion | 11.x | 宣言的・React 製・物理アニメーションが綺麗。 |
| i18n | react-i18next | 14.x | デファクト標準。JA/EN の2言語なら十分。 |
| 音声 | Howler.js | 2.x | iOS Safari 含むブラウザ差異を吸収。BGM/SFX 両対応。 |
| ML 推論 | onnxruntime-web | 1.x | PyTorch→ONNX→ブラウザの最短経路。WASM/WebGPU 切替可。 |
| アイコン | lucide-react | 最新 | 軽量・モダン・OSS。 |

## 2. テスト・品質

| カテゴリ | 採用 | 理由 |
|---|---|---|
| ユニット | Vitest | Vite ネイティブ。Jest 互換 API。 |
| コンポーネント | React Testing Library | デファクト。 |
| E2E | Playwright | スマホブラウザ含む実機相当の検証が可能。 |
| Lint | ESLint + typescript-eslint | 標準。 |
| Format | Prettier | 議論不要にするため。 |
| Git hook | Husky + lint-staged | コミット時に lint/format を強制。 |
| 型チェック | tsc --noEmit | CI で別ステップ。 |

## 3. パッケージマネージャ

- **pnpm** を採用
- 理由: 速度・ディスク効率・厳密な依存解決。

## 4. ML 学習側（Python）

| カテゴリ | 採用 | 理由 |
|---|---|---|
| 言語 | Python 3.11+ | PyTorch の対応・型ヒント完成度。 |
| DL フレームワーク | PyTorch | ONNX エクスポートが標準。 |
| ONNX 変換 | torch.onnx | 公式経路。 |
| パッケージ管理 | uv または rye | pip より速く再現性が高い。 |
| 数値計算 | NumPy | 必須。 |
| 並列自己対戦 | multiprocessing + torch.cuda | RTX 2080 を使い切るため。 |
| ロギング | TensorBoard or Weights & Biases (free tier) | 学習可視化、ポートフォリオ価値も高い。 |

## 5. CI / CD

| 項目 | 採用 |
|---|---|
| CI | GitHub Actions |
| デプロイ先 | GitHub Pages (`gh-pages` ブランチ) |
| 自動デプロイトリガ | `main` への push |
| PR チェック | lint / typecheck / unit test / build |

## 6. 採用しないもの（参考）

- **Next.js**: GitHub Pages で SSR が動かない。SSG だけなら Vite で十分。
- **Redux Toolkit**: Zustand で十分。
- **Emotion / styled-components**: Tailwind に集約。
- **Jest**: Vitest 採用のため不要。
- **TensorFlow.js**: PyTorch→ONNX の経路を ONNX Runtime Web で扱うため不要。
- **PWA / オフライン対応**: スコープ外。後の拡張余地として残す。

## 7. バージョンの方針

- メジャーバージョンは固定、マイナー以下は `^` で許容
- Renovate / Dependabot を後で導入検討（実装フェーズで）

## 8. 参考: ライセンス確認

すべて MIT/Apache/BSD 系の OSS で構成可能。商用利用・配布に制約なし。
モデル重みは自家学習なのでライセンス問題なし。
アバター・BGM・SFX は別途確認（`08_assets.md`）。
