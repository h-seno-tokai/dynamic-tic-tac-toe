# 09. コーディング規約・コンポーネント設計指針

最終更新: 2026-04-29

実装フェーズ前の指針として、コードベースの一貫性とポートフォリオとしての評価を高めるための規約を定める。

## 1. 設計原則

### 1.1 汎用性最大化（最重要）
- **値のハードコード禁止**: 盤面サイズ・駒サイズ・色・テキスト・座標等
- props / 引数 / 設定ファイル / `GameRules` で外部から制御
- 同じ振る舞いをする処理を複数箇所に書かない（DRY）
- 例: 「3x3 を前提にしたループ」は禁止 → `rules.boardSize` 駆動

### 1.2 関心の分離
- **ロジック層**（`domain/`）: React/DOM 非依存・純粋関数
- **状態層**（`stores/`）: 薄く、ロジックを呼び出すだけ
- **表示層**（`components/`）: 表示と入力に集中
- **副作用**（`infra/`）: fetch・audio・storage・i18n に集約

### 1.3 構成可能性
- **Headless パターン**: ロジックは Hook、見た目は別コンポーネント
- **Compound Components**: `Modal.Header` / `Modal.Body` 等
- 子要素の差し替えで多用途化

### 1.4 早期最適化禁止
- `React.memo` / `useMemo` / `useCallback` は計測してから入れる
- 重い計算は Web Worker（既に AI で適用）

## 2. ディレクトリ構成

```
src/
├── domain/                  # ピュアロジック（React/DOM 非依存）
│   ├── rules/               # GameRules 型・プリセット
│   ├── engine/              # ルールエンジン（applyMove / legalMoves 等）
│   └── types/
│
├── ai/                      # 推論レイヤ
│   ├── worker/              # Web Worker entry
│   ├── mcts/
│   ├── inference/           # ONNX Runtime Web ラッパ
│   └── difficulty/          # 難易度プロファイル
│
├── stores/                  # Zustand stores
│   ├── gameStore.ts
│   ├── settingsStore.ts
│   ├── statsStore.ts
│   └── sessionStore.ts
│
├── components/
│   ├── primitives/          # 完全に汎用な原子レベル
│   │   ├── Button.tsx
│   │   ├── Toggle.tsx
│   │   ├── Slider.tsx
│   │   ├── RadioGroup.tsx
│   │   ├── Modal.tsx
│   │   ├── Tooltip.tsx
│   │   └── ...
│   │
│   ├── layout/              # ページ枠組み
│   │   ├── PageShell.tsx
│   │   ├── NavBar.tsx
│   │   └── ...
│   │
│   ├── game/                # ゲーム特化（だがルール非依存）
│   │   ├── Board.tsx        # 任意の boardSize に対応
│   │   ├── Cell.tsx         # 任意の Piece[] を表示
│   │   ├── Piece.tsx        # 任意の (size, owner) を描画
│   │   ├── ReserveStack.tsx # 任意の駒構成を表示
│   │   ├── MoveHistory.tsx
│   │   └── ThinkingIndicator.tsx
│   │
│   ├── avatar/              # アバター関連
│   │   ├── UserAvatar.tsx   # DiceBear Avataaars wrapper
│   │   ├── CpuAvatar.tsx    # Bot icon + 難易度バッジ
│   │   └── AvatarPicker.tsx
│   │
│   └── feedback/            # 通知・フィードバック
│       ├── Toast.tsx
│       ├── LoadingSpinner.tsx
│       └── ErrorBoundary.tsx
│
├── pages/                   # 画面（ルートレベル）
│   ├── TitlePage.tsx
│   ├── GamePage.tsx
│   └── ...
│
├── hooks/                   # カスタム Hook
│   ├── useGameSession.ts
│   ├── useTheme.ts
│   ├── useAudio.ts
│   └── ...
│
├── infra/                   # 副作用ラッパ
│   ├── i18n/
│   ├── audio/               # Howler.js wrapper
│   ├── storage/             # localStorage 抽象
│   └── theme/
│
├── utils/                   # 汎用ユーティリティ
├── App.tsx
└── main.tsx
```

## 3. 命名規則

| 種別 | 規約 | 例 |
|---|---|---|
| コンポーネントファイル | `PascalCase.tsx` | `Board.tsx` |
| Hook ファイル | `useCamelCase.ts` | `useTheme.ts` |
| utility ファイル | `camelCase.ts` | `formatTime.ts` |
| ストアファイル | `camelCaseStore.ts` | `gameStore.ts` |
| 型ファイル | `camelCase.ts` | `gameTypes.ts` |
| props 型名 | `ComponentNameProps` | `BoardProps` |
| Hook 名 | `use*` | `useGameSession` |
| グローバル定数 | `UPPER_SNAKE_CASE` | `MAX_BOARD` |
| イベントハンドラ | `handleXxx` | `handleCellClick` |
| Boolean prop | `is*` / `has*` / `can*` | `isLoading`, `hasFocus` |

## 4. コンポーネント設計の具体ルール

### 4.1 props 経由の汎用化（例）
```ts
// ❌ NG: 内部で 3x3 を前提
const Board = () => {
  return [0,1,2].map(row => ...);
};

// ✅ OK: rules を受け取って汎用化
type BoardProps = {
  state: GameState;
  onCellClick: (pos: Position) => void;
  highlight?: Position[];
};
const Board: FC<BoardProps> = ({ state, onCellClick, highlight }) => {
  const N = state.rules.boardSize;
  return Array.from({ length: N }, (_, row) =>
    Array.from({ length: N }, (_, col) => (
      <Cell
        key={`${row}-${col}`}
        pieces={state.board[row][col]}
        onClick={() => onCellClick({ row, col })}
      />
    ))
  );
};
```

### 4.2 状態と表示の分離
- 表示コンポーネントは store に直接アクセスしない
- ページレベルが store から値を取り、props 経由で渡す
- 例外: 設定・テーマのような普遍的 store は直接 hook で参照可

### 4.3 アクセシビリティの最低ライン
- すべての interactive element に `aria-label` または可視ラベル
- `tabIndex` 順序を意識
- Modal は `role="dialog"` + フォーカストラップ
- 盤面は矢印キーでフォーカス移動 + Enter/Space で選択

## 5. TypeScript 設定

### 5.1 tsconfig.json（厳格モード）
```json
{
  "compilerOptions": {
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "noImplicitOverride": true,
    "exactOptionalPropertyTypes": true,
    "noFallthroughCasesInSwitch": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "paths": {
      "@/*": ["./src/*"]
    }
  }
}
```

### 5.2 型ファースト
- props 型を先に書いてから実装
- `any` 禁止 → `unknown` で受けて narrow
- `as` 型アサーションは最後の手段
- enum より union literal type を優先

## 6. ESLint / Prettier

### 6.1 ESLint extends
- `@typescript-eslint/recommended-type-checked`
- `@typescript-eslint/stylistic-type-checked`
- `react/recommended`
- `react-hooks/recommended`
- `jsx-a11y/recommended`
- `prettier`（最後、競合無効化）

### 6.2 Prettier
- default 設定
- `prettier-plugin-tailwindcss` でクラス自動ソート
- `printWidth: 100`

### 6.3 自動化
- Husky + lint-staged で pre-commit 時に lint + format
- ESLint 違反 / フォーマット崩れはコミット不可

## 7. テスト方針

### 7.1 単体テスト（Vitest + React Testing Library）
- `domain/` : カバレッジ 90%+ 目標（ルールエンジンの正当性が肝）
- `stores/` : 主要シナリオ
- `components/primitives/` : interaction テスト
- `ai/mcts/` : 探索ロジックの正当性

### 7.2 E2E テスト（Playwright）
- ゴールデンパス: タイトル → CPU対戦設定 → 対局 → 結果
- ローカル2人対戦の主要フロー
- レスポンシブ確認（mobile viewport）

### 7.3 アクセシビリティテスト
- `vitest-axe` で primitives の自動チェック
- 主要画面は手動でスクリーンリーダー確認

## 8. Git / コミット規約

### 8.1 Conventional Commits 厳守
- `feat:` 新機能
- `fix:` バグ修正
- `refactor:` リファクタ
- `chore:` 雑務（設定・ツール変更等）
- `docs:` ドキュメント
- `test:` テスト
- `style:` フォーマット
- `perf:` パフォーマンス改善
- `ci:` CI 設定

### 8.2 ブランチ戦略
- `main`: always-deployable
- 機能開発: `feat/short-description`
- バグ修正: `fix/short-description`
- マージは原則 squash merge

## 9. テーマ・カラーパレット（実装フェーズで具体化）

CSS 変数で管理し、`data-theme="light|dark"` で切替。
具体カラーコードは実装時にタイポグラフィと合わせて決定。

```css
:root[data-theme="light"] {
  --color-bg: ...;
  --color-fg: ...;
  --color-accent: ...;
  --color-board-cell: ...;
  --color-piece-p1: ...;
  --color-piece-p2: ...;
  /* ... */
}
```

Tailwind は CSS 変数経由で参照（`bg-[var(--color-bg)]` 等）。

## 10. パフォーマンス目標

- 初期ロード合計: 10MB 以内（モデル重み込み、`01_requirements.md` 4.1 と整合）
- TTI（Time to Interactive）: 3秒以内（4G 環境）
- フレームレート: 60fps（盤面操作中）
- AI 推論: モバイル含めて 1 forward pass ≤ 5ms（達成困難な場合は MCTS シミュレーション数を動的調整）
