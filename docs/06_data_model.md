# 06. データモデル

最終更新: 2026-04-29

**最重要原則**: ゲームに関する数値（盤面サイズ・駒サイズ段階・駒数）は**一切ハードコードせず**、`GameRules` オブジェクトから駆動する。

## 1. 設計目標（再掲）

ユーザーの拡張要求:
- 盤面: 3x3（既定）、4x4 への拡張、それ以上もあり得る
- 駒サイズ段階: 大中小（既定3段階）、極大追加で4段階、さらに増設もあり得る
- 各サイズの駒数: 2個ずつ（既定）、可変

これらが「設定オブジェクトを変えるだけ」で動くようにする。

## 2. 主要型定義（TypeScript）

```ts
// ============================================
// ルール定義
// ============================================

/** 駒のサイズ段階。順序があり、大きいほど rank が大きい */
export interface PieceSize {
  /** 識別子（"S" | "M" | "L" | "XL" 等。任意の文字列） */
  id: string;
  /** 大きさの順序。0 が最小、増えるほど大きい */
  rank: number;
  /** 表示名（多言語対応） */
  displayName: { ja: string; en: string };
}

/** 勝利条件（拡張可能な ADT） */
export type WinCondition =
  | { kind: "lineOfN"; n: number }   // N目並べ系。盤面サイズに合わせる
  | { kind: "custom"; predicate: (board: Board) => Player | null };

/** ゲームルール。すべてここから駆動される */
export interface GameRules {
  /** 盤面の一辺のマス数。3, 4, ... */
  boardSize: number;

  /** 駒サイズ定義（rank 昇順）。少なくとも1要素 */
  pieceSizes: PieceSize[];

  /** プレイヤーごとの、各サイズの所持数（pieceSizes と同じ長さ） */
  piecesPerSize: number[];

  /** 勝利条件 */
  winCondition: WinCondition;

  /** 同サイズの駒で覆い被せ可能か（既定 false: より大きいサイズのみ） */
  allowSameSizeCover: boolean;

  /** 自分の駒の上に自分の駒を被せ可能か（既定 false） */
  allowSelfCover: boolean;
}

/** 既定ルール（3x3、大中小2個ずつ） */
export const DEFAULT_RULES: GameRules = {
  boardSize: 3,
  pieceSizes: [
    { id: "S", rank: 0, displayName: { ja: "小", en: "Small" } },
    { id: "M", rank: 1, displayName: { ja: "中", en: "Medium" } },
    { id: "L", rank: 2, displayName: { ja: "大", en: "Large" } },
  ],
  piecesPerSize: [2, 2, 2],
  winCondition: { kind: "lineOfN", n: 3 },
  allowSameSizeCover: false,
  allowSelfCover: false,
};

/** 4x4 拡張プリセット例 */
export const RULES_4X4_XL: GameRules = {
  boardSize: 4,
  pieceSizes: [
    { id: "S", rank: 0, displayName: { ja: "小", en: "Small" } },
    { id: "M", rank: 1, displayName: { ja: "中", en: "Medium" } },
    { id: "L", rank: 2, displayName: { ja: "大", en: "Large" } },
    { id: "XL", rank: 3, displayName: { ja: "極大", en: "X-Large" } },
  ],
  piecesPerSize: [2, 2, 2, 2],
  winCondition: { kind: "lineOfN", n: 4 },
  allowSameSizeCover: false,
  allowSelfCover: false,
};

// ============================================
// プレイヤー・駒
// ============================================

export type Player = "P1" | "P2";

export interface Piece {
  owner: Player;
  /** PieceSize.id を参照 */
  sizeId: string;
}

// ============================================
// 盤面
// ============================================

/** 1マスは駒のスタック。トップが見えている駒 */
export type Cell = Piece[];

/** 盤面: row-major の2次元配列。長さは GameRules.boardSize */
export type Board = Cell[][];

// ============================================
// 残り駒
// ============================================

/** プレイヤーの手駒（盤外スタック）。サイズID → 残り個数 */
export type Reserve = Record<string, number>;

// ============================================
// 手 (Move)
// ============================================

export type Position = { row: number; col: number };

/** 手駒から盤上に置く */
export interface PlaceFromReserveMove {
  kind: "placeFromReserve";
  player: Player;
  sizeId: string;
  to: Position;
}

/** 盤上の駒を移動（覆い被せを含む） */
export interface MoveOnBoardMove {
  kind: "moveOnBoard";
  player: Player;
  from: Position;
  to: Position;
}

export type Move = PlaceFromReserveMove | MoveOnBoardMove;

// ============================================
// ゲーム状態
// ============================================

export interface GameState {
  rules: GameRules;
  board: Board;
  reserves: { P1: Reserve; P2: Reserve };
  toMove: Player;
  /** これまでの手の履歴。待った・リプレイで使用 */
  history: Move[];
  /** 結果。null は対局中、引き分けは "draw" */
  outcome: Player | "draw" | null;
}

// ============================================
// エンジン関数（純関数）
// ============================================

export interface GameEngine {
  initialState(rules: GameRules): GameState;
  legalMoves(state: GameState): Move[];
  applyMove(state: GameState, move: Move): GameState;
  undo(state: GameState): GameState; // history が空なら no-op
  isTerminal(state: GameState): boolean;
  outcome(state: GameState): Player | "draw" | null;
  validateRules(rules: GameRules): { ok: boolean; errors: string[] };
}
```

## 3. 拡張性ポイント

### 3.1 盤面サイズ
- `Board` は2次元配列。`boardSize` 駆動。
- 全ての操作（座標生成・勝敗判定・UI 描画）は `state.rules.boardSize` を参照する。
- ハードコード禁止: `for (let i = 0; i < 3; i++)` は ❌、`for (let i = 0; i < state.rules.boardSize; i++)` ✅。

### 3.2 駒サイズ段階
- `pieceSizes` は配列。`rank` で順序付け。
- 覆い被せ判定は `rank` 比較のみで済む（"L" などのID依存禁止）。
- 新しいサイズ "極大" を足す = `pieceSizes` に1要素追加するだけ。

### 3.3 駒数
- `piecesPerSize[i]` で駒サイズ i の個数を変更可能。
- バリデーション必要: 総駒数が盤面に対して過多/過少でないか。

### 3.4 勝利条件
- `WinCondition` を ADT にして将来の拡張（斜め禁止・対角線必須・カウント方式等）を許す。
- 既定は `lineOfN` で N = boardSize に合わせる。

## 4. AlphaZero との接続

### 4.1 入力テンソル化（ユーザー単独決定だが、設計の枠組みのみ提示）
- 入力次元は `boardSize × boardSize × C` のような形を想定
- C = チャンネル数。サイズ段階数 × プレイヤー数 + 補助プレーン
- ルールが拡張されるとモデル再学習が必要（重みの再生成）
- → **ルール毎に学習済みモデルを用意する必要があるか**を判断する必要あり。当面は既定ルール (3x3 大中小2個ずつ) でのみCPU対戦をサポートし、4x4 等カスタムは「ローカル2人対戦専用 + 簡易CPU（ランダム/ヒューリスティック）」とするのが現実的。

### 4.2 行動空間
- 全アクション = (手駒からの配置) + (盤上の移動)
- 配置: `pieceSizes.length × boardSize²`
- 移動: `boardSize² × boardSize²`（出発マス × 到着マス）
- 合計次元 = `pieceSizes.length × boardSize² + boardSize⁴`
- 既定ルールで: 3 × 9 + 81 = 108 次元
- 4x4 + 4サイズで: 4 × 16 + 256 = 320 次元

## 5. localStorage スキーマ

```ts
interface PersistedSettings {
  language: "ja" | "en";
  theme: "light" | "dark" | "system";
  bgmVolume: number;   // 0..1
  sfxVolume: number;   // 0..1
  bgmEnabled: boolean;
  sfxEnabled: boolean;
}

interface PersistedStats {
  perDifficulty: Record<number, { wins: number; losses: number; draws: number }>;
  lastPlayedAt: string | null; // ISO 8601
  totalGames: number;
}

interface PersistedSession {
  // ローカル2人対戦の前回入力を覚えておく
  lastP1Name?: string;
  lastP2Name?: string;
  lastP1AvatarId?: string;
  lastP2AvatarId?: string;
}
```

すべて単一の名前空間プレフィックス（例: `dttt:`）で始める。バージョン管理キー `dttt:schemaVersion` を持ち、互換性のないスキーマ変更時はマイグレーションする。

## 6. 未決事項

ユーザー単独決定:
- 勝利条件の正式定義（lineOfN で N = boardSize で良いか）
- 覆い被せ可否（同サイズ・自分の駒）の最終仕様
- 入力テンソル化の具体（チャンネル設計）
- カスタムルール時の CPU 対応方針（同モデルで動かす？簡易AIで代替？）

こちら相談:
- AlphaZero の行動空間の実装詳細（ベクトル化・マスキング）
- リプレイ（履歴）データの永続化要否
