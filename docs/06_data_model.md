# 06. データモデル

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
  | { kind: 'lineOfN'; n: number } // N目並べ系。盤面サイズに合わせる
  | { kind: 'custom'; predicate: (board: Board) => Player | null };

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

  /**
   * 引き分け判定: 手数（プライ）がこの値を超えたら強制引き分け。
   * Gobblet は駒を動かし続けられるため「合法手なし」だけでは終局しない。
   * 自己対戦で無限ループを避けるため必須。
   */
  maxPly: number;

  /**
   * 同一局面が N 回出現したら引き分け（threefold repetition）。
   * 既定: 3
   */
  drawByRepetition: number;
}

/**
 * ユーザーが選択できるルールプリセットは 2 種に限定。
 * 完全カスタムルールは UI から提供しない（ただし型は GameRules で汎用に保つ）。
 */

/** プリセットA: 3x3、大中小 各2個（既定） */
export const PRESET_3X3: GameRules = {
  boardSize: 3,
  pieceSizes: [
    { id: 'S', rank: 0, displayName: { ja: '小', en: 'Small' } },
    { id: 'M', rank: 1, displayName: { ja: '中', en: 'Medium' } },
    { id: 'L', rank: 2, displayName: { ja: '大', en: 'Large' } },
  ],
  piecesPerSize: [2, 2, 2],
  winCondition: { kind: 'lineOfN', n: 3 },
  allowSameSizeCover: false,
  allowSelfCover: true, // 標準 Gobblet 準拠: 自分の駒の上に被せ可
  maxPly: 60,
  drawByRepetition: 3,
};

/** プリセットB: 4x4、巨大・大・中・小 各3個 */
export const PRESET_4X4_XL: GameRules = {
  boardSize: 4,
  pieceSizes: [
    { id: 'S', rank: 0, displayName: { ja: '小', en: 'Small' } },
    { id: 'M', rank: 1, displayName: { ja: '中', en: 'Medium' } },
    { id: 'L', rank: 2, displayName: { ja: '大', en: 'Large' } },
    { id: 'XL', rank: 3, displayName: { ja: '巨大', en: 'Huge' } },
  ],
  piecesPerSize: [3, 3, 3, 3],
  winCondition: { kind: 'lineOfN', n: 4 },
  allowSameSizeCover: false,
  allowSelfCover: true,
  maxPly: 120,
  drawByRepetition: 3,
};

/** ユーザーに提示するプリセット一覧（並び順固定） */
export const RULE_PRESETS = [
  { id: '3x3-classic', label: { ja: '3x3 クラシック', en: '3x3 Classic' }, rules: PRESET_3X3 },
  { id: '4x4-huge', label: { ja: '4x4 巨大入り', en: '4x4 Huge' }, rules: PRESET_4X4_XL },
] as const;

export type RulePresetId = (typeof RULE_PRESETS)[number]['id'];

/** AI ユニバーサルネットワークが対応する上限（学習時に固定） */
export const AI_LIMITS = {
  MAX_BOARD: 4,
  MAX_PIECE_SIZES: 4,
  MAX_PIECES_PER_SIZE: 3,
} as const;

/** ルールが AI で対戦可能か（範囲内か）を判定 */
export function isRuleSupportedByAI(rules: GameRules): boolean {
  return (
    rules.boardSize <= AI_LIMITS.MAX_BOARD &&
    rules.pieceSizes.length <= AI_LIMITS.MAX_PIECE_SIZES &&
    rules.piecesPerSize.every((n) => n <= AI_LIMITS.MAX_PIECES_PER_SIZE)
  );
}

// ============================================
// プレイヤー・駒
// ============================================

export type Player = 'P1' | 'P2';

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
  kind: 'placeFromReserve';
  player: Player;
  sizeId: string;
  to: Position;
}

/** 盤上の駒を移動（覆い被せを含む） */
export interface MoveOnBoardMove {
  kind: 'moveOnBoard';
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
  /** これまでの手の履歴。待った（undo）操作で使用。永続化はしない。 */
  history: Move[];
  /** 経過手数（プライ数）。`maxPly` 判定に使用。 */
  ply: number;
  /** 同一局面の出現回数を追跡するハッシュ→カウントの map（`drawByRepetition` 判定用）。 */
  repetition: Map<string, number>;
  /** 結果。null は対局中、引き分けは "draw" */
  outcome: Player | 'draw' | null;
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
  outcome(state: GameState): Player | 'draw' | null;
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

## 4. AI との接続

3x3 と 4x4 で別経路（詳細は `07_ai_design.md`）:

- **3x3**: ニューラルネット非使用。`Solver3x3`（alpha-beta + Zobrist TT）が `GameState` を直接読む
- **4x4**: AlphaZero 系（4x4 専用 6×96ch CNN）+ MCTS

### 4.1 4x4 入力テンソル化

- 形状 **(1, 27, 4, 4)**（旧ユニバーサル設計の MAX_BOARD=4 形状を流用）
- ch 25 (out-of-board) と ch 26 (unused-size) は **4x4 では常時 0、退化チャネル**
  - 互換性のため形状は維持。stem の追加重み ~1.7K（全パラメータの 0.07%）で誤差
- アーキテクチャは Fully Convolutional Network（CNN）。Transformer は不採用

### 4.2 4x4 行動空間（固定 320 次元）

- 内訳:
  - PlaceFromReserve: 4 × 16 = 64
  - MoveOnBoard: 16 × 16 = 256
- 合法でない行動は **−∞ マスク** を ONNX グラフ外（TS 側）で適用 → softmax / argmax
- 4x4 巨大入り: 320 / 320（全有効）

## 5. localStorage スキーマ

```ts
interface PersistedSettings {
  language: 'ja' | 'en';
  theme: 'light' | 'dark' | 'system';
  bgmVolume: number; // 0..1
  sfxVolume: number; // 0..1
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

## 6. 確定事項

- 勝利条件: lineOfN で N = boardSize（縦・横・斜め）
- 引き分け条件:
  - **合法手なし**（ほぼ起きないが定義としては必要）
  - **`maxPly` 到達**（3x3=60, 4x4=120）
  - **threefold repetition**（同一局面が `drawByRepetition`=3 回出現）
- 覆い被せ: `allowSameSizeCover: false` / `allowSelfCover: true`（標準 Gobblet 準拠）
- 盤外スタック: サイズ別カウントのみ
- ルール選択: **2 プリセット固定**（PRESET_3X3 / PRESET_4X4_XL）。完全カスタム UI は提供しない
- AI: 盤面サイズで分割（3x3 = `Solver3x3` alpha-beta、4x4 = AlphaZero 4x4 専用 CNN (1, 27, 4, 4) 入力 / ResNet 6×96ch）
- リプレイ永続化: **不要**。戦績のみ localStorage 保存
- 同一局面の検出: 局面ハッシュ（盤面 + 手駒残数 + 手番）を canonical 文字列化して使用
