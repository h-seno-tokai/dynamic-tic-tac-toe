# Licenses

Per-asset and per-dependency license disclosures for Dynamic Tic-Tac-Toe. Code is MIT (see [LICENSE](./LICENSE)). The third-party assets bundled or fetched at runtime are listed below with their respective licenses and attribution requirements.

## Code

| Component                                       | License                                            |
| ----------------------------------------------- | -------------------------------------------------- |
| Application source (`src/`, `ai-training/src/`) | MIT — see [LICENSE](./LICENSE)                     |
| Trained model weights (`public/model.onnx`)     | MIT (own work; trained from scratch via self-play) |

## Audio

### BGM

| Track                                  | Author   | Source                                | License                                                                                          |
| -------------------------------------- | -------- | ------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 「Whip」 (`public/audio/bgm/whip.mp3`) | しゃろう | <https://dova-s.jp/bgm/play1208.html> | DOVA-SYNDROME standard license — commercial use OK, attribution optional, no registration needed |

### SFX (`public/audio/sfx/*.mp3`)

| File                                                                                                                  | Source                                     | License                                                                                                                                                                  |
| --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `pickup.mp3`, `place.mp3`, `undo.mp3`, `start.mp3`, `fanfare.mp3`, `win.mp3`, `lose.mp3`, `button.mp3`, `invalid.mp3` | 効果音ラボ <https://soundeffect-lab.info/> | 効果音ラボ規約 — commercial use OK, attribution optional, redistribution of the source files alone is prohibited but bundling them inside a game is explicitly permitted |

## Avatars

| Component          | Source                                                                             | License                                                                                                   |
| ------------------ | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| User & CPU avatars | DiceBear "Avataaars" by Pablo Stanley <https://www.dicebear.com/styles/avataaars/> | MIT (DiceBear) + Free for personal and commercial use (Avataaars by Pablo Stanley) — attribution optional |

## Runtime dependencies

JavaScript / TypeScript runtime dependencies (React, Vite, ONNX Runtime Web, Tailwind CSS, Framer Motion, Howler.js, react-i18next, Zustand, etc.) are MIT or Apache-2.0 licensed; see `package.json` and the `node_modules/` tree for the complete list.

PyTorch and the ONNX toolchain (used only at training time, not bundled with the deployed app) are BSD/MIT licensed.

## Notes

- No assets in this repository are redistributed in violation of upstream license terms to the best of the author's knowledge.
- If you spot an attribution that should be added or corrected, please open an issue.
