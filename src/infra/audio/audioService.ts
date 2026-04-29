/**
 * Audio service powered by Howler.js.
 *
 * Real audio files come from `public/audio/` later (see `docs/08_assets.md`).
 * This service is URL-agnostic: callers pass `{ bgm, sfx }` maps of
 * `name -> URL` to `init()`, and Howl instances are constructed lazily on
 * first `playBgm`/`playSfx` (or eagerly via `preload`).
 *
 * Howler abstracts away the various browser autoplay-gesture rules and
 * Web Audio differences (notably iOS Safari).
 */

import { Howl } from 'howler';

interface AudioConfig {
  bgm: Record<string, string>;
  sfx: Record<string, string>;
}

interface InternalState {
  config: AudioConfig;
  bgmInstances: Map<string, Howl>;
  sfxInstances: Map<string, Howl>;
  bgmVolume: number;
  sfxVolume: number;
  bgmEnabled: boolean;
  sfxEnabled: boolean;
  currentBgm: { name: string; soundId: number } | null;
}

const state: InternalState = {
  config: { bgm: {}, sfx: {} },
  bgmInstances: new Map(),
  sfxInstances: new Map(),
  bgmVolume: 1,
  sfxVolume: 1,
  bgmEnabled: true,
  sfxEnabled: true,
  currentBgm: null,
};

const clamp01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);

/** Configure the service with name -> URL maps. Replaces any previous config. */
export function init(config: AudioConfig): void {
  // Stop any current BGM tied to the previous config.
  stopBgm();
  state.config = { bgm: { ...config.bgm }, sfx: { ...config.sfx } };
  state.bgmInstances.clear();
  state.sfxInstances.clear();
}

export function setBgmVolume(v: number): void {
  state.bgmVolume = clamp01(v);
  for (const howl of state.bgmInstances.values()) {
    howl.volume(state.bgmEnabled ? state.bgmVolume : 0);
  }
}

export function setSfxVolume(v: number): void {
  state.sfxVolume = clamp01(v);
  for (const howl of state.sfxInstances.values()) {
    howl.volume(state.sfxEnabled ? state.sfxVolume : 0);
  }
}

export function setBgmEnabled(b: boolean): void {
  state.bgmEnabled = b;
  for (const howl of state.bgmInstances.values()) {
    howl.volume(b ? state.bgmVolume : 0);
  }
  if (!b) stopBgm();
}

export function setSfxEnabled(b: boolean): void {
  state.sfxEnabled = b;
  for (const howl of state.sfxInstances.values()) {
    howl.volume(b ? state.sfxVolume : 0);
  }
}

function getOrCreateBgm(name: string): Howl | null {
  const existing = state.bgmInstances.get(name);
  if (existing) return existing;

  const url = state.config.bgm[name];
  if (!url) return null;

  const howl = new Howl({
    src: [url],
    loop: true,
    volume: state.bgmEnabled ? state.bgmVolume : 0,
    preload: true,
    html5: true, // streaming-friendly for longer BGM tracks.
  });
  state.bgmInstances.set(name, howl);
  return howl;
}

function getOrCreateSfx(name: string): Howl | null {
  const existing = state.sfxInstances.get(name);
  if (existing) return existing;

  const url = state.config.sfx[name];
  if (!url) return null;

  const howl = new Howl({
    src: [url],
    loop: false,
    volume: state.sfxEnabled ? state.sfxVolume : 0,
    preload: true,
  });
  state.sfxInstances.set(name, howl);
  return howl;
}

export function playBgm(name: string): void {
  if (!state.bgmEnabled) return;

  // If the requested track is already playing, do nothing.
  if (state.currentBgm?.name === name) return;

  stopBgm();
  const howl = getOrCreateBgm(name);
  if (!howl) return;
  const soundId = howl.play();
  state.currentBgm = { name, soundId };
}

export function stopBgm(): void {
  const current = state.currentBgm;
  if (!current) return;
  const howl = state.bgmInstances.get(current.name);
  try {
    howl?.stop(current.soundId);
  } catch {
    // ignore: Howler can throw if the sound was unloaded.
  }
  state.currentBgm = null;
}

export function playSfx(name: string): void {
  if (!state.sfxEnabled) return;
  const howl = getOrCreateSfx(name);
  if (!howl) return;
  howl.play();
}

/**
 * Eagerly instantiate Howl objects for the given names so the audio is
 * downloaded/decoded before the user triggers a play. Resolves once every
 * named clip has either loaded or errored (errors are swallowed — they will
 * resurface on actual play attempts).
 */
export function preload(names: string[]): Promise<void> {
  const tasks: Promise<void>[] = [];

  for (const name of names) {
    const howl = getOrCreateBgm(name) ?? getOrCreateSfx(name);
    if (!howl) continue;

    if (howl.state() === 'loaded') continue;

    tasks.push(
      new Promise<void>((resolve) => {
        const done = (): void => resolve();
        howl.once('load', done);
        howl.once('loaderror', done);
      }),
    );
  }

  return Promise.all(tasks).then(() => undefined);
}

/** Test-only: reset module state. Not part of the public surface. */
export function __resetForTests(): void {
  stopBgm();
  state.config = { bgm: {}, sfx: {} };
  state.bgmInstances.clear();
  state.sfxInstances.clear();
  state.bgmVolume = 1;
  state.sfxVolume = 1;
  state.bgmEnabled = true;
  state.sfxEnabled = true;
  state.currentBgm = null;
}

export type { AudioConfig };

export const audioService = {
  init,
  setBgmVolume,
  setSfxVolume,
  setBgmEnabled,
  setSfxEnabled,
  playBgm,
  stopBgm,
  playSfx,
  preload,
};
