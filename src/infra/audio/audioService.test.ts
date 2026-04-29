import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// We mock 'howler' so unit tests do not actually decode audio.
// Each `Howl` instance records its constructor options and exposes
// spies for the methods the service calls.

interface FakeHowlOptions {
  src: string[];
  loop?: boolean;
  volume?: number;
  preload?: boolean;
  html5?: boolean;
}

interface FakeHowl {
  options: FakeHowlOptions;
  play: ReturnType<typeof vi.fn>;
  stop: ReturnType<typeof vi.fn>;
  volume: ReturnType<typeof vi.fn>;
  state: ReturnType<typeof vi.fn>;
  once: ReturnType<typeof vi.fn>;
  on: ReturnType<typeof vi.fn>;
}

const created: FakeHowl[] = [];

vi.mock('howler', () => {
  class Howl {
    public options: FakeHowlOptions;
    public play = vi.fn(() => 1);
    public stop = vi.fn();
    public volume = vi.fn();
    public state = vi.fn(() => 'loaded');
    public once = vi.fn();
    public on = vi.fn();

    constructor(options: FakeHowlOptions) {
      this.options = options;
      created.push(this);
    }
  }
  return { Howl };
});

import {
  __resetForTests,
  init,
  playBgm,
  playSfx,
  preload,
  setBgmEnabled,
  setBgmVolume,
  setSfxEnabled,
  setSfxVolume,
  stopBgm,
} from './audioService';

describe('audio/audioService', () => {
  beforeEach(() => {
    __resetForTests();
    created.length = 0;
  });

  afterEach(() => {
    __resetForTests();
    created.length = 0;
  });

  it('init() accepts a paths config without throwing', () => {
    expect(() =>
      init({
        bgm: { main: '/audio/bgm/main.mp3' },
        sfx: { click: '/audio/sfx/click.mp3' },
      }),
    ).not.toThrow();

    // No Howl instances should be constructed at init time (lazy creation).
    expect(created.length).toBe(0);
  });

  it('setBgmVolume / setSfxVolume run without throwing', () => {
    expect(() => setBgmVolume(0.5)).not.toThrow();
    expect(() => setSfxVolume(0.3)).not.toThrow();
  });

  it('clamps volume into [0, 1] and propagates to active Howl instances', () => {
    init({ bgm: { theme: 'theme.mp3' }, sfx: { tap: 'tap.mp3' } });

    playBgm('theme');
    playSfx('tap');
    expect(created.length).toBe(2);

    setBgmVolume(2);
    setSfxVolume(-1);

    const bgm = created[0];
    const sfx = created[1];
    if (!bgm || !sfx) throw new Error('expected two Howl instances');

    // last call: clamped values, modulated by enabled flag (both true -> raw value)
    expect(bgm.volume).toHaveBeenLastCalledWith(1);
    expect(sfx.volume).toHaveBeenLastCalledWith(0);
  });

  it('lazily constructs Howl instances on first playBgm/playSfx', () => {
    init({ bgm: { menu: 'menu.mp3' }, sfx: { click: 'click.mp3' } });
    expect(created.length).toBe(0);

    playBgm('menu');
    expect(created.length).toBe(1);
    expect(created[0]?.options.src).toEqual(['menu.mp3']);
    expect(created[0]?.options.loop).toBe(true);

    playSfx('click');
    expect(created.length).toBe(2);
    expect(created[1]?.options.src).toEqual(['click.mp3']);
    expect(created[1]?.options.loop).toBe(false);
  });

  it('does not double-construct when the same clip is requested twice', () => {
    init({ sfx: { click: 'click.mp3' }, bgm: {} });
    playSfx('click');
    playSfx('click');
    expect(created.length).toBe(1);
    expect(created[0]?.play).toHaveBeenCalledTimes(2);
  });

  it('setBgmEnabled(false) stops the current track', () => {
    init({ bgm: { theme: 'theme.mp3' }, sfx: {} });
    playBgm('theme');
    const bgm = created[0];
    if (!bgm) throw new Error('expected a Howl instance');

    setBgmEnabled(false);
    expect(bgm.stop).toHaveBeenCalled();

    // Subsequent plays are no-ops while disabled.
    playBgm('theme');
    expect(bgm.play).toHaveBeenCalledTimes(1);
  });

  it('setSfxEnabled(false) silences but does not throw on play', () => {
    init({ bgm: {}, sfx: { click: 'click.mp3' } });
    setSfxEnabled(false);
    expect(() => playSfx('click')).not.toThrow();
    // Disabled => no Howl is created, no play.
    expect(created.length).toBe(0);
  });

  it('playBgm() ignores unknown names', () => {
    init({ bgm: {}, sfx: {} });
    expect(() => playBgm('nope')).not.toThrow();
    expect(created.length).toBe(0);
  });

  it('stopBgm() with nothing playing is a no-op', () => {
    init({ bgm: {}, sfx: {} });
    expect(() => stopBgm()).not.toThrow();
  });

  it('preload([]) resolves immediately', async () => {
    init({ bgm: {}, sfx: {} });
    await expect(preload([])).resolves.toBeUndefined();
  });

  it('preload(names) constructs Howls for known names and resolves once loaded', async () => {
    init({ bgm: { theme: 'theme.mp3' }, sfx: { click: 'click.mp3' } });
    // Default mock state() returns 'loaded' so no waits are queued.
    await expect(preload(['theme', 'click', 'unknown'])).resolves.toBeUndefined();
    expect(created.length).toBe(2);
  });
});
