import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

export interface SessionState {
  lastP1Name: string | undefined;
  lastP2Name: string | undefined;
  lastP1AvatarId: string | undefined;
  lastP2AvatarId: string | undefined;

  setLastP1Name: (name: string | undefined) => void;
  setLastP2Name: (name: string | undefined) => void;
  setLastP1AvatarId: (avatarId: string | undefined) => void;
  setLastP2AvatarId: (avatarId: string | undefined) => void;
  reset: () => void;
}

export const SESSION_STORAGE_KEY = 'dttt:session';
export const SESSION_SCHEMA_VERSION = 1;

export const useSessionStore = create<SessionState>()(
  persist(
    (set) => ({
      lastP1Name: undefined,
      lastP2Name: undefined,
      lastP1AvatarId: undefined,
      lastP2AvatarId: undefined,

      setLastP1Name: (name) => {
        set({ lastP1Name: name });
      },
      setLastP2Name: (name) => {
        set({ lastP2Name: name });
      },
      setLastP1AvatarId: (avatarId) => {
        set({ lastP1AvatarId: avatarId });
      },
      setLastP2AvatarId: (avatarId) => {
        set({ lastP2AvatarId: avatarId });
      },
      reset: () => {
        set({
          lastP1Name: undefined,
          lastP2Name: undefined,
          lastP1AvatarId: undefined,
          lastP2AvatarId: undefined,
        });
      },
    }),
    {
      name: SESSION_STORAGE_KEY,
      version: SESSION_SCHEMA_VERSION,
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        lastP1Name: state.lastP1Name,
        lastP2Name: state.lastP2Name,
        lastP1AvatarId: state.lastP1AvatarId,
        lastP2AvatarId: state.lastP2AvatarId,
      }),
    },
  ),
);
