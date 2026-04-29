import { useCallback, useEffect, useRef } from 'react';
import type { KeyboardEvent, MouseEvent, ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { AnimatePresence, motion } from 'framer-motion';

export interface ModalRootProps {
  open: boolean;
  onOpenChange: (next: boolean) => void;
  /** id of the element labelling the dialog (e.g. the header heading). */
  labelledBy: string;
  closeOnOverlayClick?: boolean | undefined;
  /** Optional id of an element describing the dialog. */
  describedBy?: string | undefined;
  className?: string | undefined;
  children: ReactNode;
}

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'textarea:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',');

const getFocusable = (root: HTMLElement): HTMLElement[] =>
  Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
    (el) => !el.hasAttribute('data-focus-guard'),
  );

const Root = ({
  open,
  onOpenChange,
  labelledBy,
  closeOnOverlayClick = true,
  describedBy,
  className,
  children,
}: ModalRootProps) => {
  const dialogRef = useRef<HTMLDivElement>(null);
  const previouslyFocused = useRef<HTMLElement | null>(null);

  // Save/restore focus and lock body scroll while open.
  useEffect(() => {
    if (!open) return;
    previouslyFocused.current =
      typeof document !== 'undefined' ? (document.activeElement as HTMLElement | null) : null;

    const dialog = dialogRef.current;
    if (dialog) {
      const focusables = getFocusable(dialog);
      (focusables[0] ?? dialog).focus();
    }

    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';

    return () => {
      document.body.style.overflow = prevOverflow;
      const prev = previouslyFocused.current;
      if (prev && typeof prev.focus === 'function') {
        prev.focus();
      }
    };
  }, [open]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLDivElement>) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onOpenChange(false);
        return;
      }
      if (e.key === 'Tab') {
        const dialog = dialogRef.current;
        if (!dialog) return;
        const focusables = getFocusable(dialog);
        if (focusables.length === 0) {
          e.preventDefault();
          dialog.focus();
          return;
        }
        const first = focusables[0]!;
        const last = focusables[focusables.length - 1]!;
        const active = document.activeElement as HTMLElement | null;
        if (e.shiftKey) {
          if (active === first || !dialog.contains(active)) {
            e.preventDefault();
            last.focus();
          }
        } else {
          if (active === last) {
            e.preventDefault();
            first.focus();
          }
        }
      }
    },
    [onOpenChange],
  );

  const handleOverlayClick = (e: MouseEvent<HTMLDivElement>) => {
    if (!closeOnOverlayClick) return;
    if (e.target === e.currentTarget) onOpenChange(false);
  };

  if (typeof document === 'undefined') return null;

  return createPortal(
    <AnimatePresence>
      {open && (
        <motion.div
          key="modal-overlay"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          onMouseDown={handleOverlayClick}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
          data-testid="modal-overlay"
        >
          <motion.div
            key="modal-dialog"
            ref={dialogRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby={labelledBy}
            aria-describedby={describedBy}
            tabIndex={-1}
            onKeyDown={handleKeyDown}
            initial={{ opacity: 0, scale: 0.96, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.98, y: 4 }}
            transition={{ duration: 0.18 }}
            className={[
              'relative w-full max-w-lg rounded-lg border border-border bg-bg text-fg shadow-xl',
              'focus:outline-none',
              className,
            ]
              .filter(Boolean)
              .join(' ')}
          >
            {children}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>,
    document.body,
  );
};

export interface ModalSlotProps {
  children: ReactNode;
  className?: string;
}

const Header = ({ children, className }: ModalSlotProps) => (
  <div
    className={['border-b border-border px-5 py-3 text-lg font-semibold', className]
      .filter(Boolean)
      .join(' ')}
  >
    {children}
  </div>
);

const Body = ({ children, className }: ModalSlotProps) => (
  <div className={['px-5 py-4', className].filter(Boolean).join(' ')}>{children}</div>
);

const Footer = ({ children, className }: ModalSlotProps) => (
  <div
    className={['flex items-center justify-end gap-2 border-t border-border px-5 py-3', className]
      .filter(Boolean)
      .join(' ')}
  >
    {children}
  </div>
);

export const Modal = {
  Root,
  Header,
  Body,
  Footer,
};

export type ModalHeaderProps = ModalSlotProps;
export type ModalBodyProps = ModalSlotProps;
export type ModalFooterProps = ModalSlotProps;
