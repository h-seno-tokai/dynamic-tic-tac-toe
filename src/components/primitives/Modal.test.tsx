import { render, screen, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeAll } from 'vitest';
import { useState } from 'react';
import { Modal } from './Modal';

// framer-motion uses ResizeObserver/IntersectionObserver in some paths; jsdom is fine without.
beforeAll(() => {
  // Suppress matchMedia call from animation system if it occurs.
  if (!window.matchMedia) {
    Object.defineProperty(window, 'matchMedia', {
      value: () => ({
        matches: false,
        media: '',
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        addListener: vi.fn(),
        removeListener: vi.fn(),
        onchange: null,
        dispatchEvent: () => false,
      }),
    });
  }
});

const Harness = ({
  initialOpen = true,
  closeOnOverlayClick,
  onChange,
}: {
  initialOpen?: boolean;
  closeOnOverlayClick?: boolean;
  onChange?: (open: boolean) => void;
}) => {
  const [open, setOpen] = useState(initialOpen);
  return (
    <>
      <button onClick={() => setOpen(true)}>Open</button>
      <Modal.Root
        open={open}
        onOpenChange={(v) => {
          setOpen(v);
          onChange?.(v);
        }}
        labelledBy="t"
        closeOnOverlayClick={closeOnOverlayClick}
      >
        <Modal.Header>
          <h2 id="t">Title</h2>
        </Modal.Header>
        <Modal.Body>
          <button>First</button>
          <button>Last</button>
        </Modal.Body>
        <Modal.Footer>
          <button onClick={() => setOpen(false)}>Close</button>
        </Modal.Footer>
      </Modal.Root>
    </>
  );
};

describe('Modal', () => {
  it('renders dialog when open with proper ARIA', () => {
    render(<Harness />);
    const dialog = screen.getByRole('dialog');
    expect(dialog).toHaveAttribute('aria-modal', 'true');
    expect(dialog).toHaveAttribute('aria-labelledby', 't');
  });

  it('does not render when closed', () => {
    render(<Harness initialOpen={false} />);
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('closes on Escape', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Harness onChange={onChange} />);
    await user.keyboard('{Escape}');
    expect(onChange).toHaveBeenCalledWith(false);
  });

  it('closes on overlay click by default', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Harness onChange={onChange} />);
    const overlay = screen.getByTestId('modal-overlay');
    // userEvent.click triggers mousedown on the target, which is what handler reads.
    await user.click(overlay);
    expect(onChange).toHaveBeenCalledWith(false);
  });

  it('does not close on overlay click when disabled', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Harness closeOnOverlayClick={false} onChange={onChange} />);
    await user.click(screen.getByTestId('modal-overlay'));
    expect(onChange).not.toHaveBeenCalled();
  });

  it('traps focus with Tab', async () => {
    const user = userEvent.setup();
    render(<Harness />);
    const first = screen.getByRole('button', { name: 'First' });
    const close = screen.getByRole('button', { name: 'Close' });
    // First focusable should be auto-focused.
    expect(document.activeElement).toBe(first);
    // Tab through to last focusable.
    await user.tab();
    await user.tab();
    expect(document.activeElement).toBe(close);
    // Tabbing past last wraps to first.
    await user.tab();
    expect(document.activeElement).toBe(first);
    // Shift+Tab from first wraps to last.
    await user.tab({ shift: true });
    expect(document.activeElement).toBe(close);
  });

  it('restores focus to previously-focused element on close', async () => {
    const user = userEvent.setup();
    const Wrapper = () => {
      const [open, setOpen] = useState(false);
      return (
        <>
          <button onClick={() => setOpen(true)}>Trigger</button>
          <Modal.Root open={open} onOpenChange={setOpen} labelledBy="ttl">
            <Modal.Header>
              <h2 id="ttl">T</h2>
            </Modal.Header>
            <Modal.Body>
              <button onClick={() => setOpen(false)}>X</button>
            </Modal.Body>
          </Modal.Root>
        </>
      );
    };
    render(<Wrapper />);
    const trigger = screen.getByRole('button', { name: 'Trigger' });
    trigger.focus();
    await user.click(trigger);
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    await user.keyboard('{Escape}');
    // After close + animation, focus restored.
    await act(async () => {
      await new Promise((r) => setTimeout(r, 50));
    });
    expect(document.activeElement).toBe(trigger);
  });
});
