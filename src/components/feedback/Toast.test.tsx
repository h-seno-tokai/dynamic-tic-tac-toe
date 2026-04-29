import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { Toast } from './Toast';

describe('Toast', () => {
  it('renders polite status notifications by default', () => {
    render(<Toast title="Saved">Settings updated</Toast>);

    const toast = screen.getByRole('status');
    expect(toast).toHaveAttribute('aria-live', 'polite');
    expect(toast).toHaveAttribute('data-variant', 'info');
    expect(screen.getByText('Saved')).toBeInTheDocument();
    expect(screen.getByText('Settings updated')).toBeInTheDocument();
  });

  it('renders assertive alerts for errors and warnings', () => {
    const { rerender } = render(<Toast variant="error">Failed to save</Toast>);

    expect(screen.getByRole('alert')).toHaveAttribute('aria-live', 'assertive');
    expect(screen.getByRole('alert')).toHaveAttribute('data-variant', 'error');

    rerender(<Toast variant="warning">Connection is slow</Toast>);
    expect(screen.getByRole('alert')).toHaveAttribute('data-variant', 'warning');
  });

  it('renders success notifications as status', () => {
    render(<Toast variant="success">Game created</Toast>);

    expect(screen.getByRole('status')).toHaveAttribute('data-variant', 'success');
  });
});
