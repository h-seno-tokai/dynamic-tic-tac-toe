import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { LoadingSpinner } from './LoadingSpinner';

describe('LoadingSpinner', () => {
  it('renders with default props and label', () => {
    render(<LoadingSpinner />);
    const status = screen.getByRole('status');
    expect(status).toBeInTheDocument();
    expect(status).toHaveAttribute('aria-label', 'Loading...');
  });

  it('accepts a custom aria-label', () => {
    render(<LoadingSpinner aria-label="Fetching model" />);
    expect(screen.getByRole('status')).toHaveAttribute('aria-label', 'Fetching model');
  });

  it('applies size class for sm/md/lg', () => {
    const { rerender, container } = render(<LoadingSpinner size="sm" />);
    expect(container.querySelector('svg')?.getAttribute('class')).toContain('h-4');
    rerender(<LoadingSpinner size="md" />);
    expect(container.querySelector('svg')?.getAttribute('class')).toContain('h-6');
    rerender(<LoadingSpinner size="lg" />);
    expect(container.querySelector('svg')?.getAttribute('class')).toContain('h-8');
  });

  it('uses motion-safe spin class so reduced-motion is respected', () => {
    const { container } = render(<LoadingSpinner />);
    expect(container.querySelector('svg')?.getAttribute('class')).toContain(
      'motion-safe:animate-spin',
    );
  });
});
