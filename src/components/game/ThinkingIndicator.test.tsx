import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ThinkingIndicator } from './ThinkingIndicator';

describe('ThinkingIndicator', () => {
  it('renders default label', () => {
    render(<ThinkingIndicator />);
    expect(screen.getByText('CPU 思考中…')).toBeInTheDocument();
  });

  it('shows elapsed seconds with one decimal', () => {
    render(<ThinkingIndicator thinkingMs={2500} />);
    expect(screen.getByTestId('thinking-elapsed')).toHaveTextContent('2.5s');
  });

  it('omits elapsed when thinkingMs is undefined', () => {
    render(<ThinkingIndicator />);
    expect(screen.queryByTestId('thinking-elapsed')).not.toBeInTheDocument();
  });

  it('renders difficulty badge when provided', () => {
    render(<ThinkingIndicator difficulty={7} />);
    const badge = screen.getByTestId('thinking-difficulty');
    expect(badge).toHaveTextContent('Lv.7');
    expect(badge).toHaveAttribute('aria-label', '難易度 7');
  });

  it('uses custom label when provided', () => {
    render(<ThinkingIndicator label="考え中..." />);
    expect(screen.getByText('考え中...')).toBeInTheDocument();
  });
});
