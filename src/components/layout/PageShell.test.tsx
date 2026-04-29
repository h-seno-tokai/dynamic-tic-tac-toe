import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { PageShell } from './PageShell';

describe('PageShell', () => {
  it('renders title, subtitle, actions, and children', () => {
    render(
      <PageShell
        title="Local game"
        subtitle={<span>Play on this device</span>}
        actions={<button type="button">New game</button>}
      >
        <section>Board area</section>
      </PageShell>,
    );

    expect(screen.getByRole('heading', { name: 'Local game', level: 1 })).toBeInTheDocument();
    expect(screen.getByText('Play on this device')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'New game' })).toBeInTheDocument();
    expect(screen.getByText('Board area')).toBeInTheDocument();
  });

  it('does not require subtitle or actions', () => {
    render(<PageShell title="Stats">Wins and losses</PageShell>);

    expect(screen.getByRole('heading', { name: 'Stats' })).toBeInTheDocument();
    expect(screen.getByText('Wins and losses')).toBeInTheDocument();
  });
});
