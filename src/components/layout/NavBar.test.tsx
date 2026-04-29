import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import { NavBar } from './NavBar';

const renderNav = (path = '/') =>
  render(
    <MemoryRouter initialEntries={[path]}>
      <NavBar />
    </MemoryRouter>,
  );

describe('NavBar', () => {
  it('renders the brand and default links', () => {
    renderNav();

    expect(screen.getByRole('navigation', { name: 'Main navigation' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Dynamic Tic-Tac-Toe' })).toHaveAttribute('href', '/');

    for (const label of ['Home', 'Local', 'CPU', 'Rules', 'Settings', 'Stats']) {
      expect(screen.getByRole('link', { name: label })).toBeInTheDocument();
    }
  });

  it('marks the current route active', () => {
    renderNav('/cpu/setup');

    expect(screen.getByText('CPU')).toHaveAttribute('aria-current', 'page');
    expect(screen.getByText('CPU')).toHaveAttribute('data-active', 'true');
    expect(screen.getByText('Home')).toHaveAttribute('data-active', 'false');
  });

  it('supports custom nav items and brand', () => {
    render(
      <MemoryRouter initialEntries={['/custom']}>
        <NavBar
          brand="DTTT"
          items={[
            { label: 'Start', to: '/', end: true },
            { label: 'Custom', to: '/custom' },
          ]}
        />
      </MemoryRouter>,
    );

    expect(screen.getByRole('link', { name: 'DTTT' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Start' })).toBeInTheDocument();
    expect(screen.getByText('Custom')).toHaveAttribute('aria-current', 'page');
    expect(screen.queryByRole('link', { name: 'CPU' })).not.toBeInTheDocument();
  });
});
