import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { ErrorBoundary } from './ErrorBoundary';

const Thrower = ({ shouldThrow }: { shouldThrow: boolean }) => {
  if (shouldThrow) {
    throw new Error('Boom');
  }

  return <div>All good</div>;
};

describe('ErrorBoundary', () => {
  it('renders children when no error is thrown', () => {
    render(
      <ErrorBoundary>
        <Thrower shouldThrow={false} />
      </ErrorBoundary>,
    );

    expect(screen.getByText('All good')).toBeInTheDocument();
  });

  it('renders the default fallback and reports errors', () => {
    const onError = vi.fn();
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => undefined);

    render(
      <ErrorBoundary onError={onError}>
        <Thrower shouldThrow />
      </ErrorBoundary>,
    );

    expect(screen.getByRole('alert')).toHaveTextContent('Something went wrong.');
    expect(screen.getByText('Boom')).toBeInTheDocument();
    expect(onError).toHaveBeenCalledTimes(1);

    consoleError.mockRestore();
  });

  it('supports render-prop fallbacks and reset', async () => {
    const user = userEvent.setup();
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    let shouldThrow = true;
    const RetryThrower = () => {
      if (shouldThrow) {
        throw new Error('First render failed');
      }

      return <div>Recovered</div>;
    };

    render(
      <ErrorBoundary
        fallback={({ error, reset }) => (
          <div role="alert">
            <span>{error.message}</span>
            <button
              type="button"
              onClick={() => {
                shouldThrow = false;
                reset();
              }}
            >
              Retry
            </button>
          </div>
        )}
      >
        <RetryThrower />
      </ErrorBoundary>,
    );

    expect(screen.getByRole('alert')).toHaveTextContent('First render failed');
    await user.click(screen.getByRole('button', { name: 'Retry' }));
    expect(screen.getByText('Recovered')).toBeInTheDocument();

    consoleError.mockRestore();
  });

  it('resets when resetKey changes', () => {
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const { rerender } = render(
      <ErrorBoundary resetKey="failed">
        <Thrower shouldThrow />
      </ErrorBoundary>,
    );

    expect(screen.getByRole('alert')).toBeInTheDocument();

    rerender(
      <ErrorBoundary resetKey="ready">
        <Thrower shouldThrow={false} />
      </ErrorBoundary>,
    );

    expect(screen.getByText('All good')).toBeInTheDocument();
    consoleError.mockRestore();
  });
});
