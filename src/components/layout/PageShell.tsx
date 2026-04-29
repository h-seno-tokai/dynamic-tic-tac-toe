import type { ReactNode } from 'react';

export interface PageShellProps {
  title: string;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
}

export const PageShell = ({ title, subtitle, actions, children }: PageShellProps) => {
  return (
    <main className="min-h-screen bg-bg text-fg">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8">
        <header className="flex flex-col gap-4 border-b border-border pb-5 sm:flex-row sm:items-end sm:justify-between">
          <div className="min-w-0">
            <h1 className="text-2xl font-bold tracking-tight text-fg sm:text-3xl">{title}</h1>
            {subtitle ? <div className="mt-2 max-w-3xl text-sm text-muted">{subtitle}</div> : null}
          </div>
          {actions ? (
            <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>
          ) : null}
        </header>
        <div className="min-w-0">{children}</div>
      </div>
    </main>
  );
};
