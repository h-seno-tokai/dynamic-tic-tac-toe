import { Routes, Route } from 'react-router-dom';

export const App = () => {
  return (
    <div className="min-h-screen bg-bg text-fg">
      <Routes>
        <Route path="/" element={<TitlePagePlaceholder />} />
        <Route path="*" element={<TitlePagePlaceholder />} />
      </Routes>
    </div>
  );
};

const TitlePagePlaceholder = () => (
  <main className="flex min-h-screen items-center justify-center p-8">
    <div className="text-center">
      <h1 className="text-4xl font-bold tracking-tight md:text-5xl">Dynamic Tic-Tac-Toe</h1>
      <p className="mt-3 text-muted">Implementation in progress</p>
      <p className="mt-1 text-sm text-muted">
        See{' '}
        <a
          className="text-accent underline"
          href="https://github.com/h-seno-tokai/dynamic-tic-tac-toe"
        >
          design docs
        </a>{' '}
        for details
      </p>
    </div>
  </main>
);
