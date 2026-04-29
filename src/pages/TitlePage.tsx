import { Bot, ChartNoAxesColumn, Gamepad2, Settings, UsersRound } from 'lucide-react';
import { Link } from 'react-router-dom';

const menuItems = [
  {
    to: '/local/setup',
    label: 'ローカル2人対戦',
    description: '同じ端末で交互に手を進めます。',
    icon: <UsersRound className="h-5 w-5" />,
  },
  {
    to: '/cpu/setup',
    label: 'CPU対戦',
    description: '学習済みAI接続前は暫定CPUで動かします。',
    icon: <Bot className="h-5 w-5" />,
  },
  {
    to: '/rules',
    label: 'ルール',
    description: '覆い被せと勝利条件を確認します。',
    icon: <Gamepad2 className="h-5 w-5" />,
  },
  {
    to: '/stats',
    label: '戦績',
    description: 'CPU難易度別の記録を表示します。',
    icon: <ChartNoAxesColumn className="h-5 w-5" />,
  },
  {
    to: '/settings',
    label: '設定',
    description: 'テーマ、音量、言語を変更します。',
    icon: <Settings className="h-5 w-5" />,
  },
];

export const TitlePage = () => {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-5xl flex-col px-5 py-6">
      <header className="flex flex-1 flex-col justify-end pb-8">
        <p className="text-sm font-medium text-accent">Gobblet inspired board game</p>
        <h1 className="mt-2 text-4xl font-bold tracking-normal md:text-6xl">Dynamic Tic-Tac-Toe</h1>
        <p className="mt-4 max-w-2xl text-base leading-7 text-muted">
          サイズ違いの駒を重ねて、見えている自分の駒を一列に並べる対戦ゲームです。
        </p>
      </header>

      <section className="grid gap-3 pb-8 md:grid-cols-2">
        {menuItems.map((item) => (
          <Link
            key={item.to}
            to={item.to}
            className="rounded-lg border border-border bg-[color:var(--color-cell-light)] p-4 transition hover:border-accent focus-visible:border-accent"
          >
            <div className="flex items-start gap-3">
              <span className="mt-0.5 text-accent">{item.icon}</span>
              <span>
                <span className="block text-lg font-semibold">{item.label}</span>
                <span className="mt-1 block text-sm leading-6 text-muted">{item.description}</span>
              </span>
            </div>
          </Link>
        ))}
      </section>

      <footer className="flex flex-wrap gap-2 pb-2 text-sm">
        <Link
          to="/local/setup"
          className="inline-flex h-8 items-center justify-center rounded-md px-3 text-sm font-medium text-fg hover:bg-[color:var(--color-cell-light)]"
        >
          すぐ始める
        </Link>
      </footer>
    </main>
  );
};
