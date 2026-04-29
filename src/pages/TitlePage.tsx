import { useTranslation } from 'react-i18next';
import { Bot, ChartNoAxesColumn, Gamepad2, Settings, UsersRound } from 'lucide-react';
import { Link } from 'react-router-dom';

export const TitlePage = () => {
  const { t } = useTranslation();

  const menuItems = [
    {
      to: '/local/setup',
      label: t('title.menu.local'),
      description: t('title.menu.localDesc'),
      icon: <UsersRound className="h-5 w-5" />,
    },
    {
      to: '/cpu/setup',
      label: t('title.menu.cpu'),
      description: t('title.menu.cpuDesc'),
      icon: <Bot className="h-5 w-5" />,
    },
    {
      to: '/rules',
      label: t('title.menu.rules'),
      description: t('title.menu.rulesDesc'),
      icon: <Gamepad2 className="h-5 w-5" />,
    },
    {
      to: '/stats',
      label: t('title.menu.stats'),
      description: t('title.menu.statsDesc'),
      icon: <ChartNoAxesColumn className="h-5 w-5" />,
    },
    {
      to: '/settings',
      label: t('title.menu.settings'),
      description: t('title.menu.settingsDesc'),
      icon: <Settings className="h-5 w-5" />,
    },
  ];

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-5xl flex-col px-5 py-6">
      <header className="pb-8 pt-12">
        <p className="text-sm font-medium text-accent">{t('title.tagline')}</p>
        <h1 className="mt-2 text-4xl font-bold tracking-normal md:text-6xl">
          {t('title.appName')}
        </h1>
        <p className="mt-4 max-w-2xl text-base leading-7 text-muted">{t('title.description')}</p>
      </header>

      <section className="grid gap-3 pb-8 md:grid-cols-2">
        {menuItems.map((item) => (
          <Link
            key={item.to}
            to={item.to}
            className="rounded-lg border-2 border-border bg-[color:var(--color-cell-light)] p-4 transition hover:border-accent focus-visible:border-accent"
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
    </main>
  );
};
