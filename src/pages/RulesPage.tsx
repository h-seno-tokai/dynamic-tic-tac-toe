import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/primitives';

export const RulesPage = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const sections = t('rules.sections', { returnObjects: true }) as {
    title: string;
    body: string;
  }[];

  return (
    <main className="mx-auto min-h-screen w-full max-w-3xl px-5 py-6">
      <header className="mb-8">
        <p className="text-sm font-medium text-accent">{t('rules.subtitle')}</p>
        <h1 className="mt-2 text-3xl font-bold tracking-normal">{t('rules.title')}</h1>
        <p className="mt-2 text-sm text-muted">{t('rules.desc')}</p>
      </header>

      <div className="grid gap-6">
        {sections.map((s) => (
          <section key={s.title}>
            <h2 className="mb-2 text-base font-semibold">{s.title}</h2>
            <p className="text-sm leading-7">
              {s.body.split('\n').map((line, i) => (
                <span key={i}>
                  {line}
                  {i < s.body.split('\n').length - 1 && <br />}
                </span>
              ))}
            </p>
          </section>
        ))}
      </div>

      <div className="mt-10">
        <Button variant="secondary" onClick={() => navigate('/')}>
          {t('common.backToMenu')}
        </Button>
      </div>
    </main>
  );
};
