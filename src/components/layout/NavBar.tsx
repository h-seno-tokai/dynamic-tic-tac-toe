import type { ReactNode } from 'react';
import { BarChart3, Bot, Home, MapPin, ScrollText, Settings } from 'lucide-react';
import { Link, NavLink } from 'react-router-dom';

export interface NavItem {
  label: string;
  to: string;
  icon?: ReactNode;
  end?: boolean;
}

export interface NavBarProps {
  brand?: ReactNode;
  items?: NavItem[];
  className?: string;
}

const defaultItems: NavItem[] = [
  { label: 'Home', to: '/', icon: <Home className="h-4 w-4" />, end: true },
  { label: 'Local', to: '/local/setup', icon: <MapPin className="h-4 w-4" /> },
  { label: 'CPU', to: '/cpu/setup', icon: <Bot className="h-4 w-4" /> },
  { label: 'Rules', to: '/rules', icon: <ScrollText className="h-4 w-4" /> },
  { label: 'Settings', to: '/settings', icon: <Settings className="h-4 w-4" /> },
  { label: 'Stats', to: '/stats', icon: <BarChart3 className="h-4 w-4" /> },
];

export const NavBar = ({
  brand = 'Dynamic Tic-Tac-Toe',
  items = defaultItems,
  className,
}: NavBarProps) => {
  return (
    <nav
      aria-label="Main navigation"
      className={[
        'bg-bg/95 supports-[backdrop-filter]:bg-bg/80 border-b border-border text-fg backdrop-blur',
        className,
      ]
        .filter(Boolean)
        .join(' ')}
    >
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-3 px-4 py-3 sm:px-6 md:flex-row md:items-center md:justify-between lg:px-8">
        <Link
          to="/"
          className="inline-flex min-w-0 items-center font-semibold tracking-tight text-fg"
        >
          <span className="truncate">{brand}</span>
        </Link>
        <div className="flex gap-1 overflow-x-auto pb-1 md:pb-0">
          {items.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              {...(item.end !== undefined ? { end: item.end } : {})}
              className={({ isActive }) =>
                [
                  'inline-flex h-9 shrink-0 items-center gap-2 rounded-md px-3 text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-accent text-bg'
                    : 'text-muted hover:bg-[color:var(--color-cell-light)] hover:text-fg',
                ].join(' ')
              }
            >
              {({ isActive }) => (
                <span
                  className="inline-flex items-center gap-2"
                  aria-current={isActive ? 'page' : undefined}
                  data-active={isActive ? 'true' : 'false'}
                >
                  {item.icon ? <span aria-hidden="true">{item.icon}</span> : null}
                  {item.label}
                </span>
              )}
            </NavLink>
          ))}
        </div>
      </div>
    </nav>
  );
};
