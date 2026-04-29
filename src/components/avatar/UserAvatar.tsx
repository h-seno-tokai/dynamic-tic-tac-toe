import type { ImgHTMLAttributes } from 'react';

const DICEBEAR_AVATAAARS_BASE_URL = 'https://api.dicebear.com/9.x/avataaars/svg';

export interface DiceBearAvatarUrlOptions {
  size?: number;
  backgroundColor?: string;
}

export function buildDiceBearAvataaarsUrl(
  seed: string,
  { size, backgroundColor }: DiceBearAvatarUrlOptions = {},
): string {
  const params = new URLSearchParams({ seed });

  if (size != null) {
    params.set('size', String(size));
  }

  if (backgroundColor) {
    params.set('backgroundColor', backgroundColor.replace(/^#/, ''));
  }

  return `${DICEBEAR_AVATAAARS_BASE_URL}?${params.toString()}`;
}

export interface UserAvatarProps extends Omit<
  ImgHTMLAttributes<HTMLImageElement>,
  'alt' | 'height' | 'src' | 'width'
> {
  seed: string;
  size?: number;
  label?: string;
  selected?: boolean;
}

export function UserAvatar({
  seed,
  size = 48,
  label,
  selected = false,
  className,
  ...rest
}: UserAvatarProps) {
  const url = buildDiceBearAvataaarsUrl(seed, { size });

  return (
    <img
      {...rest}
      alt={label ?? ''}
      aria-hidden={label ? undefined : true}
      src={url}
      width={size}
      height={size}
      data-selected={selected ? 'true' : 'false'}
      className={[
        'inline-block shrink-0 rounded-full border bg-bg object-cover transition',
        selected
          ? 'border-accent ring-2 ring-accent ring-offset-2 ring-offset-bg'
          : 'border-border',
        className,
      ]
        .filter(Boolean)
        .join(' ')}
    />
  );
}
