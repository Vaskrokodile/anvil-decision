import { useState } from 'react';
import { Moon, Sun } from '@phosphor-icons/react';
import { getTheme, toggleTheme } from '../lib/theme';

/** Monochrome theme switch — sun/moon pill matching the nav's ghost buttons. */
export default function ThemeToggle({ hero = false }: { hero?: boolean }) {
  const [dark, setDark] = useState(() => getTheme() === 'dark');
  return (
    <button
      type="button"
      onClick={() => setDark(toggleTheme() === 'dark')}
      aria-label={dark ? 'Switch to light theme' : 'Switch to dark theme'}
      title={dark ? 'Light theme' : 'Dark theme'}
      className={`flex h-8 w-8 items-center justify-center rounded-full border transition-colors ${
        hero
          ? 'border-ice-line text-ice-dim hover:border-ice-dim hover:text-ice'
          : 'border-line text-dim hover:border-fg hover:text-fg'
      }`}
    >
      {dark ? <Sun size={14} /> : <Moon size={14} />}
    </button>
  );
}
