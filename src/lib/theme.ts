// theme.ts — light/dark theme switch. Persists to localStorage; applied as a
// data-theme attribute on <html>, which remaps the @theme color tokens in
// index.css. Default follows prefers-color-scheme.
export type Theme = 'light' | 'dark';

const KEY = 'anvil-theme';

export function getTheme(): Theme {
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === 'light' || saved === 'dark') return saved;
  } catch {
    /* storage unavailable */
  }
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function applyTheme(t: Theme) {
  document.documentElement.dataset.theme = t;
  try {
    localStorage.setItem(KEY, t);
  } catch {
    /* storage unavailable */
  }
}

export function toggleTheme(): Theme {
  const next: Theme = (document.documentElement.dataset.theme ?? getTheme()) === 'dark' ? 'light' : 'dark';
  applyTheme(next);
  return next;
}
