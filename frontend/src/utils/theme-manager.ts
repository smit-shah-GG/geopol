/**
 * Theme manager -- dark-only.
 *
 * Light theme has been removed. This module ensures the DOM always has
 * data-theme="dark" set and provides getCurrentTheme() for any code
 * that still queries the active theme.
 */

import { invalidateColorCache } from './theme-colors';

export type Theme = 'dark';

/**
 * Return the current theme. Always 'dark'.
 */
export function getCurrentTheme(): Theme {
  return 'dark';
}

/**
 * Apply dark theme to the document root.
 * Called once at boot as a FOUC-prevention safety net (the inline script
 * in index.html already handles the fast path).
 */
export function applyStoredTheme(): void {
  document.documentElement.dataset['theme'] = 'dark';
  invalidateColorCache();
  const meta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]');
  if (meta) {
    meta.content = '#0a0e14';
  }
}
