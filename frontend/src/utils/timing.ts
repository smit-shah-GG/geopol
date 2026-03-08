/**
 * Shared debounce and throttle utilities.
 *
 * Extracted from SearchBar.ts for reuse across the codebase.
 */

// -----------------------------------------------------------------------
// Debounce (trailing edge)
// -----------------------------------------------------------------------

/**
 * Standard trailing-edge debounce. Delays invocation of `fn` until `ms`
 * milliseconds have elapsed since the last call. Resets the timer on each
 * subsequent call within the window.
 */
export function debounce<T extends (...args: never[]) => void>(
  fn: T,
  ms: number,
): (...args: Parameters<T>) => void {
  let timer: ReturnType<typeof setTimeout> | null = null;
  return (...args: Parameters<T>): void => {
    if (timer !== null) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      fn(...args);
    }, ms);
  };
}

// -----------------------------------------------------------------------
// Throttle (leading edge)
// -----------------------------------------------------------------------

/**
 * Leading-edge throttle. Fires `fn` immediately on first call, then
 * suppresses subsequent calls for `ms` milliseconds. After the cooldown
 * expires, the next call fires immediately again.
 */
export function throttle<T extends (...args: never[]) => void>(
  fn: T,
  ms: number,
): (...args: Parameters<T>) => void {
  let lastCall = 0;
  return (...args: Parameters<T>): void => {
    const now = Date.now();
    if (now - lastCall >= ms) {
      lastCall = now;
      fn(...args);
    }
  };
}
