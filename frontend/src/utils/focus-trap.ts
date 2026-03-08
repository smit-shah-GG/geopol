/**
 * Reusable focus trap for modal dialogs.
 *
 * trapFocus(container) queries focusable elements inside the container and
 * constrains Tab cycling within them. Returns a cleanup function that removes
 * the keydown listener and restores focus to the previously focused element.
 *
 * Focusable elements are re-queried on every Tab press to handle dynamic
 * content (lazy-loaded buttons, async form fields, etc.).
 */

// -----------------------------------------------------------------------
// Focusable selector
// -----------------------------------------------------------------------

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(', ');

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

/**
 * Trap keyboard focus within `container`.
 *
 * - Stores the currently focused element as `previousFocus`
 * - On Tab/Shift+Tab: cycles focus within focusable children
 * - Focuses the first focusable element (or the container itself if none)
 * - Returns a `releaseFocus` cleanup function
 */
export function trapFocus(container: HTMLElement): () => void {
  const previousFocus = document.activeElement as HTMLElement | null;

  const onKeyDown = (e: KeyboardEvent): void => {
    if (e.key !== 'Tab') return;

    // Re-query on every tab press for dynamic content support
    const focusable = Array.from(
      container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR),
    );

    if (focusable.length === 0) return;

    const first = focusable[0]!;
    const last = focusable[focusable.length - 1]!;

    if (e.shiftKey) {
      if (document.activeElement === first) {
        e.preventDefault();
        last.focus();
      }
    } else {
      if (document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
  };

  container.addEventListener('keydown', onKeyDown);

  // Focus first focusable element, or the container itself
  const focusable = container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR);
  if (focusable.length > 0) {
    focusable[0]!.focus();
  } else {
    container.setAttribute('tabindex', '-1');
    container.focus();
  }

  // Return cleanup function
  return (): void => {
    container.removeEventListener('keydown', onKeyDown);
    previousFocus?.focus();
  };
}

/**
 * No-op export for import convenience.
 *
 * The real release function is returned by trapFocus(). This export exists
 * so consumers can import both symbols from the same module without
 * conditional logic.
 */
export function releaseFocus(): void {
  // no-op -- the real release is the function returned by trapFocus()
}
