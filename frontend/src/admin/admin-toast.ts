/**
 * Shared toast notification for admin panels.
 *
 * Renders a fixed-position toast at bottom-right with auto-dismiss.
 * CSS classes: .admin-toast, .toast-error, .toast-visible (in admin-styles.css).
 */

import { h } from '@/utils/dom-utils';

export function showToast(message: string, isError = false): void {
  const existing = document.querySelector('.admin-toast');
  if (existing) existing.remove();

  const toast = h('div', {
    className: `admin-toast ${isError ? 'toast-error' : ''}`,
  }, message);
  document.body.appendChild(toast);

  setTimeout(() => { toast.classList.add('toast-visible'); }, 10);
  setTimeout(() => {
    toast.classList.remove('toast-visible');
    setTimeout(() => toast.remove(), 300);
  }, 3_000);
}
