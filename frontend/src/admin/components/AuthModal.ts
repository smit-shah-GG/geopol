/**
 * AuthModal -- admin key entry overlay with brute-force rate limiting.
 *
 * Renders a fixed overlay dimming the page with a centered modal card.
 * On successful verification via AdminClient.verify(), resolves the
 * waitForAuth() promise with the validated key and persists it in
 * sessionStorage.
 *
 * Rate limiting: 5 consecutive failures trigger a 30-second cooldown
 * with a visible countdown timer.
 */

import { h } from '@/utils/dom-utils';
import { AdminClient } from '@/admin/admin-client';

const SESSION_KEY = 'geopol_admin_key';
const MAX_ATTEMPTS = 5;
const COOLDOWN_MS = 30_000;

export class AuthModal {
  private overlay: HTMLElement | null = null;
  private resolve: ((key: string) => void) | null = null;
  private attempts = 0;
  private cooldownTimer: ReturnType<typeof setInterval> | null = null;

  /**
   * Show the auth modal and wait for a valid admin key.
   * Resolves with the verified key string.
   */
  waitForAuth(): Promise<string> {
    return new Promise<string>((res) => {
      this.resolve = res;
      this.render();
    });
  }

  /** Remove the overlay from the DOM and clean up timers. */
  destroy(): void {
    if (this.cooldownTimer !== null) {
      clearInterval(this.cooldownTimer);
      this.cooldownTimer = null;
    }
    if (this.overlay) {
      this.overlay.remove();
      this.overlay = null;
    }
    this.resolve = null;
  }

  // -----------------------------------------------------------------------
  // Private
  // -----------------------------------------------------------------------

  private render(): void {
    const errorMsg = h('div', { className: 'auth-error', style: 'display:none' });
    const input = h('input', {
      type: 'password',
      className: 'auth-input',
      placeholder: 'Enter admin key',
      autocomplete: 'off',
      spellcheck: 'false',
    }) as HTMLInputElement;

    const submitBtn = h('button', { className: 'auth-submit' }, 'AUTHENTICATE') as HTMLButtonElement;

    const handleSubmit = (): void => {
      const key = input.value.trim();
      if (!key) return;
      void this.attemptAuth(key, input, submitBtn, errorMsg);
    };

    submitBtn.addEventListener('click', handleSubmit);
    input.addEventListener('keydown', (e: Event) => {
      if ((e as KeyboardEvent).key === 'Enter') handleSubmit();
    });

    const card = h('div', { className: 'auth-card' },
      h('h2', { className: 'auth-title' }, 'ADMIN ACCESS'),
      input,
      submitBtn,
      errorMsg,
    );

    this.overlay = h('div', { className: 'admin-auth-overlay' }, card);
    document.body.appendChild(this.overlay);

    // Focus the input after mount
    requestAnimationFrame(() => input.focus());
  }

  private async attemptAuth(
    key: string,
    input: HTMLInputElement,
    btn: HTMLButtonElement,
    errorEl: HTMLElement,
  ): Promise<void> {
    // Disable while verifying
    input.disabled = true;
    btn.disabled = true;
    btn.textContent = 'VERIFYING...';

    try {
      const client = new AdminClient(key);
      await client.verify();

      // Success -- persist and resolve
      sessionStorage.setItem(SESSION_KEY, key);
      this.resolve?.(key);
      this.destroy();
    } catch {
      this.attempts++;
      input.disabled = false;
      btn.disabled = false;
      btn.textContent = 'AUTHENTICATE';
      input.value = '';
      input.focus();

      if (this.attempts >= MAX_ATTEMPTS) {
        this.startCooldown(input, btn, errorEl);
      } else {
        errorEl.textContent = `Invalid admin key (${this.attempts}/${MAX_ATTEMPTS})`;
        errorEl.style.display = 'block';
      }
    }
  }

  private startCooldown(
    input: HTMLInputElement,
    btn: HTMLButtonElement,
    errorEl: HTMLElement,
  ): void {
    input.disabled = true;
    btn.disabled = true;

    let remaining = COOLDOWN_MS / 1000;
    errorEl.textContent = `Too many attempts. Retry in ${remaining}s`;
    errorEl.style.display = 'block';

    this.cooldownTimer = setInterval(() => {
      remaining--;
      if (remaining <= 0) {
        if (this.cooldownTimer !== null) {
          clearInterval(this.cooldownTimer);
          this.cooldownTimer = null;
        }
        this.attempts = 0;
        input.disabled = false;
        btn.disabled = false;
        errorEl.textContent = '';
        errorEl.style.display = 'none';
        input.focus();
      } else {
        errorEl.textContent = `Too many attempts. Retry in ${remaining}s`;
      }
    }, 1000);
  }
}
