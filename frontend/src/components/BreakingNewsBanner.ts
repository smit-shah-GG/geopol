/**
 * BreakingNewsBanner -- Full-width alert overlay for high-severity events.
 *
 * Listens for `geopol:breaking-news` CustomEvent on document.
 * Renders up to 3 concurrent alerts with auto-dismiss timers.
 * Visibility-aware: pauses timers when document is hidden.
 * Severity colors: critical = red (#dc2626), high = amber (#d97706).
 *
 * NOT a Panel subclass -- standalone overlay attached to document.body.
 * Ported from World Monitor BreakingNewsBanner pattern.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Event detail dispatched via `geopol:breaking-news` CustomEvent. */
export interface BreakingNewsDetail {
  id: string;
  headline: string;
  source: string;
  threatLevel: 'critical' | 'high';
  timestamp: Date;
}

interface ActiveAlert {
  detail: BreakingNewsDetail;
  element: HTMLElement;
  timer: ReturnType<typeof setTimeout> | null;
  remainingMs: number;
  timerStartedAt: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_ALERTS = 3;
const CRITICAL_DISMISS_MS = 60_000;
const HIGH_DISMISS_MS = 30_000;
const SOUND_COOLDOWN_MS = 5 * 60 * 1000;

// Minimal notification beep as base64 WAV (32 bytes, ~50ms)
const ALERT_SOUND_B64 =
  'data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2teleQYjfKapmWswEjCJvuPQfSoXZZ+3qqBJESSP0unGaxMJVYiytrFeLhR6p8znrFUXRW+bs7V3Qx1hn8Xjp1cYPnegprhkMCFmoLi1k0sZTYGlqqlUIA==';

// ---------------------------------------------------------------------------
// BreakingNewsBanner
// ---------------------------------------------------------------------------

export class BreakingNewsBanner {
  private container: HTMLElement;
  private activeAlerts: ActiveAlert[] = [];
  private audio: HTMLAudioElement | null = null;
  private lastSoundMs = 0;
  private dismissed = new Map<string, number>();

  // Bound event handlers for cleanup
  private boundOnAlert: (e: Event) => void;
  private boundOnVisibility: () => void;

  constructor() {
    this.container = document.createElement('div');
    this.container.className = 'breaking-news-container';
    document.body.appendChild(this.container);

    this.initAudio();

    this.boundOnAlert = (e: Event) =>
      this.handleAlert((e as CustomEvent<BreakingNewsDetail>).detail);
    this.boundOnVisibility = () => this.handleVisibility();

    document.addEventListener('geopol:breaking-news', this.boundOnAlert);
    document.addEventListener('visibilitychange', this.boundOnVisibility);

    // Click delegation on container
    this.container.addEventListener('click', (e) => {
      const target = e.target as HTMLElement;
      const alertEl = target.closest('.breaking-alert') as HTMLElement | null;
      if (!alertEl) return;

      if (target.closest('.breaking-alert-dismiss')) {
        const id = alertEl.getAttribute('data-alert-id');
        if (id) this.dismissAlert(id);
        return;
      }

      // Click headline => dispatch navigation event for future wiring
      const headline = alertEl.querySelector('.breaking-alert-headline');
      if (headline && headline.contains(target)) {
        document.dispatchEvent(
          new CustomEvent('geopol:navigate-alert', {
            detail: { id: alertEl.getAttribute('data-alert-id') },
          }),
        );
      }
    });
  }

  // -----------------------------------------------------------------------
  // Audio
  // -----------------------------------------------------------------------

  private initAudio(): void {
    try {
      this.audio = new Audio(ALERT_SOUND_B64);
      this.audio.volume = 0.3;
    } catch {
      this.audio = null;
    }
  }

  private playSound(): void {
    // Respect user preference (default: off)
    const pref = localStorage.getItem('geopol-alert-sound');
    if (pref !== 'on' || !this.audio) return;
    if (Date.now() - this.lastSoundMs < SOUND_COOLDOWN_MS) return;

    this.audio.currentTime = 0;
    this.audio.play()?.catch(() => {
      /* autoplay policy may block; ignore */
    });
    this.lastSoundMs = Date.now();
  }

  // -----------------------------------------------------------------------
  // Alert handling
  // -----------------------------------------------------------------------

  private isDismissedRecently(id: string): boolean {
    const ts = this.dismissed.get(id);
    if (ts === undefined) return false;
    // 30 min cooldown before same alert can reappear
    if (Date.now() - ts >= 30 * 60 * 1000) {
      this.dismissed.delete(id);
      return false;
    }
    return true;
  }

  private handleAlert(detail: BreakingNewsDetail): void {
    if (this.isDismissedRecently(detail.id)) return;

    // Deduplicate by ID
    const existing = this.activeAlerts.find((a) => a.detail.id === detail.id);
    if (existing) return;

    // Critical alerts evict high alerts
    if (detail.threatLevel === 'critical') {
      const highAlerts = this.activeAlerts.filter(
        (a) => a.detail.threatLevel === 'high',
      );
      for (const h of highAlerts) {
        this.removeAlert(h);
        const idx = this.activeAlerts.indexOf(h);
        if (idx !== -1) this.activeAlerts.splice(idx, 1);
      }
    }

    // Enforce max concurrent
    while (this.activeAlerts.length >= MAX_ALERTS) {
      const oldest = this.activeAlerts.shift();
      if (oldest) this.removeAlert(oldest);
    }

    const el = this.createAlertElement(detail);
    this.container.appendChild(el);

    const dismissMs =
      detail.threatLevel === 'critical' ? CRITICAL_DISMISS_MS : HIGH_DISMISS_MS;
    const now = Date.now();

    const active: ActiveAlert = {
      detail,
      element: el,
      timer: null,
      remainingMs: dismissMs,
      timerStartedAt: now,
    };

    // Start timer only if page is visible
    if (!document.hidden) {
      active.timer = setTimeout(
        () => this.dismissAlert(detail.id),
        dismissMs,
      );
    }

    this.activeAlerts.push(active);
    this.playSound();
    this.updateOffset();
  }

  // -----------------------------------------------------------------------
  // DOM construction
  // -----------------------------------------------------------------------

  private createAlertElement(detail: BreakingNewsDetail): HTMLElement {
    const el = document.createElement('div');
    el.className = `breaking-alert severity-${detail.threatLevel}`;
    el.setAttribute('data-alert-id', detail.id);
    el.style.cursor = 'pointer';

    const icon = detail.threatLevel === 'critical' ? '!!' : '!';
    const levelText =
      detail.threatLevel === 'critical' ? 'CRITICAL' : 'HIGH';
    const timeAgo = this.formatTimeAgo(detail.timestamp);

    const iconSpan = document.createElement('span');
    iconSpan.className = 'breaking-alert-icon';
    iconSpan.textContent = icon;

    const content = document.createElement('div');
    content.className = 'breaking-alert-content';

    const levelSpan = document.createElement('span');
    levelSpan.className = 'breaking-alert-level';
    levelSpan.textContent = levelText;

    const headlineSpan = document.createElement('span');
    headlineSpan.className = 'breaking-alert-headline';
    headlineSpan.textContent = detail.headline;

    const metaSpan = document.createElement('span');
    metaSpan.className = 'breaking-alert-meta';
    metaSpan.textContent = `${detail.source} \u00b7 ${timeAgo}`;

    content.appendChild(levelSpan);
    content.appendChild(headlineSpan);
    content.appendChild(metaSpan);

    const dismissBtn = document.createElement('button');
    dismissBtn.className = 'breaking-alert-dismiss';
    dismissBtn.textContent = '\u00d7';
    dismissBtn.title = 'Dismiss alert';

    el.appendChild(iconSpan);
    el.appendChild(content);
    el.appendChild(dismissBtn);

    return el;
  }

  // -----------------------------------------------------------------------
  // Dismiss / remove
  // -----------------------------------------------------------------------

  private dismissAlert(id: string): void {
    this.dismissed.set(id, Date.now());
    const idx = this.activeAlerts.findIndex((a) => a.detail.id === id);
    if (idx === -1) return;
    const active = this.activeAlerts[idx]!;
    this.removeAlert(active);
    this.activeAlerts.splice(idx, 1);
    this.updateOffset();
  }

  private removeAlert(active: ActiveAlert): void {
    if (active.timer) clearTimeout(active.timer);
    active.element.remove();
  }

  // -----------------------------------------------------------------------
  // Visibility handling (pause/resume timers)
  // -----------------------------------------------------------------------

  private handleVisibility(): void {
    const now = Date.now();
    if (document.hidden) {
      // Pause all timers
      for (const active of this.activeAlerts) {
        if (active.timer) {
          clearTimeout(active.timer);
          active.timer = null;
          const elapsed = now - active.timerStartedAt;
          active.remainingMs = Math.max(0, active.remainingMs - elapsed);
        }
      }
    } else {
      // Resume timers
      const expired: string[] = [];
      for (const active of this.activeAlerts) {
        if (!active.timer && active.remainingMs > 0) {
          active.timerStartedAt = now;
          active.timer = setTimeout(
            () => this.dismissAlert(active.detail.id),
            active.remainingMs,
          );
        } else if (active.remainingMs <= 0) {
          expired.push(active.detail.id);
        }
      }
      for (const id of expired) this.dismissAlert(id);
    }
  }

  // -----------------------------------------------------------------------
  // Body offset management
  // -----------------------------------------------------------------------

  private updateOffset(): void {
    const height = this.container.offsetHeight;
    document.documentElement.style.setProperty(
      '--breaking-alert-offset',
      height > 0 ? `${height}px` : '0px',
    );
    document.body.classList.toggle(
      'has-breaking-alert',
      this.activeAlerts.length > 0,
    );
  }

  // -----------------------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------------------

  private formatTimeAgo(date: Date): string {
    const ms = Date.now() - date.getTime();
    if (ms < 60_000) return 'just now';
    if (ms < 3_600_000) return `${Math.floor(ms / 60_000)}m ago`;
    return `${Math.floor(ms / 3_600_000)}h ago`;
  }

  // -----------------------------------------------------------------------
  // Cleanup
  // -----------------------------------------------------------------------

  public destroy(): void {
    document.removeEventListener('geopol:breaking-news', this.boundOnAlert);
    document.removeEventListener('visibilitychange', this.boundOnVisibility);

    for (const active of this.activeAlerts) {
      if (active.timer) clearTimeout(active.timer);
    }
    this.activeAlerts = [];
    this.container.remove();
    document.body.classList.remove('has-breaking-alert');
    document.documentElement.style.removeProperty('--breaking-alert-offset');
  }
}
