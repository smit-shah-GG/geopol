/**
 * SourceManager -- rich feed management panel with CRUD operations,
 * per-feed health metadata, tier selectors, and auto-disable alerts.
 *
 * Reads from GET /api/v1/admin/feeds (FeedInfo) instead of the older
 * GET /api/v1/admin/sources (SourceInfo). Provides full feed lifecycle:
 * add, edit tier, enable/disable toggle, soft-delete with optional purge.
 *
 * Auto-refreshes every 30s. Cards update text nodes in-place to avoid
 * destroying interactive state (dropdowns, toggles).
 */

import { h, clearChildren } from '@/utils/dom-utils';
import { showToast } from '@/admin/admin-toast';
import type { AdminClient } from '@/admin/admin-client';
import type { FeedInfo, AddFeedRequest } from '@/admin/admin-types';
import type { AdminPanel } from '@/admin/panels/ProcessTable';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function relativeTime(iso: string | null): string {
  if (!iso) return 'Never';
  const diff = Date.now() - new Date(iso).getTime();
  if (diff < 0) return 'now';
  const s = Math.floor(diff / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const hr = Math.floor(m / 60);
  if (hr < 24) return `${hr}h ago`;
  const d = Math.floor(hr / 24);
  return `${d}d ago`;
}

function truncate(text: string, max: number): string {
  return text.length > max ? text.slice(0, max) + '...' : text;
}

/** Health dot class based on error count and enabled state. */
function healthClass(feed: FeedInfo): string {
  if (!feed.enabled && feed.error_count >= 5) return 'status-failed';
  if (feed.error_count >= 5) return 'status-failed';
  if (feed.error_count >= 1) return 'status-paused'; // yellow
  return 'status-success';
}

// ---------------------------------------------------------------------------
// Feed card refs for in-place updates
// ---------------------------------------------------------------------------

interface CardRefs {
  healthDot: HTMLElement;
  lastPoll: Text;
  articles24h: Text;
  articlesTotal: Text;
  avgPerPoll: Text;
  errorCount: Text;
  lastError: HTMLElement;
  toggle: HTMLInputElement;
  tierSelect: HTMLSelectElement;
}

// ---------------------------------------------------------------------------
// SourceManager
// ---------------------------------------------------------------------------

export class SourceManager implements AdminPanel {
  private el: HTMLElement | null = null;
  private alertBanner: HTMLElement | null = null;
  private grid: HTMLElement | null = null;
  private footer: HTMLElement | null = null;
  private addFormWrap: HTMLElement | null = null;
  private addFormVisible = false;
  private intervalId: ReturnType<typeof setInterval> | null = null;
  private cards: Map<number, { el: HTMLElement; refs: CardRefs }> = new Map();
  private feeds: FeedInfo[] = [];

  constructor(private readonly client: AdminClient) {}

  async mount(container: HTMLElement): Promise<void> {
    this.el = h('div', { className: 'source-manager' });

    // Header bar
    const header = h('div', { className: 'feed-mgmt-header' });
    const title = h('span', { className: 'feed-mgmt-title' }, 'Feed Management');
    const addBtn = h('button', { className: 'trigger-btn' }, '+ ADD FEED');
    addBtn.addEventListener('click', () => this.toggleAddForm());
    header.appendChild(title);
    header.appendChild(addBtn);
    this.el.appendChild(header);

    // Add feed form (hidden by default)
    this.addFormWrap = h('div', { className: 'feed-add-form hidden' });
    this.buildAddForm();
    this.el.appendChild(this.addFormWrap);

    // Auto-disabled alert banner (hidden by default)
    this.alertBanner = h('div', { className: 'feed-alert-banner hidden' });
    this.el.appendChild(this.alertBanner);

    // Feed card grid
    this.grid = h('div', { className: 'feed-grid' });
    this.el.appendChild(this.grid);

    // Stats footer
    this.footer = h('div', { className: 'feed-stats-footer' });
    this.el.appendChild(this.footer);

    container.appendChild(this.el);

    await this.refresh();
    this.intervalId = setInterval(() => { void this.refresh(); }, 30_000);
  }

  destroy(): void {
    if (this.intervalId !== null) clearInterval(this.intervalId);
    this.el?.remove();
    this.el = null;
    this.grid = null;
    this.footer = null;
    this.alertBanner = null;
    this.addFormWrap = null;
    this.cards.clear();
  }

  // -----------------------------------------------------------------------
  // Add feed form
  // -----------------------------------------------------------------------

  private toggleAddForm(): void {
    this.addFormVisible = !this.addFormVisible;
    this.addFormWrap?.classList.toggle('hidden', !this.addFormVisible);
  }

  private buildAddForm(): void {
    if (!this.addFormWrap) return;

    const nameInput = h('input', {
      type: 'text',
      className: 'config-input',
      placeholder: 'Feed name (e.g. "Reuters World")',
    }) as HTMLInputElement;

    const urlInput = h('input', {
      type: 'text',
      className: 'config-input',
      placeholder: 'RSS feed URL',
    }) as HTMLInputElement;

    const tierSelect = h('select', { className: 'feed-tier-select' }) as HTMLSelectElement;
    tierSelect.appendChild(h('option', { value: '1' }, 'Tier 1 (Wire/Official)'));
    tierSelect.appendChild(h('option', { value: '2', selected: true }, 'Tier 2 (Major)'));

    const categorySelect = h('select', { className: 'feed-tier-select' }) as HTMLSelectElement;
    for (const cat of ['regional', 'wire', 'mainstream', 'defense', 'thinktank', 'finance', 'crisis']) {
      categorySelect.appendChild(h('option', { value: cat }, cat.toUpperCase()));
    }

    const submitBtn = h('button', { className: 'save-btn' }, 'ADD FEED');
    submitBtn.addEventListener('click', async () => {
      const name = nameInput.value.trim();
      const url = urlInput.value.trim();
      if (!name || !url) {
        showToast('Name and URL are required', true);
        return;
      }

      const data: AddFeedRequest = {
        name,
        url,
        tier: Number(tierSelect.value) as 1 | 2,
        category: categorySelect.value,
      };

      try {
        submitBtn.setAttribute('disabled', '');
        await this.client.addFeed(data);
        showToast(`Feed "${name}" added`);
        nameInput.value = '';
        urlInput.value = '';
        this.toggleAddForm();
        await this.refresh();
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Failed to add feed';
        showToast(msg, true);
      } finally {
        submitBtn.removeAttribute('disabled');
      }
    });

    const row1 = h('div', { className: 'feed-form-row' });
    row1.appendChild(nameInput);
    row1.appendChild(urlInput);

    const row2 = h('div', { className: 'feed-form-row' });
    const tierLabel = h('label', { className: 'feed-form-label' }, 'Tier');
    const catLabel = h('label', { className: 'feed-form-label' }, 'Category');
    row2.appendChild(tierLabel);
    row2.appendChild(tierSelect);
    row2.appendChild(catLabel);
    row2.appendChild(categorySelect);
    row2.appendChild(submitBtn);

    this.addFormWrap.appendChild(row1);
    this.addFormWrap.appendChild(row2);
  }

  // -----------------------------------------------------------------------
  // Data refresh
  // -----------------------------------------------------------------------

  private async refresh(): Promise<void> {
    if (!this.grid) return;
    try {
      this.feeds = await this.client.getFeeds();
      this.updateCards();
      this.updateAlertBanner();
      this.updateFooter();
    } catch {
      // Keep stale data on error
    }
  }

  // -----------------------------------------------------------------------
  // Alert banner
  // -----------------------------------------------------------------------

  private updateAlertBanner(): void {
    if (!this.alertBanner) return;
    const autoDisabled = this.feeds.filter(
      (f) => !f.enabled && f.error_count >= 5,
    );
    if (autoDisabled.length > 0) {
      this.alertBanner.textContent =
        `${autoDisabled.length} feed${autoDisabled.length > 1 ? 's' : ''} auto-disabled due to repeated errors. Review below.`;
      this.alertBanner.classList.remove('hidden');
    } else {
      this.alertBanner.classList.add('hidden');
    }
  }

  // -----------------------------------------------------------------------
  // Stats footer
  // -----------------------------------------------------------------------

  private updateFooter(): void {
    if (!this.footer) return;
    const total = this.feeds.length;
    const enabled = this.feeds.filter((f) => f.enabled).length;
    const articles24h = this.feeds.reduce((sum, f) => sum + f.articles_24h, 0);
    const withErrors = this.feeds.filter((f) => f.error_count > 0).length;

    this.footer.textContent =
      `${total} feeds | ${enabled} enabled | ${articles24h} articles today | ${withErrors} with errors`;
  }

  // -----------------------------------------------------------------------
  // Card grid
  // -----------------------------------------------------------------------

  private updateCards(): void {
    if (!this.grid) return;

    // If card set changed (different IDs), rebuild everything
    const currentIds = new Set(this.feeds.map((f) => f.id));
    const cardIds = new Set(this.cards.keys());
    const idsMatch =
      currentIds.size === cardIds.size &&
      [...currentIds].every((id) => cardIds.has(id));

    if (!idsMatch || this.cards.size === 0) {
      this.buildCards();
      return;
    }

    // Update text nodes in-place
    for (const feed of this.feeds) {
      const card = this.cards.get(feed.id);
      if (card) this.updateCardRefs(card.refs, feed);
    }
  }

  private buildCards(): void {
    if (!this.grid) return;
    clearChildren(this.grid);
    this.cards.clear();

    for (const feed of this.feeds) {
      const refs = this.createCardRefs(feed);
      const card = this.buildCard(feed, refs);
      this.cards.set(feed.id, { el: card, refs });
      this.grid.appendChild(card);
    }
  }

  private createCardRefs(feed: FeedInfo): CardRefs {
    return {
      healthDot: h('span', {
        className: `status-dot ${healthClass(feed)}`,
      }),
      lastPoll: document.createTextNode(relativeTime(feed.last_poll_at)),
      articles24h: document.createTextNode(String(feed.articles_24h)),
      articlesTotal: document.createTextNode(String(feed.articles_total)),
      avgPerPoll: document.createTextNode(feed.avg_articles_per_poll.toFixed(1)),
      errorCount: document.createTextNode(String(feed.error_count)),
      lastError: h('span', {
        className: 'feed-error-text',
        title: feed.last_error ?? '',
      }, feed.last_error ? truncate(feed.last_error, 80) : ''),
      toggle: null as unknown as HTMLInputElement,
      tierSelect: null as unknown as HTMLSelectElement,
    };
  }

  private buildCard(feed: FeedInfo, refs: CardRefs): HTMLElement {
    const card = h('div', {
      className: `feed-card${!feed.enabled && feed.error_count >= 5 ? ' feed-auto-disabled' : ''}`,
    });

    // Header: name + tier badge
    const headerRow = h('div', { className: 'source-header' });
    headerRow.appendChild(h('span', { className: 'source-name' }, feed.name));
    const tierBadge = h('span', {
      className: `feed-tier-badge ${feed.tier === 1 ? 'tier-1' : 'tier-2'}`,
    }, `T${feed.tier}`);
    headerRow.appendChild(tierBadge);
    card.appendChild(headerRow);

    // URL (truncated, full in tooltip)
    const urlRow = h('div', { className: 'source-row' });
    urlRow.appendChild(h('span', { className: 'source-label' }, 'URL'));
    const urlVal = h('span', {
      className: 'source-value feed-url',
      title: feed.url,
    }, truncate(feed.url, 40));
    urlRow.appendChild(urlVal);
    card.appendChild(urlRow);

    // Health dot
    const healthRow = h('div', { className: 'source-row' });
    healthRow.appendChild(h('span', { className: 'source-label' }, 'Health'));
    healthRow.appendChild(refs.healthDot);
    card.appendChild(healthRow);

    // Last poll
    const pollRow = h('div', { className: 'source-row' });
    pollRow.appendChild(h('span', { className: 'source-label' }, 'Last Poll'));
    const pollVal = h('span', { className: 'source-value' });
    pollVal.appendChild(refs.lastPoll);
    pollRow.appendChild(pollVal);
    card.appendChild(pollRow);

    // Articles: X today / Y total
    const articlesRow = h('div', { className: 'source-row' });
    articlesRow.appendChild(h('span', { className: 'source-label' }, 'Articles'));
    const articlesVal = h('span', { className: 'source-value' });
    articlesVal.appendChild(refs.articles24h);
    articlesVal.appendChild(document.createTextNode(' today / '));
    articlesVal.appendChild(refs.articlesTotal);
    articlesVal.appendChild(document.createTextNode(' total'));
    articlesRow.appendChild(articlesVal);
    card.appendChild(articlesRow);

    // Avg per poll
    const avgRow = h('div', { className: 'source-row' });
    avgRow.appendChild(h('span', { className: 'source-label' }, 'Avg/Poll'));
    const avgVal = h('span', { className: 'source-value' });
    avgVal.appendChild(refs.avgPerPoll);
    avgRow.appendChild(avgVal);
    card.appendChild(avgRow);

    // Error count (only show if > 0)
    if (feed.error_count > 0) {
      const errCountRow = h('div', { className: 'source-row' });
      errCountRow.appendChild(h('span', { className: 'source-label feed-error-label' }, 'Errors'));
      const errCountVal = h('span', {
        className: `source-value ${feed.error_count >= 5 ? 'feed-error-critical' : 'feed-error-warning'}`,
      });
      errCountVal.appendChild(refs.errorCount);
      errCountRow.appendChild(errCountVal);
      card.appendChild(errCountRow);
    }

    // Last error (only show if non-null)
    if (feed.last_error) {
      const errRow = h('div', { className: 'source-row' });
      errRow.appendChild(h('span', { className: 'source-label feed-error-label' }, 'Last Error'));
      errRow.appendChild(refs.lastError);
      card.appendChild(errRow);
    }

    // Tier selector
    const tierRow = h('div', { className: 'source-row' });
    tierRow.appendChild(h('span', { className: 'source-label' }, 'Tier'));
    const tierSelect = h('select', { className: 'feed-tier-inline-select' }) as HTMLSelectElement;
    const opt1 = h('option', { value: '1' }, 'Tier 1') as HTMLOptionElement;
    const opt2 = h('option', { value: '2' }, 'Tier 2') as HTMLOptionElement;
    if (feed.tier === 1) opt1.selected = true;
    else opt2.selected = true;
    tierSelect.appendChild(opt1);
    tierSelect.appendChild(opt2);
    tierSelect.addEventListener('change', () => {
      void this.handleTierChange(feed.id, tierSelect);
    });
    refs.tierSelect = tierSelect;
    tierRow.appendChild(tierSelect);
    card.appendChild(tierRow);

    // Enable/disable toggle
    const toggleRow = h('div', { className: 'source-row' });
    toggleRow.appendChild(h('span', { className: 'source-label' }, 'Enabled'));
    const toggleLabel = h('label', { className: 'toggle-switch' });
    const toggle = h('input', {
      type: 'checkbox',
      className: 'toggle-input',
    }) as HTMLInputElement;
    toggle.checked = feed.enabled;
    toggle.addEventListener('change', () => {
      void this.handleToggle(feed.id, toggle);
    });
    refs.toggle = toggle;

    const slider = h('span', { className: 'toggle-slider' });
    toggleLabel.appendChild(toggle);
    toggleLabel.appendChild(slider);
    toggleRow.appendChild(toggleLabel);
    card.appendChild(toggleRow);

    // Delete button
    const deleteRow = h('div', { className: 'feed-delete-row' });
    const deleteBtn = h('button', { className: 'feed-delete-btn' }, 'DELETE');
    deleteBtn.addEventListener('click', () => {
      void this.handleDelete(feed);
    });
    deleteRow.appendChild(deleteBtn);
    card.appendChild(deleteRow);

    return card;
  }

  private updateCardRefs(refs: CardRefs, feed: FeedInfo): void {
    refs.healthDot.className = `status-dot ${healthClass(feed)}`;
    refs.lastPoll.textContent = relativeTime(feed.last_poll_at);
    refs.articles24h.textContent = String(feed.articles_24h);
    refs.articlesTotal.textContent = String(feed.articles_total);
    refs.avgPerPoll.textContent = feed.avg_articles_per_poll.toFixed(1);
    refs.errorCount.textContent = String(feed.error_count);

    // Update last error text
    if (feed.last_error) {
      refs.lastError.textContent = truncate(feed.last_error, 80);
      refs.lastError.title = feed.last_error;
    } else {
      refs.lastError.textContent = '';
      refs.lastError.title = '';
    }

    // Toggle (only if not mid-user-interaction)
    if (document.activeElement !== refs.toggle) {
      refs.toggle.checked = feed.enabled;
    }

    // Tier select
    if (document.activeElement !== refs.tierSelect) {
      refs.tierSelect.value = String(feed.tier);
    }
  }

  // -----------------------------------------------------------------------
  // Event handlers
  // -----------------------------------------------------------------------

  private async handleToggle(feedId: number, toggle: HTMLInputElement): Promise<void> {
    const newState = toggle.checked;
    try {
      await this.client.updateFeed(feedId, { enabled: newState });
      showToast(newState ? 'Feed enabled' : 'Feed disabled');
    } catch {
      toggle.checked = !newState;
      showToast('Failed to update feed', true);
    }
  }

  private async handleTierChange(feedId: number, select: HTMLSelectElement): Promise<void> {
    const newTier = Number(select.value) as 1 | 2;
    try {
      await this.client.updateFeed(feedId, { tier: newTier });
      showToast(`Tier changed to ${newTier}`);
      await this.refresh();
    } catch {
      showToast('Failed to change tier', true);
      await this.refresh(); // Revert visual
    }
  }

  private async handleDelete(feed: FeedInfo): Promise<void> {
    // Build confirmation dialog
    const overlay = h('div', { className: 'confirm-dialog' });
    const card = h('div', { className: 'confirm-card' });

    const msg = h('div', { className: 'confirm-message' });
    msg.textContent = `Delete feed "${feed.name}"? This will soft-delete the feed (disable it). Choose "Delete + Purge" to also remove all articles.`;
    card.appendChild(msg);

    const btns = h('div', { className: 'confirm-btns' });

    const cancelBtn = h('button', { className: 'confirm-cancel' }, 'CANCEL');
    cancelBtn.addEventListener('click', () => overlay.remove());

    const softDeleteBtn = h('button', { className: 'confirm-ok' }, 'DELETE');
    softDeleteBtn.addEventListener('click', async () => {
      overlay.remove();
      try {
        await this.client.deleteFeed(feed.id, false);
        showToast(`Feed "${feed.name}" deleted`);
        await this.refresh();
      } catch {
        showToast('Failed to delete feed', true);
      }
    });

    const purgeBtn = h('button', {
      className: 'confirm-ok',
      style: 'background: #991b1b;',
    }, 'DELETE + PURGE');
    purgeBtn.addEventListener('click', async () => {
      overlay.remove();
      try {
        await this.client.deleteFeed(feed.id, true);
        showToast(`Feed "${feed.name}" purged`);
        await this.refresh();
      } catch {
        showToast('Failed to purge feed', true);
      }
    });

    btns.appendChild(cancelBtn);
    btns.appendChild(softDeleteBtn);
    btns.appendChild(purgeBtn);
    card.appendChild(btns);
    overlay.appendChild(card);
    document.body.appendChild(overlay);
  }
}
