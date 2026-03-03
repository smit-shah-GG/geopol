/**
 * ForecastPanel -- expandable forecast cards with progressive disclosure.
 *
 * Collapsed card: question + probability bar + country + age.
 * Expanded card: two-column layout with ensemble weights, calibration metadata,
 *   mini d3 scenario tree, evidence summaries, and "View Full Analysis" button.
 *
 * Key behaviors:
 * - Multiple cards expandable simultaneously (NOT accordion).
 * - Expanded state survives 60-second data refresh via diff-based DOM updates.
 * - Listens to search-results/search-cleared events from SearchBar.
 * - "View Full Analysis" dispatches forecast-selected event for ScenarioExplorer.
 *
 * Card rendering delegated to shared expandable-card utility to avoid DRY
 * violation across dashboard, globe drill-down, and forecasts queue screens.
 *
 * Fulfills FUX-01 (progressive disclosure) requirements.
 */

import { Panel } from './Panel';
import { replaceChildren, clearChildren } from '@/utils/dom-utils';
import { h } from '@/utils/dom-utils';
import { forecastClient } from '@/services/forecast-client';
import type { ForecastResponse, SearchResponse } from '@/types/api';
import {
  buildExpandableCard,
  buildExpandedContent,
  updateCardInPlace,
} from './expandable-card';

// ---------------------------------------------------------------------------
// ForecastPanel
// ---------------------------------------------------------------------------

export class ForecastPanel extends Panel {
  /** IDs of currently expanded cards. */
  private expandedIds = new Set<string>();

  /** Full forecast list from last update() call. */
  private allForecasts: ForecastResponse[] = [];

  /** Map of forecast_id -> card DOM element for diff-based updates. */
  private cardElements = new Map<string, HTMLElement>();

  /** Whether we're showing search results (vs. full list). */
  private isSearchActive = false;

  private readonly onSearchResults: EventListener;
  private readonly onSearchCleared: EventListener;

  constructor() {
    super({ id: 'forecasts', title: 'ACTIVE FORECASTS', showCount: true });

    this.onSearchResults = ((e: CustomEvent<SearchResponse>) => {
      this.isSearchActive = true;
      const forecasts = e.detail.results.map(r => r.forecast);
      this.renderFilteredForecasts(forecasts);
    }) as EventListener;

    this.onSearchCleared = (() => {
      this.isSearchActive = false;
      this.renderFilteredForecasts(this.allForecasts);
    }) as EventListener;

    // Listen on window so SearchBar events (which bubble) reach us
    window.addEventListener('search-results', this.onSearchResults);
    window.addEventListener('search-cleared', this.onSearchCleared);
  }

  /** Self-fetch via forecastClient. Used by RefreshScheduler. */
  public async refresh(): Promise<void> {
    this.showLoading();
    try {
      const forecasts = await forecastClient.getTopForecasts(10);
      this.applyBreakerBadge('forecast');
      this.update(forecasts);
    } catch (err: unknown) {
      if (this.isAbortError(err)) return;
      console.error('[ForecastPanel] refresh failed:', err);
      this.showError('Failed to load forecasts');
    }
  }

  /**
   * External data injection -- diff-based update preserving expanded state.
   * Called by dashboard-screen refresh scheduler every 60s.
   */
  public update(forecasts: ForecastResponse[]): void {
    this.allForecasts = forecasts;

    // If search is active, don't overwrite search results display
    if (this.isSearchActive) return;

    this.renderFilteredForecasts(forecasts);
  }

  public override destroy(): void {
    window.removeEventListener('search-results', this.onSearchResults);
    window.removeEventListener('search-cleared', this.onSearchCleared);
    super.destroy();
  }

  // -----------------------------------------------------------------------
  // Private: rendering
  // -----------------------------------------------------------------------

  private applyBreakerBadge(endpoint: 'forecast' | 'country' | 'health'): void {
    const state = forecastClient.getDataState(endpoint);
    this.setDataBadge(state.mode);
  }

  /**
   * Diff-based render: add new cards, remove stale ones, update existing ones
   * in-place without destroying DOM (preserving expanded state).
   */
  private renderFilteredForecasts(forecasts: ForecastResponse[]): void {
    // Sort by recency (newest first)
    const sorted = [...forecasts].sort(
      (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
    );
    this.setCount(sorted.length);

    if (sorted.length === 0) {
      this.cardElements.clear();
      replaceChildren(this.content,
        h('div', { className: 'empty-state' },
          this.isSearchActive ? 'No forecasts match your search' : 'No active forecasts',
        ),
      );
      return;
    }

    const newIds = new Set(sorted.map(f => f.forecast_id));
    const existingIds = new Set(this.cardElements.keys());

    // Remove cards no longer in list
    for (const id of existingIds) {
      if (!newIds.has(id)) {
        const el = this.cardElements.get(id);
        el?.remove();
        this.cardElements.delete(id);
        this.expandedIds.delete(id);
      }
    }

    // Build ordered list: update existing, create new
    const orderedCards: HTMLElement[] = [];
    for (const f of sorted) {
      const existing = this.cardElements.get(f.forecast_id);
      if (existing) {
        // Update data in-place without destroying DOM
        updateCardInPlace(existing, f, this.expandedIds);
        orderedCards.push(existing);
      } else {
        const card = buildExpandableCard(f, {
          expandedIds: this.expandedIds,
          onToggle: (id, cardEl) => this.toggleCard(id, cardEl),
        });
        this.cardElements.set(f.forecast_id, card);
        orderedCards.push(card);
      }
    }

    // Reorder DOM to match sorted order
    const frag = document.createDocumentFragment();
    for (const card of orderedCards) {
      frag.appendChild(card);
    }
    clearChildren(this.content);
    this.content.appendChild(frag);
  }

  // -----------------------------------------------------------------------
  // Expansion toggle
  // -----------------------------------------------------------------------

  private toggleCard(id: string, card: HTMLElement): void {
    if (this.expandedIds.has(id)) {
      // Collapse
      this.expandedIds.delete(id);
      card.classList.remove('expanded');
      const expandedContent = card.querySelector('.expanded-content');
      expandedContent?.remove();
    } else {
      // Expand -- get current data from dataset
      this.expandedIds.add(id);
      card.classList.add('expanded');
      const dataStr = card.dataset['forecastData'];
      if (!dataStr) return;
      const forecast = JSON.parse(dataStr) as ForecastResponse;
      card.appendChild(buildExpandedContent(forecast));
    }
  }
}
