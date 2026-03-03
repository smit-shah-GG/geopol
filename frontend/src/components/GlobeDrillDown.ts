/**
 * GlobeDrillDown -- Slide-in panel from the right edge showing country forecasts.
 *
 * Triggered by country-selected event on the globe. Shows:
 *   - Country name + flag + risk score with trend arrow
 *   - Paginated forecast list with expandable cards (same progressive disclosure
 *     as dashboard ForecastPanel and forecasts screen)
 *   - GDELT event sparkline placeholder (Phase 17 data)
 *   - "View Details" link to open CountryBriefPage modal
 *
 * Race condition prevention: requestToken increments on each open().
 * If a slower fetch resolves after a newer click, the stale token mismatch
 * causes the result to be silently discarded.
 */

import { h, clearChildren } from '@/utils/dom-utils';
import { forecastClient } from '@/services/forecast-client';
import { countryGeometry } from '@/services/country-geometry';
import {
  buildExpandableCard,
  buildExpandedContent,
  isoToFlag,
  severityClass,
} from '@/components/expandable-card';
import type { ForecastResponse, CountryRiskSummary, PaginatedResponse } from '@/types/api';

// ---------------------------------------------------------------------------
// Trend arrow display
// ---------------------------------------------------------------------------

function trendArrow(trend: 'rising' | 'stable' | 'falling'): string {
  switch (trend) {
    case 'rising': return '\u2191';   // up arrow
    case 'falling': return '\u2193';  // down arrow
    case 'stable': return '\u2192';   // right arrow
  }
}

function trendClass(trend: 'rising' | 'stable' | 'falling'): string {
  switch (trend) {
    case 'rising': return 'trend-rising';
    case 'falling': return 'trend-falling';
    case 'stable': return 'trend-stable';
  }
}

// ---------------------------------------------------------------------------
// GlobeDrillDown
// ---------------------------------------------------------------------------

export class GlobeDrillDown {
  private readonly panel: HTMLElement;
  private readonly headerName: HTMLElement;
  private readonly headerFlag: HTMLElement;
  private readonly headerRisk: HTMLElement;
  private readonly content: HTMLElement;
  private readonly sparklineSection: HTMLElement;

  private currentIso: string | null = null;
  private requestToken = 0;

  /** Track which forecast cards are expanded (survives re-renders for same country) */
  private readonly expandedIds = new Set<string>();

  /** Track card DOM elements for potential future diff updates */
  private readonly cardElements = new Map<string, HTMLElement>();

  constructor() {
    // Header components
    this.headerFlag = h('span', { className: 'drilldown-flag' });
    this.headerName = h('span', { className: 'drilldown-name' }, '--');
    this.headerRisk = h('span', { className: 'drilldown-risk' });

    const closeBtn = h('button', { className: 'drilldown-close' }, '\u00D7');
    closeBtn.addEventListener('click', () => this.close());

    const header = h('div', { className: 'drilldown-header' },
      h('div', { className: 'drilldown-header-left' },
        this.headerFlag,
        this.headerName,
        this.headerRisk,
      ),
      closeBtn,
    );

    // Scrollable content area
    this.content = h('div', { className: 'drilldown-content' });

    // Sparkline placeholder (Phase 17)
    this.sparklineSection = h('div', { className: 'drilldown-sparkline' },
      h('div', { className: 'drilldown-section-label' }, 'GDELT EVENTS'),
      h('div', { className: 'drilldown-sparkline-placeholder' },
        'Event data available in Phase 17',
      ),
    );

    this.panel = h('div', { className: 'globe-drilldown' },
      header,
      this.content,
      this.sparklineSection,
    );
  }

  /**
   * Open the drill-down for a country. Fetches forecasts + risk data.
   * Stale responses from prior clicks are discarded via requestToken.
   */
  async open(iso: string): Promise<void> {
    const token = ++this.requestToken;
    const upperIso = iso.toUpperCase();

    // Show panel immediately with loading state
    this.currentIso = upperIso;
    this.panel.classList.add('active');

    // Update header with known data
    const countryName = countryGeometry.getNameByIso(upperIso) ?? upperIso;
    const flag = isoToFlag(upperIso);
    this.headerFlag.textContent = flag;
    this.headerName.textContent = countryName;
    this.headerRisk.textContent = '';
    this.headerRisk.className = 'drilldown-risk';

    // Clear content area for new country
    if (this.currentIso !== upperIso || this.cardElements.size === 0) {
      clearChildren(this.content);
      this.expandedIds.clear();
      this.cardElements.clear();
      this.content.appendChild(
        h('div', { className: 'drilldown-loading' }, 'Loading forecasts...'),
      );
    }

    // Fetch data in parallel
    let forecasts: PaginatedResponse<ForecastResponse>;
    let risk: CountryRiskSummary | null;
    try {
      [forecasts, risk] = await Promise.all([
        forecastClient.getForecastsByCountry(upperIso, undefined, 20),
        forecastClient.getCountryRisk(upperIso),
      ]);
    } catch (err) {
      // If this request was superseded, silently discard
      if (token !== this.requestToken) return;
      console.error('[GlobeDrillDown] Data fetch failed:', err);
      clearChildren(this.content);
      this.content.appendChild(
        h('div', { className: 'drilldown-error' }, 'Failed to load data'),
      );
      return;
    }

    // Guard: discard stale response
    if (token !== this.requestToken) return;

    // Render -- wrapped in try/catch so a malformed API response (e.g.
    // circuit breaker cache returning wrong type) never leaves the panel
    // stuck on "Loading forecasts..."
    try {
      this.renderData(countryName, upperIso, forecasts, risk);
    } catch (err) {
      console.error('[GlobeDrillDown] Render failed:', err);
      clearChildren(this.content);
      this.content.appendChild(
        h('div', { className: 'drilldown-error' }, 'Failed to render data'),
      );
    }
  }

  /** Render fetched data into the panel. Separated for error boundary. */
  private renderData(
    countryName: string,
    iso: string,
    forecasts: PaginatedResponse<ForecastResponse>,
    risk: CountryRiskSummary | null,
  ): void {
    // Render risk score
    if (risk && typeof risk.risk_score === 'number') {
      const scorePct = risk.risk_score.toFixed(1);
      const sev = severityClass(risk.risk_score / 100);
      const arrow = trendArrow(risk.trend);
      const tc = trendClass(risk.trend);
      this.headerRisk.className = `drilldown-risk severity-${sev}`;
      this.headerRisk.innerHTML = '';
      this.headerRisk.appendChild(
        h('span', { className: 'drilldown-risk-value' }, `${scorePct}%`),
      );
      this.headerRisk.appendChild(
        h('span', { className: `drilldown-trend ${tc}` }, arrow),
      );
    }

    // Render forecast list
    clearChildren(this.content);
    this.cardElements.clear();

    const items = Array.isArray(forecasts?.items) ? forecasts.items : [];
    if (items.length === 0) {
      this.content.appendChild(
        h('div', { className: 'drilldown-empty' },
          `No active forecasts for ${countryName}`,
        ),
      );
    } else {
      const forecastsLabel = h('div', { className: 'drilldown-section-label' },
        `FORECASTS (${items.length})`,
      );
      this.content.appendChild(forecastsLabel);

      for (const f of items) {
        const card = buildExpandableCard(f, {
          expandedIds: this.expandedIds,
          onToggle: (id: string, cardEl: HTMLElement) => {
            if (this.expandedIds.has(id)) {
              this.expandedIds.delete(id);
              cardEl.classList.remove('expanded');
              const expanded = cardEl.querySelector('.expanded-content');
              if (expanded) expanded.remove();
            } else {
              this.expandedIds.add(id);
              cardEl.classList.add('expanded');
              cardEl.appendChild(buildExpandedContent(f));
            }
          },
        });
        this.cardElements.set(f.forecast_id, card);
        this.content.appendChild(card);
      }
    }

    // "View Details" link at bottom
    const viewDetailsLink = h('button', { className: 'drilldown-view-details' },
      'View Full Country Brief',
    );
    viewDetailsLink.addEventListener('click', () => {
      window.dispatchEvent(
        new CustomEvent('country-brief-requested', {
          detail: { iso },
          bubbles: true,
        }),
      );
    });
    this.content.appendChild(viewDetailsLink);
  }

  close(): void {
    this.panel.classList.remove('active');
    this.currentIso = null;
    this.expandedIds.clear();
    this.cardElements.clear();
  }

  isOpen(): boolean {
    return this.panel.classList.contains('active');
  }

  getCurrentIso(): string | null {
    return this.currentIso;
  }

  getElement(): HTMLElement {
    return this.panel;
  }

  destroy(): void {
    this.close();
  }
}
