/**
 * SearchBar -- debounced full-text search with country/category dropdown filters.
 *
 * Lightweight inline component (NOT a Panel subclass). Dispatches CustomEvents
 * for search results, search cleared, and search errors consumed by ForecastPanel.
 *
 * Race condition prevention: each search creates a new AbortController, aborting
 * any in-flight request from the previous keystroke. AbortErrors are silently
 * swallowed -- they're expected when a newer search supersedes an older one.
 */

import { h } from '@/utils/dom-utils';
import { debounce } from '@/utils/timing';
import { forecastClient } from '@/services/forecast-client';
import type { CountryRiskSummary, SearchResponse } from '@/types/api';

// ---------------------------------------------------------------------------
// Category options (matches CAMEO quadrants from geopol.md)
// ---------------------------------------------------------------------------

const CATEGORIES: readonly { value: string; label: string }[] = [
  { value: '', label: 'All Categories' },
  { value: 'Conflict', label: 'Conflict' },
  { value: 'Cooperation', label: 'Cooperation' },
  { value: 'Diplomacy', label: 'Diplomacy' },
  { value: 'Economic', label: 'Economic' },
  { value: 'Military', label: 'Military' },
] as const;

// ---------------------------------------------------------------------------
// SearchBar
// ---------------------------------------------------------------------------

export class SearchBar {
  private readonly container: HTMLElement;
  private readonly input: HTMLInputElement;
  private readonly countrySelect: HTMLSelectElement;
  private readonly categorySelect: HTMLSelectElement;
  private searchController: AbortController | null = null;
  private readonly debouncedSearch: () => void;

  constructor() {
    this.input = document.createElement('input');
    this.input.type = 'text';
    this.input.placeholder = 'Search forecasts...';
    this.input.className = 'search-input';

    this.countrySelect = document.createElement('select');
    this.countrySelect.className = 'search-country-select';
    this.countrySelect.appendChild(this.makeOption('', 'All Countries'));

    this.categorySelect = document.createElement('select');
    this.categorySelect.className = 'search-category-select';
    for (const cat of CATEGORIES) {
      this.categorySelect.appendChild(this.makeOption(cat.value, cat.label));
    }

    this.container = h('div', { className: 'search-bar' },
      this.input,
      this.countrySelect,
      this.categorySelect,
    );

    // Debounced text search (300ms)
    this.debouncedSearch = debounce(() => {
      this.doSearch();
    }, 300);

    this.input.addEventListener('input', this.debouncedSearch);

    // Dropdown changes fire immediately (no debounce needed)
    this.countrySelect.addEventListener('change', () => {
      this.doSearch();
    });

    this.categorySelect.addEventListener('change', () => {
      this.doSearch();
    });
  }

  // -----------------------------------------------------------------------
  // Public API
  // -----------------------------------------------------------------------

  /** Returns the root container element for DOM insertion. */
  public getElement(): HTMLElement {
    return this.container;
  }

  /** Populate country dropdown from live country risk data. */
  public updateCountries(countries: CountryRiskSummary[]): void {
    // Preserve current selection
    const currentValue = this.countrySelect.value;

    // Clear existing options (keep "All Countries")
    while (this.countrySelect.options.length > 1) {
      this.countrySelect.remove(1);
    }

    // Sort alphabetically by ISO code, then append
    const sorted = [...countries].sort((a, b) =>
      a.iso_code.localeCompare(b.iso_code),
    );
    for (const c of sorted) {
      this.countrySelect.appendChild(
        this.makeOption(c.iso_code, `${c.iso_code} (${c.forecast_count})`),
      );
    }

    // Restore selection if still valid
    if (currentValue) {
      this.countrySelect.value = currentValue;
    }
  }

  /** Programmatically set country dropdown (for cross-column sync with Col 1). */
  public setCountry(iso: string | null): void {
    this.countrySelect.value = iso ?? '';
    this.doSearch();
  }

  /** Clean up listeners and abort pending requests. */
  public destroy(): void {
    this.searchController?.abort();
    this.searchController = null;
    this.input.removeEventListener('input', this.debouncedSearch);
  }

  // -----------------------------------------------------------------------
  // Private
  // -----------------------------------------------------------------------

  private async doSearch(): Promise<void> {
    // Abort any in-flight request
    this.searchController?.abort();

    const query = this.input.value.trim();
    const country = this.countrySelect.value;
    const category = this.categorySelect.value;

    // If everything is empty, signal "search cleared" so ForecastPanel reverts
    if (!query && !country && !category) {
      this.container.dispatchEvent(
        new CustomEvent<void>('search-cleared', { bubbles: true }),
      );
      return;
    }

    this.searchController = new AbortController();
    const { signal } = this.searchController;

    try {
      const response: SearchResponse = await forecastClient.search(
        query,
        {
          country: country || undefined,
          category: category || undefined,
          signal,
        },
      );
      this.container.dispatchEvent(
        new CustomEvent<SearchResponse>('search-results', {
          detail: response,
          bubbles: true,
        }),
      );
    } catch (err: unknown) {
      // AbortError is expected when a newer search supersedes this one
      if (err instanceof DOMException && err.name === 'AbortError') return;
      console.error('[SearchBar] search failed:', err);
      this.container.dispatchEvent(
        new CustomEvent<{ error: unknown }>('search-error', {
          detail: { error: err },
          bubbles: true,
        }),
      );
    }
  }

  private makeOption(value: string, label: string): HTMLOptionElement {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = label;
    return opt;
  }
}
