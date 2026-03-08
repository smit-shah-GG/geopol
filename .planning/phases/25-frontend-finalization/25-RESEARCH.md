# Phase 25: Frontend Finalization - Research

**Researched:** 2026-03-08
**Domain:** Vanilla TypeScript frontend polish -- loading states, error handling, empty states, lazy loading, accessibility
**Confidence:** HIGH

## Summary

This phase is a polish pass across an existing vanilla TypeScript frontend (NO framework -- custom DOM via `h()` utility, Vite bundler, dark-only theme). The codebase has 14,025 lines across ~30 component files, 3 screens, and 2 CSS files. The architecture uses a `Panel` base class for dashboard panels and standalone modal classes for ScenarioExplorer, CountryBriefPage, and SettingsModal.

The current state has inconsistent loading/error/empty handling across panels -- some panels call `showLoading()` (radar sweep animation), some show bare text, some have no empty state. Error handling swallows failures with `console.error` and a generic `showError()` call without retry capability. No skeletons exist. Accessibility is minimal -- a few `aria-label` attributes on close buttons, Escape key handling on modals, but no focus trapping, no `prefers-reduced-motion`, and no systematic keyboard navigation.

The standard approach is: (1) replace the radar sweep loading indicator with shimmer skeletons per panel, (2) add a two-tier error boundary (initial failure = full error + retry, refresh failure = toast banner over stale data), (3) enhance empty states with contextual messages and icons, (4) lazy-load ScenarioExplorer and CalibrationPanel (CountryBriefPage is already lazy on globe), (5) add focus trapping to modals, ARIA labels, and `prefers-reduced-motion` media queries.

**Primary recommendation:** Build a shared `skeleton()` utility function and `PanelErrorBoundary` mixin that all Panel subclasses consume, rather than hand-coding loading/error/empty states per panel. The WM codebase has a reference shimmer implementation to adapt.

## Standard Stack

No new dependencies required. This is a pure CSS + vanilla TypeScript implementation phase.

### Core (Already Installed)
| Library | Version | Purpose | Relevance |
|---------|---------|---------|-----------|
| Vite | ^6.0.7 | Build tool | Manual chunks already configured for code splitting |
| TypeScript | ^5.7.2 | Type safety | Strict mode enabled, `noUncheckedIndexedAccess` |
| d3 | ^7.9.0 | Charts/trees | Already chunked separately in Vite config |
| deck.gl | ^9.2.6 | Map rendering | Already dynamically imported on globe route |
| maplibre-gl | ^5.16.0 | Map tiles | Already dynamically imported on globe route |

### No New Dependencies
| Problem | Solution | Why No Library |
|---------|----------|----------------|
| Shimmer skeletons | Pure CSS `@keyframes` + gradient | 15 lines of CSS, no runtime dependency needed |
| Focus trapping | Manual DOM query for focusable elements | ~30 lines of TS, no `focus-trap` library needed |
| Debounce | Already exists in `SearchBar.ts` | Extract to `utils/timing.ts` and reuse |
| Error boundaries | Custom Panel mixin | Framework-specific libraries don't apply (no React) |
| ARIA attributes | Direct `setAttribute` calls in `h()` | Already supported by the `DomProps` interface |

## Architecture Patterns

### Pattern 1: Skeleton Builder Utility

**What:** A `buildSkeleton(shape: SkeletonShape): HTMLElement` function that generates DOM-based shimmer placeholders matching each panel's real content layout.

**When to use:** Every Panel subclass calls this instead of the current `showLoading()` radar animation on initial mount.

**Why this pattern:** Skeletons must match the exact layout of real content to prevent CLS (Cumulative Layout Shift). A builder function takes a shape descriptor and generates the right block layout.

```typescript
// src/utils/skeleton.ts
type SkeletonShape =
  | { type: 'card-list'; count: number }    // ForecastPanel, MyForecastsPanel
  | { type: 'row-list'; count: number }     // RiskIndexPanel, SystemHealthPanel
  | { type: 'bar-pairs'; count: number }    // ComparisonPanel
  | { type: 'text-block'; lines: number }   // PolymarketPanel, SourcesPanel
  | { type: 'news-feed'; count: number }    // NewsFeedPanel

function buildSkeleton(shape: SkeletonShape): HTMLElement {
  const container = h('div', {
    className: 'skeleton-container',
    role: 'status',
    'aria-busy': 'true',
  },
    h('span', { className: 'sr-only' }, 'Loading...'),
  );
  // Build shape-specific skeleton blocks
  // ...
  return container;
}
```

**Source:** Pattern adapted from WM's `intel-skeleton` implementation in `CountryIntelModal.ts`.

### Pattern 2: Two-Tier Error Handling via Panel Base Class

**What:** Extend the `Panel` base class with `showErrorWithRetry(msg, retryFn)` and `showRefreshError(msg)` methods that implement the two-tier error model from CONTEXT.md.

**When to use:** All Panel subclasses use this instead of the current bare `showError(msg)`.

```typescript
// Add to Panel.ts
public showErrorWithRetry(message: string, retryFn: () => Promise<void>): void {
  replaceChildren(this.content,
    h('div', { className: 'panel-error-full' },
      h('div', { className: 'panel-error-icon' }, '\u26A0'),
      h('div', { className: 'panel-error-text' }, message),
      h('button', {
        className: 'panel-retry-btn',
        onclick: () => { void retryFn(); },
      }, 'Retry'),
    ),
  );
}

public showRefreshToast(message: string, severity: 'amber' | 'red'): void {
  // Insert toast banner at top of content, preserve existing children
  const existing = this.content.querySelector('.panel-refresh-toast');
  if (existing) existing.remove();
  const toast = h('div', {
    className: `panel-refresh-toast severity-${severity}`,
  }, message);
  this.content.insertBefore(toast, this.content.firstChild);
  setTimeout(() => toast.remove(), 10_000);
}
```

### Pattern 3: Focus Trap for Modals

**What:** A reusable `trapFocus(container: HTMLElement)` / `releaseFocus()` utility that cycles Tab/Shift+Tab within a modal and restores focus on close.

**When to use:** ScenarioExplorer, CountryBriefPage, SettingsModal.

```typescript
// src/utils/focus-trap.ts
const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), textarea:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function trapFocus(container: HTMLElement, triggerEl?: HTMLElement): () => void {
  const focusable = () => Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE));

  const handler = (e: KeyboardEvent) => {
    if (e.key !== 'Tab') return;
    const elements = focusable();
    if (elements.length === 0) return;
    const first = elements[0]!;
    const last = elements[elements.length - 1]!;
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  };

  container.addEventListener('keydown', handler);
  // Focus first focusable element
  const elements = focusable();
  if (elements.length > 0) elements[0]!.focus();

  return () => {
    container.removeEventListener('keydown', handler);
    if (triggerEl) triggerEl.focus();
  };
}
```

### Pattern 4: Lazy Loading via Dynamic Import

**What:** `import()` at the call site, not at the module level. Vite automatically code-splits dynamic imports into separate chunks.

**When to use:** ScenarioExplorer (already lazy on globe, needs lazy on dashboard), CalibrationPanel (CountryBriefPage tab).

**Current state:** Globe screen already does this correctly:
```typescript
const [{ DeckGLMap }, { LayerPillBar: LayerPillBarClass }] = await Promise.all([
  import('@/components/DeckGLMap'),
  import('@/components/LayerPillBar'),
]);
```

Dashboard screen does NOT -- it statically imports ScenarioExplorer:
```typescript
// Current: static import (in bundle)
import { ScenarioExplorer } from '@/components/ScenarioExplorer';

// Should be: dynamic import (separate chunk)
const { ScenarioExplorer } = await import('@/components/ScenarioExplorer');
```

### Recommended File Structure (new/modified files)
```
frontend/src/
├── utils/
│   ├── skeleton.ts          # NEW: skeleton builder utility
│   ├── focus-trap.ts        # NEW: modal focus trapping
│   ├── timing.ts            # NEW: extracted debounce + throttle
│   └── dom-utils.ts         # EXISTING: h(), fragment(), etc.
├── components/
│   ├── Panel.ts             # MODIFY: add showErrorWithRetry, showRefreshToast, showSkeleton
│   ├── ForecastPanel.ts     # MODIFY: skeleton shape, error handling, empty state
│   ├── RiskIndexPanel.ts    # MODIFY: skeleton shape, error handling, empty state
│   ├── ComparisonPanel.ts   # MODIFY: skeleton shape, error handling, empty state
│   ├── [all panels]         # MODIFY: same pattern for each
│   ├── ScenarioExplorer.ts  # MODIFY: focus trap, ARIA
│   ├── CountryBriefPage.ts  # MODIFY: focus trap, ARIA, CAMEO stub fix
│   ├── SettingsModal.ts     # MODIFY: focus trap
│   ├── GlobeDrillDown.ts    # MODIFY: sparkline wiring, error/empty states
│   └── LayerPillBar.ts      # MODIFY: ARIA labels on pills
├── screens/
│   ├── dashboard-screen.ts  # MODIFY: lazy import ScenarioExplorer
│   └── forecasts-screen.ts  # MODIFY: lazy import ScenarioExplorer
├── styles/
│   ├── main.css             # MODIFY: skeleton CSS, prefers-reduced-motion
│   └── panels.css           # MODIFY: error/empty state styles, toast styles
└── index.html               # MODIFY: lang attribute if missing
```

### Anti-Patterns to Avoid
- **Per-panel CSS skeletons:** Each panel defining its own skeleton CSS. Use shared classes with shape-specific sizing via the utility function.
- **innerHTML for skeleton rendering:** Skeletons must be built with `h()` to stay consistent with the codebase and avoid XSS vectors.
- **Retry loops with exponential backoff:** The CONTEXT.md specifies manual retry for initial load failures. The CircuitBreaker already handles automatic backoff for refresh cycles. Don't add a competing retry mechanism.
- **Focus trap via `inert` attribute:** Browser support for `inert` is good, but setting `inert` on `#app` while a modal is open would disable the modal's own parent if the modal is inside `#app`. Use the Tab cycling approach instead.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Shimmer animation | Custom JS requestAnimationFrame loop | Pure CSS `@keyframes` with `linear-gradient` | GPU-composited, no JS thread blocking, ~15 lines of CSS |
| Focus trap | Custom event delegation system | Direct `keydown` listener on modal container | Simple, debuggable, no framework overhead |
| Color contrast checking | Runtime contrast ratio calculator | Pre-computed during this phase, hardcoded fixes | Contrast ratios don't change at runtime |
| Debounce | New debounce implementation | Extract existing one from SearchBar.ts | Already tested, working pattern in codebase |
| ARIA live regions | Custom mutation observer announcer | Native `role="status"` + `aria-live="polite"` | Browser handles screen reader announcements |

## Common Pitfalls

### Pitfall 1: Layout Shift from Skeleton-to-Content Transition (CLS)
**What goes wrong:** Skeleton placeholder has different dimensions than real content. When data arrives, the layout jumps -- panels below shift, scroll position resets.
**Why it happens:** Skeletons use fixed heights that don't match the variable-height real content.
**How to avoid:** Skeleton dimensions must match the content's CSS box model. Use `min-height` on the panel content area to prevent collapse. The skeleton should use the same padding, gap, and border dimensions as the real content.
**Warning signs:** Visual "jump" when content loads. Lighthouse CLS score above 0.1.

### Pitfall 2: Error Toast Accumulation
**What goes wrong:** Multiple refresh failures stack toast banners, consuming the entire panel viewport.
**Why it happens:** Each failed refresh creates a new toast without removing the previous one.
**How to avoid:** The `showRefreshToast()` method must remove any existing toast before inserting a new one. The CONTEXT.md specifies: "if next refresh also fails, toast reappears" -- this means remove-then-reinsert, not accumulate.
**Warning signs:** Panel content scrolled out of view by stacked error banners.

### Pitfall 3: Focus Trap Escape on Dynamic Content
**What goes wrong:** Modal content updates (e.g., CountryBriefPage tab switch) add new focusable elements that the focus trap doesn't know about.
**Why it happens:** Focus trap caches the list of focusable elements at construction time.
**How to avoid:** Query focusable elements dynamically on each Tab keydown, not at trap construction time. The utility above does this correctly with `const elements = focusable()` inside the handler.
**Warning signs:** Tab key escapes the modal after content changes.

### Pitfall 4: Skeleton Flicker on Fast API Responses
**What goes wrong:** API responds in <100ms, causing skeleton to flash briefly before real content appears.
**Why it happens:** Skeleton appears immediately, content replaces it almost instantly.
**How to avoid:** CONTEXT.md specifies "Skeletons appear on initial page load only" -- on subsequent refreshes, the existing content stays visible. For initial load, the skeleton should have a minimum display time of ~300ms (CSS `animation-delay` or a setTimeout guard) to prevent sub-frame flicker.
**Warning signs:** Brief flash of gray blocks on fast connections.

### Pitfall 5: WCAG Contrast Failures on `--text-muted`
**What goes wrong:** `--text-muted` (#506070) has a contrast ratio of 2.99:1 against `--bg` (#0a0e14), which FAILS WCAG AA for all text sizes (needs 4.5:1 for normal, 3.0:1 for large).
**Why it happens:** Muted text was designed for aesthetic (faded look) without checking contrast math.
**How to avoid:** `--text-muted` is used in 114+ CSS rules across panels.css and main.css, plus 14 inline uses in component files. Two strategies: (A) Bump `--text-muted` to #6a7a8c (contrast ratio ~4.5:1 against #0a0e14) -- a subtle visual change that preserves the muted aesthetic while meeting AA. (B) Accept the aesthetic tradeoff and only fix interactive/informational text (leave decorative text as-is, since WCAG allows exceptions for "pure decoration"). Strategy A is recommended -- it's a single CSS variable change with global effect.
**Warning signs:** Lighthouse accessibility audit failures, automated a11y scanning tools flagging contrast.

### Pitfall 6: Dynamic Import Race on Dashboard ScenarioExplorer
**What goes wrong:** User clicks "View Full Analysis" before `ScenarioExplorer` has finished loading via dynamic import.
**Why it happens:** Dashboard currently imports ScenarioExplorer statically, so it's always available. Switching to dynamic import creates a window where the event fires but the handler isn't registered yet.
**How to avoid:** Register a synchronous event listener immediately that queues the event, then replays it after the dynamic import resolves. Or: import ScenarioExplorer eagerly during `mountDashboard` (it's ~532 lines / ~15KB gzipped -- not worth lazy-loading on dashboard since users always interact with it there). Only lazy-load it on the forecasts screen where it's rarely used.
**Warning signs:** "View Full Analysis" button does nothing on cold page load.

### Pitfall 7: `accent` on `bg_surface` Contrast Ratio
**What goes wrong:** `--accent` (#3a7bd5) on `--bg-surface` (#111620) has a contrast ratio of 4.29:1, which FAILS WCAG AA for normal text (needs 4.5:1). Passes for large text (3:1).
**How to avoid:** Bump `--accent` to #4080dd (ratio ~4.6:1) or add a subtle text-shadow/outline to accent-colored text on panel backgrounds. The accent color is used for active nav links, selected states, and emphasis text -- all normal-size text.
**Warning signs:** Active nav link text fails contrast checks.

## Code Examples

### Shimmer Skeleton CSS (verified against WM reference implementation)
```css
/* Shimmer animation -- left-to-right gradient sweep */
@keyframes skeleton-shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

.skeleton-block {
  background: linear-gradient(
    90deg,
    var(--bg-elevated) 25%,
    var(--border) 50%,
    var(--bg-elevated) 75%
  );
  background-size: 200% 100%;
  animation: skeleton-shimmer 1.5s ease-in-out infinite;
  border-radius: 2px;
}

/* Respect reduced motion preference */
@media (prefers-reduced-motion: reduce) {
  .skeleton-block {
    animation: none;
    opacity: 0.5;
  }
}

/* Shape variants */
.skeleton-line { height: 12px; margin-bottom: 8px; }
.skeleton-line.short { width: 60%; }
.skeleton-line.medium { width: 80%; }
.skeleton-card {
  height: 52px;
  margin-bottom: 4px;
  border-bottom: 1px solid var(--border-subtle);
}
.skeleton-bar { height: 20px; margin-bottom: 6px; }
```

### Error Boundary with Retry Button
```typescript
// Panel.ts addition
public showErrorWithRetry(message: string, retryFn: () => Promise<void>): void {
  replaceChildren(this.content,
    h('div', { className: 'panel-error-full' },
      h('div', { className: 'panel-error-icon' }, '\u26A0'),
      h('div', { className: 'panel-error-message' }, message),
      h('button', {
        className: 'panel-retry-btn',
        'aria-label': `Retry: ${message}`,
        onclick: () => {
          this.showSkeleton();  // Show skeleton during retry
          void retryFn();
        },
      }, 'Retry'),
    ),
  );
}
```

### Focus Trap Utility
```typescript
// src/utils/focus-trap.ts
const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'textarea:not([disabled])',
  'select:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(', ');

export function createFocusTrap(
  container: HTMLElement,
  triggerEl?: HTMLElement | null,
): () => void {
  const getFocusable = () =>
    Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR));

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key !== 'Tab') return;
    const elements = getFocusable();
    if (elements.length === 0) return;
    const first = elements[0]!;
    const last = elements[elements.length - 1]!;
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  };

  container.addEventListener('keydown', onKeyDown);
  // Auto-focus first focusable element
  requestAnimationFrame(() => {
    const els = getFocusable();
    if (els.length > 0) els[0]!.focus();
  });

  // Return cleanup function
  return () => {
    container.removeEventListener('keydown', onKeyDown);
    if (triggerEl) triggerEl.focus();
  };
}
```

### GlobeDrillDown Sparkline Wiring
```typescript
// Replace the Phase 17 placeholder with real event data
private async loadSparkline(iso: string): Promise<void> {
  try {
    const events = await forecastClient.getEventsByCountry(iso, { days: 30 });
    // Bucket events by day
    const buckets = new Map<string, number>();
    for (const ev of events) {
      const day = ev.event_date.slice(0, 10);
      buckets.set(day, (buckets.get(day) ?? 0) + 1);
    }
    // Render inline SVG sparkline
    this.renderSparkline(buckets);
  } catch {
    this.sparklineSection.querySelector('.drilldown-sparkline-placeholder')!
      .textContent = 'Event data unavailable';
  }
}
```

### prefers-reduced-motion Global Wrapper
```css
/* All non-essential animations gated behind motion preference */
@media (prefers-reduced-motion: reduce) {
  .skeleton-block,
  .panel-radar-sweep,
  .breaking-alert,
  .settings-modal,
  .live-stream-dot {
    animation: none !important;
  }

  ::view-transition-old(root),
  ::view-transition-new(root) {
    animation-duration: 0ms !important;
  }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Spinner/radar loading | Skeleton placeholders with shimmer | 2022+ standard | Reduces perceived load time, prevents CLS |
| Single error state | Two-tier (initial failure vs refresh failure) | - | Stale data stays visible during transient errors |
| Global error boundary | Per-component error containment | - | One panel failing doesn't crash the screen |
| `tabindex` management | `inert` attribute + focus trap | 2023+ (`inert` shipped in all browsers) | Note: `inert` is for background content, focus trap is for modal content |
| `aria-live` for updates | `role="status"` on loading containers | Stable ARIA 1.1 | Screen readers announce state changes |

## WCAG AA Contrast Audit Results

Pre-computed contrast ratios for the current theme. These MUST be fixed in this phase.

| Color Pair | Ratio | AA Normal (4.5:1) | AA Large (3.0:1) | Action |
|-----------|-------|-------------------|------------------|--------|
| `--text-primary` (#d0d8e0) on `--bg` (#0a0e14) | 13.43 | PASS | PASS | None |
| `--text-primary` on `--bg-surface` (#111620) | 12.58 | PASS | PASS | None |
| `--text-secondary` (#8090a4) on `--bg` | 5.93 | PASS | PASS | None |
| `--text-secondary` on `--bg-surface` | 5.56 | PASS | PASS | None |
| `--text-muted` (#506070) on `--bg` | 2.99 | **FAIL** | **FAIL** | Bump to #6a7a8c |
| `--text-muted` on `--bg-surface` | 2.80 | **FAIL** | **FAIL** | Bump to #6a7a8c |
| `--text-muted` on `--bg-elevated` (#1a2030) | 2.51 | **FAIL** | **FAIL** | Bump to #6a7a8c |
| `--accent` (#3a7bd5) on `--bg` | 4.58 | PASS | PASS | None |
| `--accent` on `--bg-surface` | 4.29 | **FAIL** | PASS | Bump to #4080dd |
| `--semantic-critical` (#e63946) on `--bg` | 4.64 | PASS | PASS | None |
| `--semantic-warning` (#e6a23c) on `--bg` | 8.84 | PASS | PASS | None |
| `--semantic-success` (#2ecc71) on `--bg` | 9.20 | PASS | PASS | None |

**Recommended fixes (2 CSS variable changes):**
- `--text-muted: #6a7a8c` (currently #506070) -- achieves ~4.5:1 on #0a0e14
- `--accent: #4080dd` (currently #3a7bd5) -- achieves ~4.6:1 on #111620

## Panel Inventory: Current State Audit

Exhaustive audit of every panel's current loading/error/empty handling.

| Panel | Loading State | Error State | Empty State | Needs Work |
|-------|--------------|-------------|-------------|------------|
| ForecastPanel | `showLoading()` radar | `showError(msg)` bare text | `'No active forecasts'` plain text | Skeleton, error+retry, enhanced empty |
| RiskIndexPanel | `showLoading()` radar | `showError(msg)` bare text | `'No country risk data available'` | Skeleton, error+retry, enhanced empty |
| ComparisonPanel | Constructor calls `showEmpty()` | None (swallowed in refresh) | `'No Polymarket comparisons yet...'` (decent) | Skeleton, error handling |
| SystemHealthPanel | `showLoading()` radar | `showError(msg)` bare text | None | Skeleton, error+retry, add empty |
| PolymarketPanel | Constructor calls `showLoading()` | None visible | `'No Polymarket data'` bare | Skeleton, error, enhanced empty |
| MyForecastsPanel | None | `showError(msg)` bare text | `'No submitted forecasts yet'` bare | Skeleton, error+retry, enhanced empty with CTA |
| NewsFeedPanel | None explicit | `showError(msg)` bare text | `'No articles match this filter'` | Skeleton, error+retry |
| EventTimelinePanel | None | `showError(msg)` bare text | `'No events...'` | Skeleton, error+retry, enhanced empty |
| SourcesPanel | None | `showError(msg)` bare text | `'No source data'` bare | Skeleton, error+retry |
| LiveStreamsPanel | None | `showError(msg)` for YouTube API | Existing empty state div | Error handling |
| GlobeHud | None (simple stat display) | None | N/A (always shows stats) | None needed |
| GlobeDrillDown | `'Loading forecasts...'` text | `'Failed to load data'` text | `'No active forecasts for {country}'` | Skeleton, error+retry |
| ScenarioExplorer | None (modal, instant) | None | N/A | Focus trap, ARIA |
| CountryBriefPage | Tab-specific loading | None | Tab-specific empty | Focus trap, ARIA, CAMEO stub fix |
| SettingsModal | None | None | `'No sources match'` filter | Focus trap |
| SearchBar | None | Error event dispatch | None | ARIA on input |
| SubmissionForm | Button loading state | Error text | N/A | ARIA labels |
| SubmissionQueue | Loading text | None visible | `'No submissions yet'` with icon | Enhanced empty |
| LayerPillBar | N/A | N/A | N/A | ARIA labels on pills |
| NavBar | N/A | N/A | N/A | ARIA labels on links |

## Stale Placeholder Cleanup Audit

Items from CONTEXT.md that need fixing:

### 1. GlobeDrillDown Sparkline (line 90-95)
**Current:** Hardcoded placeholder text `'Event data available in Phase 17'`
**Fix:** Wire to `forecastClient.getEventsByCountry()` and render inline SVG sparkline (event count per day, last 30 days)
**Complexity:** MEDIUM -- need to add an API method and sparkline renderer

### 2. CountryBriefPage CAMEO Trend Stub (line 869-870)
**Current:** `const trend = count > 2 ? 'rising' : count > 0 ? 'stable' : 'stable';`
**Fix:** Either compute actual trend from historical event data (comparing current 7-day count vs previous 7-day count) or remove the trend column entirely if historical data isn't available via API
**Complexity:** LOW if removing, MEDIUM if computing real trends

### 3. Globe Sizing
**Current:** Potential MapLibre container sizing issues on first load
**Fix:** Ensure the map container has explicit dimensions before DeckGLMap constructor runs; call `map.resize()` after mount
**Complexity:** LOW

## Integration Points with CircuitBreaker

The CONTEXT.md notes: "error toasts should integrate with existing CircuitBreaker state rather than duplicating failure detection."

**Current CircuitBreaker API surface:**
- `getDataState()` returns `{ mode: 'live' | 'cached' | 'unavailable', timestamp: number | null }`
- `execute(fetchFn, fallback)` handles stale-while-revalidate internally
- Background refresh failures are logged but NOT surfaced to the panel

**Integration approach:**
- The error toast should be driven by `BreakerDataState.mode` changes, not by catching errors in each panel's refresh function
- When `mode` transitions from `'live'` to `'cached'`, show amber toast ("Using cached data")
- When `mode` transitions to `'unavailable'`, show red toast ("Data temporarily unavailable")
- The Panel base class should poll `getDataState()` after each `execute()` call and conditionally render the toast

## Lazy Loading Analysis

### Components to Lazy-Load
| Component | Current Import | Lines | Estimated Size | Route(s) Used |
|-----------|---------------|-------|----------------|---------------|
| ScenarioExplorer | Static on dashboard, static on forecasts | 532 | ~15KB gzipped | All 3 screens |
| CountryBriefPage | Static on dashboard, dynamic on globe | 1594 | ~40KB gzipped | Dashboard (static), Globe (dynamic) |
| CalibrationPanel | CSS-only in panels.css | N/A | In CountryBriefPage tab | N/A (not a separate component) |

**Recommendation:**
- ScenarioExplorer: Keep static on dashboard (core interaction), lazy on forecasts screen (already done via forecasts-screen.ts creating a new `ScenarioExplorer()`)
- CountryBriefPage: Switch to dynamic import on dashboard-screen.ts (matches globe pattern). It's 1594 lines and only used when a user clicks a country.
- CalibrationPanel: Not a separate file -- it's a tab renderer inside CountryBriefPage. No separate lazy loading needed.

### Vite Chunk Configuration (already optimal)
```typescript
manualChunks: {
  deckgl: ['deck.gl', '@deck.gl/core', ...],  // ~1.9MB, globe-only
  maplibre: ['maplibre-gl'],                    // ~800KB, globe-only
  d3: ['d3'],                                   // ~250KB, used in ScenarioExplorer + CountryBriefPage
}
```

## Open Questions

1. **GlobeDrillDown sparkline API endpoint**
   - What we know: `forecastClient` has `getEventsByCountry()` but need to verify it returns date-bucketed data suitable for a sparkline
   - What's unclear: Whether the API returns raw events (need client-side bucketing) or pre-aggregated daily counts
   - Recommendation: Check the API response shape during planning; if raw events, add client-side bucketing (~10 lines)

2. **CountryBriefPage CAMEO trend: remove or fix?**
   - What we know: The stub uses `count > 2 ? 'rising' : 'stable'` which is meaningless
   - What's unclear: Whether historical CAMEO frequency data is available via API to compute real trends
   - Recommendation: Remove the trend column entirely rather than shipping fake data. Revisit if/when historical event frequency comparison is available.

3. **CalibrationPanel as lazy-load target**
   - What we know: CONTEXT.md lists CalibrationPanel for lazy loading, but it's CSS in panels.css + rendering code inside CountryBriefPage.ts (calibration tab)
   - What's unclear: Whether there's a standalone CalibrationPanel component that should exist
   - Recommendation: The calibration tab in CountryBriefPage is already lazy (tab content renders on click). No additional code splitting needed. The Lighthouse target (>80 performance) should be achievable through the other optimizations.

## Sources

### Primary (HIGH confidence)
- Codebase audit: Full read of all 30 component files, 3 screen files, 2 CSS files, Panel base class, CircuitBreaker, RefreshScheduler, Router, Vite config, tsconfig
- WM reference: `CountryIntelModal.ts` shimmer skeleton implementation, `main.css` skeleton-shimmer keyframes
- WCAG contrast computation: Python script computing relative luminance ratios per WCAG 2.2 formula against all theme color pairs

### Secondary (MEDIUM confidence)
- [MDN: Color Contrast](https://developer.mozilla.org/en-US/docs/Web/Accessibility/Guides/Understanding_WCAG/Perceivable/Color_contrast) -- WCAG AA requirements (4.5:1 normal, 3:1 large)
- [W3C: Understanding SC 1.4.3](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html) -- Official WCAG 2.2 understanding document
- [MDN: prefers-reduced-motion](https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/At-rules/@media/prefers-reduced-motion) -- Media feature reference

### Tertiary (LOW confidence)
- WebSearch: CSS skeleton animation patterns (multiple sources agree on gradient + keyframe approach, consistent with WM reference)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, all patterns derived from codebase audit
- Architecture: HIGH -- patterns match existing codebase conventions (h() utility, Panel class, CSS variables)
- Pitfalls: HIGH -- contrast ratios computed mathematically, CLS/focus-trap issues are well-documented in accessibility standards
- Stale placeholder audit: HIGH -- direct code reading of specific lines

**Research date:** 2026-03-08
**Valid until:** 2026-04-08 (stable -- no external dependencies changing)
