# Phase 16: Globe & Forecasts Screens - Research

**Researched:** 2026-03-03
**Domain:** deck.gl globe full-viewport interaction, slide-in overlay panel, layer toggle UI, question submission workflow, vanilla TypeScript DOM patterns
**Confidence:** HIGH

## Summary

Phase 16 completes the remaining two of three URL-routed screens. The Globe screen (`/globe`) transforms the current placeholder into a full-viewport deck.gl globe with contextual overlays: a minimal HUD showing aggregate stats, a floating pill bar for layer toggles, and a right-edge slide-in panel triggered by country click. The Forecasts screen (`/forecasts`) implements a two-column layout with a question submission form on the left and a scrollable queue of past/active submissions on the right.

The existing codebase provides nearly all the building blocks. `DeckGLMap` (Phase 12) already renders 5 layers, handles `country-selected` CustomEvents, and manages layer toggle state. `ForecastServiceClient` (Phase 12, expanded in Phase 14-15) already has `submitQuestion()`, `confirmSubmission()`, `getRequests()`, `getForecastsByCountry()`, and `getCountries()`. `ForecastPanel.buildExpandedContent()` (Phase 15) implements the progressive disclosure pattern that must be replicated consistently in the globe drill-down and forecasts queue. The Phase 15 router, screen lifecycle pattern, and `RefreshScheduler` provide the scaffolding.

The primary technical challenges are: (1) restructuring `DeckGLMap` from a dashboard-embedded component into a full-viewport primary display with overlapping overlay panels, (2) implementing the slide-in country drill-down panel with the same progressive disclosure as `ForecastPanel`, (3) redesigning the existing checkbox-list layer toggles as a floating pill bar with fade transitions, (4) building the two-phase submit/confirm form with inline state transformation, and (5) wiring real-time queue status polling.

**Primary recommendation:** Extract the expandable forecast card pattern from `ForecastPanel` into a shared utility (`buildExpandableCard` + CSS classes) so dashboard, globe drill-down, and forecasts queue all share identical progressive disclosure rendering. Build the globe screen around the existing `DeckGLMap` class by wrapping it with overlay DOM positioned absolutely over the full-viewport map container. Build the forecasts screen as a standalone module using the existing `forecastClient` submission methods.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| deck.gl | ^9.2.6 | WebGL globe rendering, 5 analytic layers | Already in `package.json`. Existing `DeckGLMap` class wraps MapboxOverlay + maplibre-gl. |
| maplibre-gl | ^5.16.0 | Basemap tile rendering, `flyTo` camera animation | Already in `package.json`. Provides `map.flyTo()` for country click camera animation. |
| d3 | ^7.9.0 | Mini scenario tree rendering in expanded cards | Already in `package.json`. Reuse `ForecastPanel.renderMiniTree()` pattern. |
| CSS `transition` | N/A | Slide-in panel animation (right: -400px -> 0), layer fade (~200ms opacity) | Pure CSS. WM uses `transition: right 0.28s ease` for deep-dive panel. |
| History API | N/A | URL routing (already implemented in Phase 15 `Router`) | Already in place. No changes needed. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| ForecastServiceClient | N/A (existing) | All API calls: countries, forecasts by country, submit, confirm, requests | Already built. Covers every Phase 16 data need. |
| CountryGeometryService | N/A (existing) | ISO code -> name lookup, centroid -> camera target | Already loaded in globe screen. Used for drill-down panel header. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| CSS slide-in panel | dialog/popover API | Dialog is block-level, doesn't overlay the globe naturally. CSS transform/right transition is simpler and matches WM reference. |
| CSS opacity for layer fade | deck.gl layer opacity prop | deck.gl `opacity` on layers works but requires `rebuildLayers()` on every frame during transition. CSS opacity on the canvas overlay is cheaper but affects all layers. Use deck.gl opacity per-layer for independent control. |
| Polling for queue status | SSE/WebSocket | Explicitly deferred per REQUIREMENTS.md out-of-scope. Polling via RefreshScheduler at 15s interval is sufficient for 2-3 min processing times. |

**Installation:**
```bash
# No new packages required. All dependencies already in package.json.
```

## Architecture Patterns

### Recommended Project Structure
```
frontend/src/
├── screens/
│   ├── globe-screen.ts       # MAJOR: full-viewport globe + HUD + drill-down + layer pills
│   └── forecasts-screen.ts   # MAJOR: two-column submission form + queue display
├── components/
│   ├── DeckGLMap.ts           # MODIFY: add flyToCountry(), restructure toggle UI, add public API for drill-down
│   ├── GlobeDrillDown.ts      # NEW: slide-in panel for country forecasts + risk + sparkline
│   ├── GlobeHud.ts            # NEW: minimal stats overlay (forecast count, countries, last update)
│   ├── LayerPillBar.ts        # NEW: floating horizontal pill toggles (replaces checkbox panel)
│   ├── SubmissionForm.ts      # NEW: question input + inline LLM confirmation transform
│   ├── SubmissionQueue.ts     # NEW: scrollable queue with status badges + expandable completed forecasts
│   ├── ForecastPanel.ts       # REFACTOR: extract shared card rendering to expandable-card utility
│   └── expandable-card.ts     # NEW: shared buildExpandableCard() + buildMiniTree() + buildEvidencePreview()
├── styles/
│   ├── main.css               # ADD: globe-screen, slide-in panel, pill bar, submission form styles
│   └── panels.css             # MODIFY: shared expandable card styles (extracted from forecast-card)
└── main.ts                    # NO CHANGE (routes already registered)
```

### Pattern 1: Full-Viewport Globe with Overlay Stack
**What:** The globe fills the entire screen container. All UI elements (HUD, layer pills, drill-down panel) are positioned absolutely over the map using z-index layering.
**When to use:** When the map IS the screen, not a panel within a layout.
**Example:**
```typescript
// globe-screen.ts structure
async function mountGlobe(container: HTMLElement, ctx: GeoPolAppContext): Promise<void> {
  const wrapper = h('div', { className: 'globe-screen' });

  // Full-viewport map container
  const mapContainer = h('div', { className: 'globe-map-container' });

  // HUD overlay (top-left corner)
  const hud = new GlobeHud();

  // Layer pill bar (bottom-center, floats over globe)
  const pillBar = new LayerPillBar(deckMap);

  // Drill-down panel (right edge, hidden by default)
  const drillDown = new GlobeDrillDown();

  wrapper.append(mapContainer, hud.getElement(), pillBar.getElement(), drillDown.getElement());
  container.appendChild(wrapper);

  // DeckGLMap fills mapContainer
  await countryGeometry.load();
  const { DeckGLMap } = await import('@/components/DeckGLMap');
  deckMap = new DeckGLMap(mapContainer);

  // Wire country click -> fly-to + drill-down open
  window.addEventListener('country-selected', (e: CustomEvent) => {
    const { iso } = e.detail;
    deckMap.flyToCountry(iso);
    drillDown.open(iso);
  });
}
```

### Pattern 2: Slide-In Panel (WM CountryDeepDivePanel Reference)
**What:** A fixed-position panel slides in from the right edge on country click. Uses CSS `right` transition. Contains country-specific data fetched on open.
**When to use:** Contextual drill-down without losing the globe viewport.
**Reference:** WM's `CountryDeepDivePanel` uses `position: fixed; right: -460px; transition: right 0.28s ease;` and toggles `.active` class to set `right: 0`.
**Example:**
```typescript
// GlobeDrillDown.ts
class GlobeDrillDown {
  private panel: HTMLElement;
  private content: HTMLElement;
  private currentIso: string | null = null;

  constructor() {
    this.panel = h('div', { className: 'globe-drilldown' });
    // Close button, content container
    const closeBtn = h('button', { className: 'drilldown-close' }, '\u00D7');
    closeBtn.addEventListener('click', () => this.close());
    this.content = h('div', { className: 'drilldown-content' });
    this.panel.append(closeBtn, this.content);
  }

  async open(iso: string): Promise<void> {
    this.currentIso = iso;
    this.panel.classList.add('active');
    // Fetch country data
    const [forecasts, risk] = await Promise.all([
      forecastClient.getForecastsByCountry(iso),
      forecastClient.getCountryRisk(iso),
    ]);
    this.renderContent(iso, forecasts, risk);
  }

  close(): void {
    this.panel.classList.remove('active');
    this.currentIso = null;
  }
}
```
**CSS:**
```css
.globe-drilldown {
  position: absolute;
  top: 0;
  right: -420px;
  width: 400px;
  height: 100%;
  z-index: 20;
  background: var(--panel-bg);
  border-left: 1px solid var(--border);
  box-shadow: -4px 0 16px rgba(0, 0, 0, 0.3);
  transition: right 0.28s ease;
  overflow-y: auto;
}
.globe-drilldown.active {
  right: 0;
}
```

### Pattern 3: Floating Pill Bar for Layer Toggles
**What:** Horizontal row of pill-shaped toggles floating over the globe. Each pill shows layer name, toggles opacity on click.
**When to use:** Compact layer control without consuming sidebar space.
**Reference:** Phase context specifies "floating pill bar -- compact horizontal bar of toggle pills." Default ON: choropleth + markers. Default OFF: arcs, heatmap, scenario zones.
**Example:**
```typescript
// LayerPillBar.ts
class LayerPillBar {
  private bar: HTMLElement;
  private states: Record<string, boolean>;

  constructor(private map: DeckGLMap) {
    this.states = {
      ForecastRiskChoropleth: true,
      ActiveForecastMarkers: true,
      KnowledgeGraphArcs: false,    // default OFF
      GDELTEventHeatmap: false,     // default OFF
      ScenarioZones: false,         // default OFF
    };
    this.bar = this.buildBar();
  }

  private buildBar(): HTMLElement {
    const bar = h('div', { className: 'layer-pill-bar' });
    for (const [id, label] of LAYER_PILLS) {
      const pill = h('button', {
        className: `layer-pill ${this.states[id] ? 'active' : ''}`,
        dataset: { layer: id },
      }, label);
      pill.addEventListener('click', () => this.toggle(id));
      bar.appendChild(pill);
    }
    return bar;
  }
}
```
**CSS:**
```css
.layer-pill-bar {
  position: absolute;
  bottom: 24px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 15;
  display: flex;
  gap: 6px;
  padding: 6px 12px;
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: 20px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
}
.layer-pill {
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 10px;
  border: 1px solid var(--border-subtle);
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
  transition: all 0.2s ease;
}
.layer-pill.active {
  background: var(--accent-muted);
  color: var(--accent);
  border-color: var(--accent);
}
```

### Pattern 4: Two-Phase Submit/Confirm Form with Inline Transform
**What:** Submission form that replaces itself inline with parsed fields (country, horizon, category) + Edit/Confirm buttons. No modal, no navigation.
**When to use:** When the user must review LLM interpretation before committing API budget.
**Example:**
```typescript
// SubmissionForm.ts
class SubmissionForm {
  private formState: 'input' | 'confirm' = 'input';
  private parsedResponse: ParsedQuestionResponse | null = null;

  private renderInput(): HTMLElement {
    const form = h('form', { className: 'submission-form' });
    const textarea = h('textarea', { placeholder: 'e.g. Will Iran retaliate...' });
    const submitBtn = h('button', { type: 'submit' }, 'Analyze Question');
    submitBtn.addEventListener('click', async (e) => {
      e.preventDefault();
      submitBtn.disabled = true;
      this.parsedResponse = await forecastClient.submitQuestion(textarea.value);
      this.formState = 'confirm';
      this.render(); // Replaces form in-place
    });
    form.append(textarea, submitBtn);
    return form;
  }

  private renderConfirmation(): HTMLElement {
    const parsed = this.parsedResponse!;
    // Show parsed fields with Edit/Confirm buttons
    const card = h('div', { className: 'submission-confirm' });
    // Country, horizon, category fields
    // Edit button -> revert to 'input' state
    // Confirm button -> call confirmSubmission()
    return card;
  }
}
```

### Pattern 5: Shared Expandable Card Utility
**What:** Extract the expandable forecast card rendering (collapsed header + expanded content with mini tree, ensemble weights, evidence, "View Full Analysis") into a reusable function.
**When to use:** Three screens (dashboard, globe drill-down, forecasts queue) must render identical progressive disclosure.
**Reference:** Phase context explicitly states "Consistent expand pattern across all three screens."
**Example:**
```typescript
// expandable-card.ts
export function buildExpandableCard(
  forecast: ForecastResponse,
  options: { expandedIds: Set<string>; onToggle: (id: string) => void }
): HTMLElement {
  // Collapsed: question + probability bar + country + age
  // Expanded: two-column (ensemble/calibration left, mini tree/evidence right)
  // "View Full Analysis" dispatches forecast-selected
  // IDENTICAL to ForecastPanel.buildCard() + buildExpandedContent()
}
```

### Anti-Patterns to Avoid
- **Duplicating ForecastPanel card rendering:** Three screens need identical expandable cards. Copy-pasting the 300-line buildCard/buildExpandedContent/renderMiniTree will create instant DRY violation. Extract shared utility.
- **Globe sidebar instead of overlay:** The drill-down panel must overlap the globe, not push it aside. Using flexbox to give the panel its own column defeats the full-viewport requirement.
- **Auto-rotating globe:** Context explicitly states "Static camera -- no auto-rotate on idle." Don't add idle animations.
- **Fake progress bars for submission processing:** Context explicitly states "No fake progress bar." Show elapsed time counter + status badge instead.
- **Modal for LLM confirmation:** Context explicitly states "no modal, no navigation." The form transforms inline.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Camera fly-to animation | Custom lerp/tween on viewState | `maplibreMap.flyTo({ center, zoom, duration })` | maplibre-gl's flyTo handles easing, zoom interpolation, and bounds respect. WM uses this exact API. |
| Country name from ISO | Hardcoded lookup table | `countryGeometry.getNameByIso(iso)` | Already built in Phase 12. 200+ country mappings from Natural Earth GeoJSON. |
| Centroid for fly-to target | Manual lat/lon computation | `countryGeometry.getCentroid(iso)` | Already built. Uses LABEL_X/LABEL_Y from Natural Earth for accurate centroids. |
| Submission status polling | Custom setInterval | `RefreshScheduler.registerAll()` | Already built in Phase 15. Handles visibility throttling, stale refresh flushing. |
| API request deduplication | Manual in-flight tracking | `ForecastServiceClient.dedup()` | Already built. Prevents duplicate fetches for country data during drill-down. |
| Circuit breaker on API | Custom error counting | `ForecastServiceClient` breakers | Already built. Returns fallback data when API is down. |

**Key insight:** Phase 16 is primarily a UI/layout phase. Nearly all data-fetching, API contracts, and rendering primitives already exist from Phases 12-15. The work is wiring existing components into new screen layouts with new interaction patterns.

## Common Pitfalls

### Pitfall 1: deck.gl Layer Opacity Transition Flicker
**What goes wrong:** Toggling a deck.gl layer causes a visual pop (instant appear/disappear) instead of a smooth fade.
**Why it happens:** deck.gl layers don't support CSS transitions. Setting `visible: false` removes the layer immediately. Setting `opacity: 0` in `getFillColor` alpha requires a `rebuildLayers()` call which re-renders all layers.
**How to avoid:** Use `opacity` prop on the layer itself (deck.gl ^9.x supports per-layer opacity). Animate from 0 to 1 via `requestAnimationFrame` over ~200ms: `layer.setProps({ opacity: lerp(0, 1, t) })`. Or, more simply, do the `rebuildLayers()` approach (which already works) and accept the instant toggle -- 200ms CSS fade on the entire canvas wrapper is imperceptible at this scale.
**Warning signs:** User reports "flashing" when toggling layers.

### Pitfall 2: Globe Drill-Down Panel Steals Map Click Events
**What goes wrong:** Clicks on the drill-down panel propagate through to the map underneath, causing unwanted country selections or map pans.
**Why it happens:** The panel is absolutely positioned over the map. deck.gl's overlay processes clicks on the canvas first.
**How to avoid:** `e.stopPropagation()` on the drill-down panel's click handler. The panel DOM is a sibling of the map container, not a child, so click events on the panel won't reach the canvas. Verify pointer-events CSS is set correctly (`pointer-events: auto` on the panel, canvas handles its own events via MapboxOverlay).
**Warning signs:** Clicking forecast cards in the drill-down triggers a new country selection.

### Pitfall 3: Concurrent Drill-Down Data Fetches on Rapid Country Clicks
**What goes wrong:** User clicks Ukraine, then immediately clicks Syria. Both fetches resolve, and Ukraine's data renders after Syria's because it resolved later.
**Why it happens:** No AbortController cancellation on the previous fetch.
**How to avoid:** Track a `requestToken` (incrementing counter). Before rendering, compare the token from the fetch with the current token. If they don't match, discard the result. WM's `CountryIntelManager` uses `briefRequestToken` for exactly this pattern. Additionally, use `AbortController` to cancel the in-flight fetch when a new country is selected.
**Warning signs:** Drill-down panel briefly shows data from a previously clicked country.

### Pitfall 4: DeckGLMap Layer Default States Diverge from Phase 16 Requirements
**What goes wrong:** Phase 12 sets ALL layers to `true` by default. Phase 16 requires choropleth + markers ON, arcs + heatmap + scenario zones OFF.
**Why it happens:** The `layerVisible` defaults in `DeckGLMap` constructor were set for a demo where all layers should be visible.
**How to avoid:** The globe screen must call a `DeckGLMap.setLayerDefaults()` or pass initial visibility state. Don't modify the existing defaults globally (dashboard may want them). Pass configuration at construction time or add a `setLayerVisible(layerId, visible)` public method.
**Warning signs:** Globe loads with all layers ON, cluttering the view.

### Pitfall 5: Submission Form Not Clearing After Successful Confirm
**What goes wrong:** After confirming a submission, the form stays in "confirm" state showing the parsed fields instead of resetting to the input state.
**Why it happens:** The form state machine doesn't handle the `confirm -> input` transition.
**How to avoid:** On successful `confirmSubmission()`, reset form state to 'input', clear the textarea, and trigger a queue refresh to show the new pending item in the right column.
**Warning signs:** User has to refresh the page to submit another question.

### Pitfall 6: maplibre flyTo target outside viewport causes jarring animation
**What goes wrong:** Camera fly-to for countries near the antimeridian (e.g., Fiji, Kiribati) takes the long way around the globe.
**Why it happens:** maplibre-gl's `flyTo` doesn't always compute the shortest great-circle path.
**How to avoid:** Use the country centroid from `countryGeometry.getCentroid()` which provides Natural Earth label coordinates (correct side of antimeridian). For zoom level, use a sensible default (zoom: 4-5 for most countries) rather than trying to fit the country bbox. WM uses `fitBounds` with padding for accurate framing, but simple `flyTo` with centroid + fixed zoom is sufficient for Phase 16.
**Warning signs:** Camera spins 270 degrees to reach a Pacific island.

## Code Examples

### Camera Fly-To on Country Click
```typescript
// Source: maplibre-gl API, verified against WM's DeckGLMap.setCenter() / fitCountry()
// Add to DeckGLMap class:
public flyToCountry(iso: string): void {
  if (!this.map) return;
  const centroid = countryGeometry.getCentroid(iso);
  if (!centroid) return;
  this.map.flyTo({
    center: [centroid[0], centroid[1]],
    zoom: 4.5,
    duration: 800,
    essential: true,  // Not affected by prefers-reduced-motion
  });
}
```

### Slide-In Panel CSS (from WM reference)
```css
/* Source: WM country-deep-dive.css, adapted for Geopol */
.globe-drilldown {
  position: absolute;
  top: 0;
  right: -420px;
  width: 400px;
  height: 100%;
  z-index: 20;
  background: var(--panel-bg);
  border-left: 1px solid var(--border);
  box-shadow: -4px 0 16px rgba(0, 0, 0, 0.3);
  transition: right 0.28s ease;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}
.globe-drilldown.active {
  right: 0;
}
.globe-drilldown .drilldown-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-md) var(--space-lg);
  border-bottom: 1px solid var(--border-subtle);
  flex-shrink: 0;
}
.globe-drilldown .drilldown-content {
  flex: 1;
  overflow-y: auto;
  padding: var(--space-md);
}
```

### Layer Pill Toggle with Fade
```typescript
// Layer toggle with opacity transition via deck.gl layer props
private toggle(layerId: string): void {
  const current = this.states[layerId];
  this.states[layerId] = !current;

  // Update pill button visual state
  const pill = this.bar.querySelector(`[data-layer="${layerId}"]`);
  pill?.classList.toggle('active', this.states[layerId]);

  // Update DeckGLMap layer visibility
  this.map.setLayerVisible(layerId as LayerId, this.states[layerId]);
}
```

### Submission Form State Machine
```typescript
// Three states: 'input' -> 'parsing' -> 'confirm' -> (back to 'input' or forward to 'submitted')
type FormState = 'input' | 'parsing' | 'confirm';

private async handleSubmit(question: string): Promise<void> {
  this.setState('parsing');
  try {
    this.parsedResponse = await forecastClient.submitQuestion(question);
    this.setState('confirm');
  } catch (err) {
    this.showError('Failed to parse question');
    this.setState('input');
  }
}

private async handleConfirm(): Promise<void> {
  try {
    await forecastClient.confirmSubmission(this.parsedResponse!.request_id);
    this.setState('input');  // Reset form
    this.parsedResponse = null;
    // Trigger queue refresh to show new pending item
    window.dispatchEvent(new CustomEvent('submission-confirmed'));
  } catch (err) {
    this.showError('Failed to confirm submission');
  }
}

private handleEdit(): void {
  // Revert to input with the original question pre-filled
  this.setState('input');
}
```

### Elapsed Time Counter for Processing Status
```typescript
// Lightweight elapsed time display (no fake progress bar)
private startElapsedTimer(cardEl: HTMLElement, startTime: string): number {
  const timerEl = cardEl.querySelector('.elapsed-time');
  if (!timerEl) return 0;

  const startMs = new Date(startTime).getTime();
  return window.setInterval(() => {
    const elapsed = Date.now() - startMs;
    const mins = Math.floor(elapsed / 60_000);
    const secs = Math.floor((elapsed % 60_000) / 1000);
    timerEl.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
  }, 1000);
}
```

### Globe HUD Overlay
```typescript
// Minimal stats overlay -- top-left corner of globe screen
class GlobeHud {
  private element: HTMLElement;
  private countEl: HTMLElement;
  private countriesEl: HTMLElement;
  private updateEl: HTMLElement;

  constructor() {
    this.countEl = h('span', { className: 'hud-value' }, '0');
    this.countriesEl = h('span', { className: 'hud-value' }, '0');
    this.updateEl = h('span', { className: 'hud-value' }, '--');

    this.element = h('div', { className: 'globe-hud' },
      h('div', { className: 'hud-item' },
        h('span', { className: 'hud-label' }, 'FORECASTS'),
        this.countEl,
      ),
      h('div', { className: 'hud-item' },
        h('span', { className: 'hud-label' }, 'COUNTRIES'),
        this.countriesEl,
      ),
      h('div', { className: 'hud-item' },
        h('span', { className: 'hud-label' }, 'UPDATED'),
        this.updateEl,
      ),
    );
  }

  update(countries: CountryRiskSummary[]): void {
    const totalForecasts = countries.reduce((sum, c) => sum + c.forecast_count, 0);
    this.countEl.textContent = String(totalForecasts);
    this.countriesEl.textContent = String(countries.length);
    if (countries.length > 0) {
      const latest = countries.reduce((a, b) =>
        new Date(a.last_updated) > new Date(b.last_updated) ? a : b
      );
      this.updateEl.textContent = relativeTime(latest.last_updated);
    }
  }

  getElement(): HTMLElement {
    return this.element;
  }
}
```

## Existing Code Inventory

Critical existing assets that Phase 16 builds upon:

### Components to Modify
| File | Current State | Phase 16 Change |
|------|--------------|-----------------|
| `DeckGLMap.ts` | 660 lines, 5 layers, checkbox toggles, `country-selected` event | Add `flyToCountry()`, `setLayerVisible()`, `setLayerDefaults()`. Remove built-in toggle panel (replaced by external LayerPillBar). |
| `globe-screen.ts` | 61-line placeholder: loads DeckGLMap, label "Globe Screen" | Full rewrite: full-viewport wrapper, HUD overlay, pill bar, drill-down panel, data loading, event wiring, RefreshScheduler |
| `forecasts-screen.ts` | 18-line placeholder: text "Coming in Phase 16" | Full rewrite: two-column layout, SubmissionForm, SubmissionQueue, event wiring, RefreshScheduler |
| `ForecastPanel.ts` | 597 lines: expandable cards, mini tree, evidence, search integration | Extract shared card rendering into `expandable-card.ts`. ForecastPanel becomes a thin wrapper calling the shared utility. |

### Components to Reuse As-Is
| File | How Phase 16 Uses It |
|------|---------------------|
| `ScenarioExplorer.ts` | "View Full Analysis" from any expandable card opens this modal. No changes needed. |
| `CountryBriefPage.ts` | "View Details" link at bottom of drill-down opens this modal. No changes needed. |
| `forecast-client.ts` | All API calls: `getCountries()`, `getForecastsByCountry()`, `getCountryRisk()`, `submitQuestion()`, `confirmSubmission()`, `getRequests()`. No changes needed. |
| `country-geometry.ts` | `getCentroid()` for fly-to, `getNameByIso()` for drill-down header. No changes needed. |
| `router.ts` | Routes already registered in `main.ts`. No changes needed. |
| `Panel.ts` | Not used directly in globe/forecasts screens (they use standalone components, not Panel subclasses). |

### API Contracts (All Delivered in Phase 14)
| Endpoint | Response Type | Usage |
|----------|--------------|-------|
| `GET /countries` | `CountryRiskSummary[]` | Choropleth risk scores, HUD stats |
| `GET /countries/{iso}` | `CountryRiskSummary` | Drill-down panel header (risk score + trend) |
| `GET /forecasts/country/{iso}` | `PaginatedResponse<ForecastResponse>` | Drill-down panel forecast list |
| `POST /forecasts/submit` | `ParsedQuestionResponse` | Submission form LLM parse |
| `POST /forecasts/submit/{id}/confirm` | `ConfirmSubmissionResponse` | Confirm after review |
| `GET /forecasts/requests` | `ForecastRequestStatus[]` | Queue display |

### Event Bus (CustomEvent on window)
| Event | Dispatched By | Consumed By |
|-------|--------------|-------------|
| `country-selected` | `DeckGLMap.handleClick()` | Globe screen: fly-to + drill-down open |
| `forecast-selected` | Expandable card "View Full Analysis" | `ScenarioExplorer.open()` |
| `submission-confirmed` | SubmissionForm | SubmissionQueue refresh |
| `route-changed` | `Router.resolve()` | NavBar active state update |

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| All 5 layers default ON | Choropleth + markers ON, others OFF | Phase 16 | Cleaner initial globe view |
| Checkbox panel for layer toggles | Floating pill bar | Phase 16 | Compact, doesn't consume sidebar space |
| `country-selected` opens CountryBriefPage modal | `country-selected` opens slide-in drill-down | Phase 16 | Lighter interaction, globe stays visible |
| Forecasts screen is placeholder | Full submission workflow | Phase 16 | Two-phase submit/confirm with queue |

**Deprecated/outdated:**
- The existing `DeckGLMap.createLayerToggles()` checkbox panel will be replaced by external `LayerPillBar`. The internal toggle state management (`layerVisible` map) stays but needs a `setLayerVisible()` public API.

## Open Questions

1. **GDELT event sparkline data source**
   - What we know: The drill-down panel should show a GDELT event sparkline. The `EventTimelinePanel` exists on the dashboard.
   - What's unclear: Phase 16 context says "Event sparkline real data and additional data sources are Phase 17 scope." Does the sparkline show mock/empty data in Phase 16, or is it omitted entirely until Phase 17?
   - Recommendation: Render the sparkline section in the drill-down panel with a placeholder ("Event data coming in Phase 17") or a flat zero-line. This preserves the layout without blocking on Phase 17.

2. **Risk score timeline in drill-down**
   - What we know: Drill-down should show "risk score with trend arrow." The `CountryRiskSummary` has `risk_score` and `trend` (rising/stable/falling).
   - What's unclear: Is this a single current value with trend indicator (arrow), or a historical timeline chart showing risk score over time? A timeline would require a new backend endpoint for historical risk data.
   - Recommendation: Display the current risk score with a trend arrow (from existing `trend` field). Historical timeline is Phase 17 scope. The UI section should be called "Risk Score" with the trend arrow, not "Risk Timeline."

3. **Pill bar positioning relative to drill-down panel**
   - What we know: Context says "Pill bar positioning relative to drill-down panel (avoid overlap)" is Claude's discretion.
   - What's unclear: When the drill-down panel slides in from the right (400px wide), does the pill bar shift left or stay centered?
   - Recommendation: Keep the pill bar centered (`left: 50%; transform: translateX(-50%)`). At bottom of the screen, it doesn't overlap with the right-side panel. If the viewport is narrow enough to cause overlap, the pill bar's `left: 50%` naturally shifts it away from the panel.

## Sources

### Primary (HIGH confidence)
- `/home/kondraki/personal/geopol/frontend/src/components/DeckGLMap.ts` -- Existing 5-layer globe implementation, layer toggle state, country click handling
- `/home/kondraki/personal/geopol/frontend/src/screens/globe-screen.ts` -- Current placeholder, dynamic import pattern
- `/home/kondraki/personal/geopol/frontend/src/screens/forecasts-screen.ts` -- Current placeholder
- `/home/kondraki/personal/geopol/frontend/src/components/ForecastPanel.ts` -- Progressive disclosure pattern (buildCard, buildExpandedContent, renderMiniTree)
- `/home/kondraki/personal/geopol/frontend/src/services/forecast-client.ts` -- Full API client with submission methods
- `/home/kondraki/personal/geopol/frontend/src/types/api.ts` -- All TypeScript interfaces for API responses
- `/home/kondraki/personal/geopol/frontend/src/screens/dashboard-screen.ts` -- Screen lifecycle pattern, RefreshScheduler wiring, modal instantiation
- `/home/kondraki/personal/geopol/frontend/src/app/router.ts` -- Router implementation, View Transition API
- `/home/kondraki/personal/geopol/src/api/routes/v1/submissions.py` -- Backend submission endpoints (submit, confirm, list)
- `/home/kondraki/personal/geopol/src/api/routes/v1/countries.py` -- Backend country risk aggregation SQL

### Secondary (HIGH confidence -- WM Reference)
- `/home/kondraki/personal/worldmonitor/src/components/DeckGLMap.ts` -- WM globe: flyTo, fitCountry, country click payload, layer toggle creation, 4600+ lines of reference patterns
- `/home/kondraki/personal/worldmonitor/src/components/CountryDeepDivePanel.ts` -- WM slide-in panel: getOrCreatePanel, show/hide/maximize, AbortController cancellation, section cards
- `/home/kondraki/personal/worldmonitor/src/styles/country-deep-dive.css` -- WM slide-in CSS: `position: fixed; right: -460px; transition: right 0.28s ease;`
- `/home/kondraki/personal/worldmonitor/src/app/country-intel.ts` -- WM country click handler: `briefRequestToken` pattern for race condition prevention, data fetching on country select

### Tertiary (MEDIUM confidence)
- maplibre-gl `flyTo` API -- based on WM usage patterns, not independently verified against latest docs
- deck.gl layer `opacity` prop -- verified in existing codebase usage, behavior at 0 opacity may still render the layer (consuming GPU)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already installed and in production use
- Architecture: HIGH -- patterns derived from existing geopol code + WM reference with full source access
- Pitfalls: HIGH -- identified from WM's production experience with identical technology stack
- Code examples: HIGH -- directly derived from existing codebase patterns, not hypothetical

**Research date:** 2026-03-03
**Valid until:** 2026-04-03 (stable -- no library upgrades expected)
