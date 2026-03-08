---
status: diagnosed
trigger: "layer visibility state is NOT independent per view (3D vs 2D) -- toggling a layer off in 3D also toggles it off in 2D"
created: 2026-03-09T00:00:00Z
updated: 2026-03-09T00:00:00Z
---

## Current Focus

hypothesis: LayerPillBar caches its own `states` Record at construction time and never refreshes it on view toggle -- so when user toggles a layer, the pill bar's internal state diverges from MapContainer's per-view state, and on view switch the stale pill bar state overwrites the new view's state
test: trace the full toggle lifecycle -- pill click -> MapContainer.setLayerVisible -> view switch -> syncLayerVisibility -> pill bar re-read
expecting: LayerPillBar.states is shared across both views; toggling updates states once, and on view switch the pill bar still shows the old toggle state and pushes it to the new view
next_action: analyze LayerPillBar.toggle() and its interaction with MapContainer on view switch

## Symptoms

expected: Each view (3D and 2D) maintains independent layer visibility. Toggling "Arcs" off in 3D should NOT affect Arcs visibility in 2D.
actual: Toggling a layer off in 3D also toggles it off in 2D, and vice versa.
errors: none reported
reproduction: Toggle a layer pill in one view, switch views, observe the layer is also toggled in the other view
started: since Phase 27 implementation

## Eliminated

- hypothesis: MapContainer aliases layerState3d and layerState2d to same object
  evidence: Lines 100-101 use independent spread copies `{ ...DEFAULT_LAYER_STATE }`
  timestamp: 2026-03-09

- hypothesis: MapContainer.setLayerVisible writes to both views
  evidence: Lines 193-201 branch on activeMode, only write to the active view's state
  timestamp: 2026-03-09

- hypothesis: MapContainer.getLayerVisible reads from wrong view
  evidence: Lines 203-207 branch on activeMode correctly
  timestamp: 2026-03-09

- hypothesis: syncLayerVisibility is broken
  evidence: Lines 310-318 read the new view's state and push to the new view's renderer -- correct
  timestamp: 2026-03-09

- hypothesis: LayerPillBar wired to renderer instead of MapContainer
  evidence: globe-screen.ts line 128 passes mapContainer to LayerPillBar constructor
  timestamp: 2026-03-09

## Evidence

- timestamp: 2026-03-09
  checked: LayerPillBar.states (line 37)
  found: Single Record<LayerId, boolean> shared across both views -- no per-view separation
  implication: Pill bar internal state does not track which view the toggle applies to

- timestamp: 2026-03-09
  checked: LayerPillBar.toggle() (lines 74-87)
  found: toggle() reads/writes this.states (view-agnostic) then calls controller.setLayerVisible
  implication: The pill DOM and internal state reflect whichever view was active at toggle time -- not the currently active view

- timestamp: 2026-03-09
  checked: globe-mode-changed event listeners
  found: Only NavBar listens for globe-mode-changed. LayerPillBar does not listen for any mode change event
  implication: LayerPillBar has no mechanism to resync states or pill DOM when view mode changes

- timestamp: 2026-03-09
  checked: LayerPillBar public API
  found: No refresh(), sync(), or update() method exists. Only getElement() and destroy()
  implication: Even if globe-screen.ts wanted to manually resync the pill bar on view toggle, there is no API to do so

## Resolution

root_cause: LayerPillBar maintains a SINGLE `states: Record<LayerId, boolean>` cache (line 37) that is view-mode-agnostic. When user toggles a layer in view A, the pill bar updates its shared `states` and DOM. When user switches to view B, MapContainer correctly restores view B's layer state to the renderer via `syncLayerVisibility()`, but nobody tells LayerPillBar to (1) re-read `getLayerVisible()` for all layers under the new view's state, or (2) update the pill DOM to match. The pill bar's stale `states` from view A persist, and subsequent pill clicks in view B compute `!staleState` rather than `!viewBState`, causing cross-view state pollution through the pill bar as intermediary.
fix:
verification:
files_changed: []
