# Phase 27: 3D Globe - Context

**Gathered:** 2026-03-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Add a globe.gl-based 3D spherical globe as the **default** globe view on `/globe`, retaining the existing deck.gl 2D flat map as a toggleable alternative. All 5 analytic layers (choropleth, markers, arcs, heatmap, scenarios) render on both views. This is additive — the 2D map is NOT replaced, it coexists.

**Desktop-only target.** No mobile/touch optimization required.

**Reference implementation:** World Monitor's `GlobeMap.ts` (2,188 lines, globe.gl v2.45.0). WM is the code quarry — patterns should be adapted, not blindly copied.

</domain>

<decisions>
## Implementation Decisions

### View toggle UX
- Toggle lives in the **nav bar** (WM-style header buttons), outside the globe viewport — not in the pill bar or HUD
- **MapContainer wrapper pattern** (WM's delegation approach): wrapper class holds both GlobeMap and DeckGLMap instances, dispatches all calls conditionally based on active mode. Both WebGL contexts stay alive — toggle swaps visibility, no destroy/recreate
- **3D is the default** for first-time visitors. Preference persisted in localStorage
- **Independent camera positions** per view — switching from 2D to 3D does NOT transfer the camera position. Each view remembers its own last state independently

### 3D globe visual identity
- **Dark topographic texture** as the globe surface — subtle, desaturated/darkened topo imagery that provides geographic context without competing with data layers. NOT photorealistic Blue Marble, NOT a bare polygon sphere
- **Atmosphere glow only, no starfield** — thin atmospheric glow ring around the globe, black background behind it. Analytical, not cinematic
- **Geopol accent blue (#4080dd)** for the atmosphere glow color — consistent with existing dark theme palette
- **One fixed visual/quality configuration** — no presets UI, no Eco/Sharp toggle. Ship one config tuned for desktop 60fps

### Layer rendering on 3D
- 4 of 5 layers map directly to globe.gl channels:
  - Risk Choropleth → `polygonsData` (GeoJSON country fills)
  - Forecast Markers → `htmlElementsData` (DOM overlays at lat/lon)
  - Knowledge Graph Arcs → `arcsData` (great-circle arcs)
  - Scenario Zones → `polygonsData` (country highlight overlays)
- **Heatmap: colored point markers** — convert H3 hex bin centers to colored dots on the sphere surface. Loses hex tessellation look but pragmatic for globe.gl
- **Slightly brighter colors on 3D** (~10-15% saturation/brightness bump) to compensate for dark topo texture and curved surface light falloff. Same palette family, just boosted
- **Same refresh intervals** as 2D (countries 120s, forecasts 60s, layers 300s). Data freshness is view-independent
- **Separate layer state per view** — toggling a layer off in 3D does NOT affect 2D layer state. Each view has its own independent layer visibility configuration. LayerPillBar dispatches to whichever renderer is active

### Interaction model
- **Rotate + zoom only** (WM style) — no panning. Globe is always centered in viewport. Prevents disorientation
- **Animated fly-to on country click** — click country → globe rotates and zooms to center on that country (1-1.5s animation) → GlobeDrillDown panel opens
- **8 regional view presets** (global, america, mena, eu, asia, latam, africa, oceania) — accessible from a selector **inside the GlobeHud** (top-left), alongside existing stats
- **Auto-rotate on desktop** — slow rotation when user isn't interacting. Pauses on mousedown, resumes after **120s** idle timeout (longer than WM's 60s — analytical tool, user is likely studying data)

### Claude's Discretion
- Globe.gl configuration details (sphere segments, altitude values, Fresnel shader params)
- Dark topo texture sourcing/processing (can reuse WM's earth-topo-bathy.jpg, darken via material settings)
- Exact brightness boost values for 3D layer colors
- Debounced flush implementation details (WM uses 100ms/300ms dual timer)
- Auto-rotate speed
- Fly-to animation duration and easing curve
- Region preset POV coordinates (can port WM's VIEW_POVS directly)
- MapContainer interface design and method delegation
- GlobeMap class structure and public API surface

</decisions>

<specifics>
## Specific Ideas

- WM's `GlobeMap.ts` (2,188 lines) is the primary reference — adapt its globe.gl initialization, OrbitControls config, polygon/arc/marker rendering, and debounced flush pattern
- WM's `MapContainer.ts` (~500 lines) is the delegation pattern reference — wrapper that dispatches to GlobeMap or DeckGLMap based on active mode
- WM's `VIEW_POVS` object with 8 region coordinates can be ported directly
- WM's renderer config: `powerPreference: 'high-performance'` for desktop, antialias when pixelRatio > 1
- WM's Cosmos preset atmosphere: dual SphereGeometry glow layers (outer radius 2.15, inner 2.08) with BackSide rendering — adapt with Geopol blue (#4080dd) instead of WM cyan (#00d4ff)
- MeshStandardMaterial for globe surface (roughness 0.8, metalness 0.1) — matte analytical look
- Earth texture file: can darken WM's `earth-topo-bathy.jpg` (699KB) at material level via emissive color

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 27-3d-globe*
*Context gathered: 2026-03-09*
