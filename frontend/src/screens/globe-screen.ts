/**
 * Globe screen -- full-height DeckGLMap with dynamic import.
 *
 * deck.gl / maplibre-gl bundles are NOT loaded on the dashboard route.
 * The dynamic import() ensures code-splitting at the route level.
 *
 * Phase 16 will expand this into the full-viewport globe with overlay
 * panels. For now it's a functional placeholder that loads the existing
 * DeckGLMap component.
 */

import { h } from '@/utils/dom-utils';
import { countryGeometry } from '@/services/country-geometry';
import type { GeoPolAppContext } from '@/app/app-context';
import type { DeckGLMap } from '@/components/DeckGLMap';

let deckMap: DeckGLMap | null = null;

export async function mountGlobe(container: HTMLElement, _ctx: GeoPolAppContext): Promise<void> {
  // Outer wrapper fills the screen container
  const wrapper = h('div', {
    className: 'globe-screen',
    style: 'display:flex;flex-direction:column;flex:1;min-height:0;overflow:hidden;',
  });

  const label = h('div', {
    style: 'padding:4px 16px;font-family:var(--font-mono);font-size:10px;color:var(--text-muted);letter-spacing:1px;text-transform:uppercase;flex-shrink:0;',
  }, 'Globe Screen');

  const mapContainer = h('div', {
    style: 'flex:1;min-height:0;position:relative;overflow:hidden;',
  });

  wrapper.appendChild(label);
  wrapper.appendChild(mapContainer);
  container.appendChild(wrapper);

  try {
    // Dynamic import: deck.gl + maplibre chunks only load when globe screen mounts
    await countryGeometry.load();
    const { DeckGLMap } = await import('@/components/DeckGLMap');
    // Also load maplibre CSS dynamically
    await import('maplibre-gl/dist/maplibre-gl.css');

    deckMap = new DeckGLMap(mapContainer);
  } catch (err) {
    console.error('[GlobeScreen] Failed to load DeckGLMap:', err);
    mapContainer.innerHTML = '';
    mapContainer.appendChild(
      h('div', { className: 'screen-placeholder' }, 'Globe failed to load'),
    );
  }
}

export function unmountGlobe(_ctx: GeoPolAppContext): void {
  if (deckMap) {
    deckMap.destroy();
    deckMap = null;
  }
}
