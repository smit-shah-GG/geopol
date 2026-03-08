/**
 * LayerPillBar -- Floating horizontal pill bar for toggling deck.gl layers.
 *
 * 5 pills matching the 5 analytic layers in DeckGLMap:
 *   1. Risk       (ForecastRiskChoropleth)  -- default ON
 *   2. Markers    (ActiveForecastMarkers)   -- default ON
 *   3. Arcs       (KnowledgeGraphArcs)      -- default OFF
 *   4. Heatmap    (GDELTEventHeatmap)       -- default OFF
 *   5. Scenarios  (ScenarioZones)           -- default OFF
 *
 * Reads initial state from deckMap.getLayerVisible() and mutates via
 * deckMap.setLayerVisible(). Purely UI -- no data fetching, no timers.
 */

import { h } from '@/utils/dom-utils';
import type { DeckGLMap, LayerId } from '@/components/DeckGLMap';

/** Display config: [layerId, short label for pill] */
const LAYER_PILLS: [LayerId, string][] = [
  ['ForecastRiskChoropleth', 'Risk'],
  ['ActiveForecastMarkers', 'Markers'],
  ['KnowledgeGraphArcs', 'Arcs'],
  ['GDELTEventHeatmap', 'Heatmap'],
  ['ScenarioZones', 'Scenarios'],
];

export class LayerPillBar {
  private readonly map: DeckGLMap;
  private readonly bar: HTMLElement;
  private readonly states: Record<LayerId, boolean>;

  constructor(deckMap: DeckGLMap) {
    this.map = deckMap;

    // Snapshot current layer visibility from the map
    this.states = {} as Record<LayerId, boolean>;
    for (const [id] of LAYER_PILLS) {
      this.states[id] = deckMap.getLayerVisible(id);
    }

    this.bar = this.buildBar();
  }

  private buildBar(): HTMLElement {
    const bar = h('div', {
      className: 'layer-pill-bar',
      role: 'toolbar',
      'aria-label': 'Map layers',
    });

    for (const [id, label] of LAYER_PILLS) {
      const isActive = this.states[id];
      const pill = h('button', {
        className: `layer-pill${isActive ? ' active' : ''}`,
        dataset: { layer: id },
        'aria-label': `Toggle ${label} layer`,
        'aria-pressed': String(isActive),
      }, label);

      pill.addEventListener('click', () => this.toggle(id));
      bar.appendChild(pill);
    }

    return bar;
  }

  private toggle(layerId: LayerId): void {
    const newState = !this.states[layerId];
    this.states[layerId] = newState;

    // Update pill visual state and ARIA pressed state
    const pill = this.bar.querySelector(`[data-layer="${layerId}"]`);
    if (pill) {
      pill.classList.toggle('active', newState);
      pill.setAttribute('aria-pressed', String(newState));
    }

    // Push to DeckGLMap
    this.map.setLayerVisible(layerId, newState);
  }

  getElement(): HTMLElement {
    return this.bar;
  }

  destroy(): void {
    // Pure DOM with inline listeners -- no external cleanup needed
  }
}
