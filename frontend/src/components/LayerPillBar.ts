/**
 * LayerPillBar -- Floating horizontal pill bar for toggling map layers.
 *
 * 5 pills matching the 5 analytic layers:
 *   1. Risk       (ForecastRiskChoropleth)  -- default ON
 *   2. Markers    (ActiveForecastMarkers)   -- default ON
 *   3. Arcs       (KnowledgeGraphArcs)      -- default OFF
 *   4. Heatmap    (GDELTEventHeatmap)       -- default OFF
 *   5. Scenarios  (ScenarioZones)           -- default OFF
 *
 * Accepts any renderer implementing LayerController (e.g. CesiumMap).
 * Reads initial state via getLayerVisible() and mutates via
 * setLayerVisible(). Purely UI -- no data fetching, no timers.
 */

import { h } from '@/utils/dom-utils';
import type { LayerId } from '@/components/CesiumMap';

/** Any renderer that supports layer visibility control. */
export interface LayerController {
  getLayerVisible(layerId: LayerId): boolean;
  setLayerVisible(layerId: LayerId, visible: boolean): void;
}

/** Display config: [layerId, short label for pill] */
const LAYER_PILLS: [LayerId, string][] = [
  ['ForecastRiskChoropleth', 'Risk'],
  ['ActiveForecastMarkers', 'Markers'],
  ['KnowledgeGraphArcs', 'Arcs'],
  ['GDELTEventHeatmap', 'Heatmap'],
  ['ScenarioZones', 'Scenarios'],
];

export class LayerPillBar {
  private readonly controller: LayerController;
  private readonly bar: HTMLElement;
  private readonly states: Record<LayerId, boolean>;
  private modeChangeHandler: (() => void) | null = null;

  constructor(controller: LayerController) {
    this.controller = controller;

    // Snapshot current layer visibility from the controller
    this.states = {} as Record<LayerId, boolean>;
    for (const [id] of LAYER_PILLS) {
      this.states[id] = controller.getLayerVisible(id);
    }

    this.bar = this.buildBar();

    // Resync pill states when the view mode changes (3D <-> 2D toggle)
    this.modeChangeHandler = () => this.syncFromController();
    window.addEventListener('globe-mode-changed', this.modeChangeHandler);
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

    // Push to active renderer via controller
    this.controller.setLayerVisible(layerId, newState);
  }

  getElement(): HTMLElement {
    return this.bar;
  }

  /** Resync pill states from controller after view mode change. */
  syncFromController(): void {
    for (const [id] of LAYER_PILLS) {
      const visible = this.controller.getLayerVisible(id);
      this.states[id] = visible;
      const pill = this.bar.querySelector(`[data-layer="${id}"]`);
      if (pill) {
        pill.classList.toggle('active', visible);
        pill.setAttribute('aria-pressed', String(visible));
      }
    }
  }

  destroy(): void {
    if (this.modeChangeHandler) {
      window.removeEventListener('globe-mode-changed', this.modeChangeHandler);
      this.modeChangeHandler = null;
    }
  }
}
