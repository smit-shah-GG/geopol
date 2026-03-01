import { Panel } from './Panel';
import { h, replaceChildren } from '@/utils/dom-utils';
import type { ForecastResponse } from '@/types/api';

const LLM_COLOR = '#3388ff';
const TKG_COLOR = '#ff8800';

/**
 * EnsembleBreakdownPanel -- LLM vs TKG contribution visualization.
 *
 * Update-driven: receives a ForecastResponse when user selects a forecast.
 * refresh() is a no-op since this panel doesn't poll independently.
 */
export class EnsembleBreakdownPanel extends Panel {
  constructor() {
    super({ id: 'ensemble', title: 'ENSEMBLE WEIGHTS' });
    this.showPlaceholder();
  }

  /** No-op. This panel is update-driven, not poll-driven. */
  public refresh(): void {
    // Intentionally empty -- data arrives via update()
  }

  /** Render ensemble breakdown for the given forecast. */
  public update(forecast: ForecastResponse): void {
    const info = forecast.ensemble_info;

    const llmPct = info.llm_probability * 100;
    const tkgPct = (info.tkg_probability ?? 0) * 100;

    // Extract weight percentages
    const llmWeight = (info.weights['llm'] ?? info.weights['LLM'] ?? 0.6) * 100;
    const tkgWeight = (info.weights['tkg'] ?? info.weights['TKG'] ?? 0.4) * 100;

    replaceChildren(this.content,
      // Stacked bar
      h('div', { className: 'ensemble-section' },
        h('div', { className: 'ensemble-label' }, 'PROBABILITY SOURCES'),
        h('div', { className: 'ensemble-bar' },
          h('div', {
            className: 'ensemble-segment llm',
            style: `width: ${llmPct}%; background: ${LLM_COLOR}`,
          }),
          h('div', {
            className: 'ensemble-segment tkg',
            style: `width: ${tkgPct}%; background: ${TKG_COLOR}`,
          }),
        ),
      ),

      // Values
      h('div', { className: 'ensemble-values' },
        h('div', { className: 'ensemble-value' },
          h('span', { className: 'ensemble-dot', style: `background: ${LLM_COLOR}` }),
          h('span', null, 'LLM'),
          h('span', { className: 'ensemble-pct' }, `${llmPct.toFixed(1)}%`),
        ),
        h('div', { className: 'ensemble-value' },
          h('span', { className: 'ensemble-dot', style: `background: ${TKG_COLOR}` }),
          h('span', null, 'TKG'),
          h('span', { className: 'ensemble-pct' }, `${tkgPct.toFixed(1)}%`),
        ),
      ),

      // Weights
      h('div', { className: 'ensemble-section' },
        h('div', { className: 'ensemble-label' }, 'WEIGHTS'),
        h('div', { className: 'ensemble-weights-text' },
          `LLM: ${llmWeight.toFixed(0)}% | TKG: ${tkgWeight.toFixed(0)}%`,
        ),
      ),

      // Temperature
      h('div', { className: 'ensemble-section' },
        h('div', { className: 'ensemble-label' }, 'TEMPERATURE'),
        h('div', { className: 'ensemble-temp' }, info.temperature_applied.toFixed(3)),
      ),
    );
  }

  private showPlaceholder(): void {
    replaceChildren(this.content,
      h('div', { className: 'empty-state' }, 'Select a forecast to see ensemble breakdown'),
    );
  }
}
