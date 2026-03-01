import { Panel } from './Panel';
import { h, replaceChildren } from '@/utils/dom-utils';
import type { CalibrationDTO } from '@/types/api';

const SVG_NS = 'http://www.w3.org/2000/svg';

/** Create an SVG element with attributes. */
function svg(tag: string, attrs: Record<string, string | number>): SVGElement {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    el.setAttribute(k, String(v));
  }
  return el;
}

/** Brier score severity: <0.1 excellent, <0.25 good, >=0.25 poor. */
function brierClass(score: number): string {
  if (score < 0.1) return 'brier-excellent';
  if (score < 0.25) return 'brier-good';
  return 'brier-poor';
}

/**
 * CalibrationPanel -- reliability diagram + Brier decomposition + track record sparkline.
 *
 * Update-driven: receives CalibrationDTO[] from coordinated loads.
 * refresh() is a no-op since calibration data comes from forecast resolution.
 */
export class CalibrationPanel extends Panel {
  constructor() {
    super({ id: 'calibration', title: 'CALIBRATION', showCount: true });
    this.showPlaceholder();
  }

  /** No-op. Calibration data arrives via update(). */
  public refresh(): void {
    // Intentionally empty -- data arrives via update()
  }

  /** Render all three calibration visualizations. */
  public update(calibrations: CalibrationDTO[]): void {
    this.setCount(calibrations.length);

    if (calibrations.length === 0) {
      this.showPlaceholder();
      return;
    }

    replaceChildren(this.content,
      this.buildReliabilityDiagram(calibrations),
      this.buildBrierTable(calibrations),
      this.buildTrackRecordSparkline(calibrations),
    );
  }

  private showPlaceholder(): void {
    replaceChildren(this.content,
      h('div', { className: 'empty-state' },
        'Calibration data populates as predictions resolve',
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // 1. Reliability Diagram (SVG ~300x300)
  // ---------------------------------------------------------------------------

  private buildReliabilityDiagram(calibrations: CalibrationDTO[]): HTMLElement {
    const W = 300;
    const H = 300;
    const PAD = 40;
    const PLOT_W = W - PAD * 2;
    const PLOT_H = H - PAD * 2;

    const root = svg('svg', {
      width: W,
      height: H,
      viewBox: `0 0 ${W} ${H}`,
      class: 'reliability-diagram',
    });

    // Background
    root.appendChild(svg('rect', {
      x: 0, y: 0, width: W, height: H,
      fill: 'transparent',
    }));

    // Plot area border
    root.appendChild(svg('rect', {
      x: PAD, y: PAD, width: PLOT_W, height: PLOT_H,
      fill: 'none',
      stroke: 'var(--border)',
      'stroke-width': 1,
    }));

    // Grid lines (0.2 intervals)
    for (let i = 1; i < 5; i++) {
      const frac = i * 0.2;
      const x = PAD + frac * PLOT_W;
      const y = PAD + (1 - frac) * PLOT_H;

      // Vertical grid
      root.appendChild(svg('line', {
        x1: x, y1: PAD, x2: x, y2: PAD + PLOT_H,
        stroke: 'var(--border-subtle)',
        'stroke-width': 0.5,
      }));

      // Horizontal grid
      root.appendChild(svg('line', {
        x1: PAD, y1: y, x2: PAD + PLOT_W, y2: y,
        stroke: 'var(--border-subtle)',
        'stroke-width': 0.5,
      }));
    }

    // Perfect calibration diagonal (dashed)
    root.appendChild(svg('line', {
      x1: PAD,
      y1: PAD + PLOT_H,
      x2: PAD + PLOT_W,
      y2: PAD,
      stroke: 'var(--text-muted)',
      'stroke-width': 1,
      'stroke-dasharray': '4,3',
    }));

    // Axis labels
    const labels = ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'];
    labels.forEach((label, i) => {
      const frac = i * 0.2;

      // X-axis
      const xLabel = svg('text', {
        x: PAD + frac * PLOT_W,
        y: H - 6,
        fill: 'var(--text-muted)',
        'font-size': 9,
        'font-family': 'var(--font-mono)',
        'text-anchor': 'middle',
      });
      xLabel.textContent = label;
      root.appendChild(xLabel);

      // Y-axis
      const yLabel = svg('text', {
        x: PAD - 6,
        y: PAD + (1 - frac) * PLOT_H + 3,
        fill: 'var(--text-muted)',
        'font-size': 9,
        'font-family': 'var(--font-mono)',
        'text-anchor': 'end',
      });
      yLabel.textContent = label;
      root.appendChild(yLabel);
    });

    // Axis titles
    const xTitle = svg('text', {
      x: W / 2,
      y: H - 0,
      fill: 'var(--text-muted)',
      'font-size': 8,
      'font-family': 'var(--font-mono)',
      'text-anchor': 'middle',
      'letter-spacing': '0.5',
    });
    xTitle.textContent = 'PREDICTED';
    root.appendChild(xTitle);

    // Data dots (predicted = x, historical_accuracy = observed proxy on y)
    const maxSampleSize = Math.max(...calibrations.map((c) => c.sample_size), 1);
    for (const cal of calibrations) {
      const brier = cal.brier_score;
      if (brier === null) continue;

      // Use brier_score position: predicted = temperature (proxy), observed = historical_accuracy
      // Since CalibrationDTO has temperature (predicted bin proxy) and historical_accuracy (observed)
      const predicted = cal.temperature;
      const observed = cal.historical_accuracy;

      // Clamp to [0, 1]
      const px = Math.max(0, Math.min(1, predicted));
      const py = Math.max(0, Math.min(1, observed));

      const cx = PAD + px * PLOT_W;
      const cy = PAD + (1 - py) * PLOT_H;
      const radius = 3 + Math.sqrt(cal.sample_size / maxSampleSize) * 8;

      root.appendChild(svg('circle', {
        cx, cy, r: radius,
        fill: 'var(--accent)',
        'fill-opacity': 0.7,
        stroke: 'var(--accent)',
        'stroke-width': 1,
      }));
    }

    const wrapper = h('div', { className: 'reliability-diagram-wrapper' },
      h('div', { className: 'section-label' }, 'RELIABILITY DIAGRAM'),
    );
    wrapper.appendChild(root as unknown as Node);
    return wrapper;
  }

  // ---------------------------------------------------------------------------
  // 2. Brier Score Table
  // ---------------------------------------------------------------------------

  private buildBrierTable(calibrations: CalibrationDTO[]): HTMLElement {
    const headerRow = h('div', { className: 'brier-row brier-header' },
      h('span', { className: 'brier-cell brier-cell-cat' }, 'CATEGORY'),
      h('span', { className: 'brier-cell brier-cell-score' }, 'BRIER'),
      h('span', { className: 'brier-cell brier-cell-n' }, 'N'),
    );

    const dataRows = calibrations.map((cal, i) => {
      const score = cal.brier_score;
      const scoreText = score !== null ? score.toFixed(4) : '--';
      const cls = score !== null ? brierClass(score) : '';

      return h('div', { className: `brier-row ${i % 2 === 0 ? 'even' : 'odd'}` },
        h('span', { className: 'brier-cell brier-cell-cat' }, cal.category),
        h('span', { className: `brier-cell brier-cell-score ${cls}` }, scoreText),
        h('span', { className: 'brier-cell brier-cell-n' }, String(cal.sample_size)),
      );
    });

    return h('div', { className: 'brier-table' },
      h('div', { className: 'section-label' }, 'BRIER DECOMPOSITION'),
      headerRow,
      ...dataRows,
    );
  }

  // ---------------------------------------------------------------------------
  // 3. Track Record Sparkline (SVG ~300x80)
  // ---------------------------------------------------------------------------

  private buildTrackRecordSparkline(calibrations: CalibrationDTO[]): HTMLElement {
    const W = 300;
    const H = 80;
    const PAD_X = 10;
    const PAD_Y = 10;
    const PLOT_W = W - PAD_X * 2;
    const PLOT_H = H - PAD_Y * 2;

    const root = svg('svg', {
      width: W,
      height: H,
      viewBox: `0 0 ${W} ${H}`,
      class: 'track-record-sparkline',
    });

    // Filter to entries with valid accuracy data
    const dataPoints = calibrations
      .filter((c) => c.historical_accuracy > 0)
      .map((c) => c.historical_accuracy);

    if (dataPoints.length < 3) {
      // Placeholder: dashed horizontal line at 0.5
      root.appendChild(svg('line', {
        x1: PAD_X,
        y1: H / 2,
        x2: W - PAD_X,
        y2: H / 2,
        stroke: 'var(--text-muted)',
        'stroke-width': 1,
        'stroke-dasharray': '4,3',
      }));

      const placeholder = svg('text', {
        x: W / 2,
        y: H / 2 - 8,
        fill: 'var(--text-muted)',
        'font-size': 8,
        'font-family': 'var(--font-mono)',
        'text-anchor': 'middle',
      });
      placeholder.textContent = 'Track record populates as predictions resolve';
      root.appendChild(placeholder);
    } else {
      // Y-axis range: 0 to 1
      // Subtle horizontal guides at 0.25, 0.5, 0.75
      for (const frac of [0.25, 0.5, 0.75]) {
        const y = PAD_Y + (1 - frac) * PLOT_H;
        root.appendChild(svg('line', {
          x1: PAD_X, y1: y, x2: W - PAD_X, y2: y,
          stroke: 'var(--border-subtle)',
          'stroke-width': 0.5,
        }));
      }

      // Polyline path
      const points = dataPoints.map((val, i) => {
        const x = PAD_X + (i / (dataPoints.length - 1)) * PLOT_W;
        const y = PAD_Y + (1 - val) * PLOT_H;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(' ');

      root.appendChild(svg('polyline', {
        points,
        fill: 'none',
        stroke: 'var(--semantic-warning)',
        'stroke-width': 1.5,
        'stroke-linejoin': 'round',
        'stroke-linecap': 'round',
      }));

      // Dots at each data point
      dataPoints.forEach((val, i) => {
        const x = PAD_X + (i / (dataPoints.length - 1)) * PLOT_W;
        const y = PAD_Y + (1 - val) * PLOT_H;
        root.appendChild(svg('circle', {
          cx: x, cy: y, r: 2,
          fill: 'var(--semantic-warning)',
        }));
      });
    }

    const wrapper = h('div', { className: 'track-record-wrapper' },
      h('div', { className: 'section-label' }, 'PREDICTION TRACK RECORD'),
    );
    wrapper.appendChild(root as unknown as Node);
    return wrapper;
  }
}
