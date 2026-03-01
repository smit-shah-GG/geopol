import { applyStoredTheme, toggleTheme } from '@/utils/theme-manager';
import { createAppContext } from '@/app/app-context';
import { h } from '@/utils/dom-utils';
import '@/styles/main.css';
import '@/styles/panels.css';

function createHeader(): HTMLElement {
  const themeBtn = h('button', {
    className: 'header-theme-toggle',
    'aria-label': 'Toggle theme',
    onClick: () => toggleTheme(),
  }, 'Theme');

  return h('header', { className: 'app-header' },
    h('div', { className: 'header-left' },
      h('span', { className: 'header-logo' }, 'GEOPOL'),
      h('span', { className: 'header-subtitle' }, 'Geopolitical Forecast Dashboard'),
    ),
    h('div', { className: 'header-right' },
      themeBtn,
    ),
  );
}

function createPanelGrid(): HTMLElement {
  const grid = h('div', { className: 'panel-grid' });

  const placeholders = [
    'Globe View',
    'Active Forecasts',
    'Country Risk',
    'Scenario Explorer',
    'Evidence Chain',
    'System Health',
  ];

  for (const label of placeholders) {
    const panel = h('div', { className: 'panel placeholder' },
      h('div', { className: 'panel-header' },
        h('span', { className: 'panel-title' }, label),
      ),
      h('div', { className: 'panel-content' },
        h('div', { className: 'panel-loading' },
          h('div', { className: 'panel-loading-text' }, 'Awaiting data...'),
        ),
      ),
    );
    grid.appendChild(panel);
  }

  return grid;
}

function boot(): void {
  applyStoredTheme();

  const app = document.getElementById('app');
  if (!app) throw new Error('Missing #app container');

  const ctx = createAppContext(app);

  app.innerHTML = '';
  app.appendChild(createHeader());
  app.appendChild(createPanelGrid());

  // Store context on window for debugging in dev
  if (import.meta.env.DEV) {
    (window as unknown as Record<string, unknown>)['__geopol'] = ctx;
  }
}

boot();
