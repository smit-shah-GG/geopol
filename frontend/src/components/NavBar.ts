/**
 * Top navigation bar with route links, active-state highlighting, and
 * globe scene mode segmented control (3D / CV / 2D).
 *
 * Listens for `route-changed` custom events (dispatched by Router) and
 * `popstate` to keep the active link indicator in sync.
 *
 * The scene mode pills are visible ONLY on the /globe route. Clicking a pill
 * dispatches a `globe-view-toggle` CustomEvent with `{ mode: '3d' | 'columbus' | '2d' }`.
 * CesiumMap handles the morph and dispatches `globe-mode-changed` back to sync
 * the active pill state.
 */

import { h } from '@/utils/dom-utils';
import type { Router } from '@/app/router';

interface NavLink {
  label: string;
  path: string;
}

const NAV_LINKS: NavLink[] = [
  { label: 'Dashboard', path: '/dashboard' },
  { label: 'Globe', path: '/globe' },
  { label: 'Forecasts', path: '/forecasts' },
];

const STORAGE_KEY = 'geopol-globe-mode';

type SceneMode = '3d' | 'columbus' | '2d';

const SCENE_MODES: ReadonlyArray<{ id: SceneMode; label: string; ariaLabel: string }> = [
  { id: '3d', label: '3D', ariaLabel: 'Switch to 3D mode' },
  { id: 'columbus', label: 'CV', ariaLabel: 'Switch to Columbus View mode' },
  { id: '2d', label: '2D', ariaLabel: 'Switch to 2D mode' },
];

// Inject segmented control styles once at module scope
const styleEl = document.createElement('style');
styleEl.textContent = `
.nav-scene-mode {
  display: inline-flex;
  gap: 2px;
  background: rgba(255, 255, 255, 0.08);
  border-radius: 6px;
  padding: 2px;
}
.scene-pill {
  padding: 4px 10px;
  border: none;
  background: transparent;
  color: var(--text-muted, #6a7a8c);
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 600;
  transition: background 0.15s, color 0.15s;
  line-height: 1;
}
.scene-pill:hover {
  color: #c0c8d0;
}
.scene-pill.active {
  background: var(--accent, #4080dd);
  color: #fff;
}
`;
document.head.appendChild(styleEl);

/**
 * Create the navigation bar element wired to a Router instance.
 *
 * @param router  The application Router for programmatic navigation
 * @returns The `<nav>` element to append to the DOM
 */
export function createNavBar(router: Router): HTMLElement {
  const linkElements: HTMLAnchorElement[] = [];

  // --- Scene mode segmented control [3D] [CV] [2D] ---
  const currentMode: SceneMode =
    (localStorage.getItem(STORAGE_KEY) as SceneMode | null) ?? '3d';

  const sceneModeContainer = h('div', {
    className: 'nav-scene-mode',
    'aria-label': 'Globe scene mode',
  }) as HTMLDivElement;

  const pillButtons: HTMLButtonElement[] = [];

  for (const mode of SCENE_MODES) {
    const pill = h('button', {
      className: `scene-pill${mode.id === currentMode ? ' active' : ''}`,
      'aria-label': mode.ariaLabel,
      dataset: { mode: mode.id },
    }, mode.label) as HTMLButtonElement;

    pill.addEventListener('click', () => {
      localStorage.setItem(STORAGE_KEY, mode.id);
      // Update pill active states immediately for responsive feel
      for (const btn of pillButtons) {
        btn.classList.toggle('active', btn === pill);
      }
      window.dispatchEvent(
        new CustomEvent('globe-view-toggle', { detail: { mode: mode.id } }),
      );
    });

    pillButtons.push(pill);
    sceneModeContainer.appendChild(pill);
  }

  // Listen for mode change confirmation from CesiumMap to sync pill state
  const modeChangedHandler = ((e: CustomEvent<{ mode: string }>) => {
    for (const btn of pillButtons) {
      btn.classList.toggle('active', btn.dataset['mode'] === e.detail.mode);
    }
  }) as EventListener;
  window.addEventListener('globe-mode-changed', modeChangedHandler);

  // --- Active link state ---
  const updateActive = (): void => {
    const current = router.getCurrentPath();
    for (const a of linkElements) {
      const isActive = a.dataset['path'] === current
        || (current === '/' && a.dataset['path'] === '/dashboard');
      a.classList.toggle('nav-link--active', isActive);
      if (isActive) {
        a.setAttribute('aria-current', 'page');
      } else {
        a.removeAttribute('aria-current');
      }
    }
    // Show/hide scene mode pills based on route -- only visible on /globe
    sceneModeContainer.style.display = current === '/globe' ? 'inline-flex' : 'none';
  };

  const linksContainer = h('div', { className: 'nav-links' });

  for (const link of NAV_LINKS) {
    const a = h('a', {
      className: 'nav-link',
      href: link.path,
      dataset: { path: link.path },
    }, link.label) as HTMLAnchorElement;

    a.addEventListener('click', (e: Event) => {
      e.preventDefault();
      router.navigate(link.path);
    });

    linkElements.push(a);
    linksContainer.appendChild(a);
  }

  // --- Right container for scene mode control ---
  const rightContainer = h('div', { className: 'nav-right' });
  rightContainer.appendChild(sceneModeContainer);

  const nav = h('nav', { className: 'nav-bar', 'aria-label': 'Main navigation' },
    h('div', { className: 'nav-left' },
      h('span', { className: 'nav-logo' }, 'GEOPOL'),
    ),
    linksContainer,
    rightContainer,
  );

  // Sync active link on route change (covers both navigate() and popstate)
  window.addEventListener('route-changed', updateActive);
  window.addEventListener('popstate', updateActive);

  // Set initial active state (also sets pill visibility)
  updateActive();

  return nav;
}
