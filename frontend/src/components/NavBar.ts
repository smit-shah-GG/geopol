/**
 * Top navigation bar with route links, active-state highlighting, and
 * globe view toggle (3D/2D).
 *
 * Listens for `route-changed` custom events (dispatched by Router) and
 * `popstate` to keep the active link indicator in sync.
 *
 * The 3D/2D toggle button is visible ONLY on the /globe route. It dispatches
 * a `globe-view-toggle` CustomEvent. MapContainer handles the actual toggle
 * and dispatches `globe-mode-changed` back to update the button label.
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

/**
 * Create the navigation bar element wired to a Router instance.
 *
 * @param router  The application Router for programmatic navigation
 * @returns The `<nav>` element to append to the DOM
 */
export function createNavBar(router: Router): HTMLElement {
  const linkElements: HTMLAnchorElement[] = [];

  // --- View toggle button (3D/2D) ---
  let is3d = (localStorage.getItem(STORAGE_KEY) ?? '3d') === '3d';
  const viewToggle = h('button', {
    className: 'nav-view-toggle',
    'aria-label': 'Toggle 3D/2D globe view',
    title: 'Toggle 3D/2D view',
  }, is3d ? '3D' : '2D') as HTMLButtonElement;

  // Dispatch toggle event -- MapContainer handles the actual mode swap
  viewToggle.addEventListener('click', () => {
    window.dispatchEvent(new CustomEvent('globe-view-toggle'));
  });

  // Listen for mode change confirmation from MapContainer
  const modeChangedHandler = ((e: CustomEvent<{ mode: string }>) => {
    is3d = e.detail.mode === '3d';
    viewToggle.textContent = is3d ? '3D' : '2D';
    viewToggle.classList.toggle('active-3d', is3d);
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
    // Show/hide view toggle based on route -- only visible on /globe
    viewToggle.style.display = current === '/globe' ? 'inline-flex' : 'none';
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

  // --- Right container for view toggle ---
  const rightContainer = h('div', { className: 'nav-right' });
  rightContainer.appendChild(viewToggle);

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

  // Set initial active state (also sets toggle visibility)
  updateActive();

  return nav;
}
