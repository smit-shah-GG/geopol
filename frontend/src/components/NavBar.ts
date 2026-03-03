/**
 * Top navigation bar with route links and active-state highlighting.
 *
 * Listens for `route-changed` custom events (dispatched by Router) and
 * `popstate` to keep the active link indicator in sync.
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

/**
 * Create the navigation bar element wired to a Router instance.
 *
 * @param router  The application Router for programmatic navigation
 * @returns The `<nav>` element to append to the DOM
 */
export function createNavBar(router: Router): HTMLElement {
  const linkElements: HTMLAnchorElement[] = [];

  const updateActive = (): void => {
    const current = router.getCurrentPath();
    for (const a of linkElements) {
      const isActive = a.dataset['path'] === current
        || (current === '/' && a.dataset['path'] === '/dashboard');
      a.classList.toggle('nav-link--active', isActive);
    }
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

  const nav = h('nav', { className: 'nav-bar' },
    h('div', { className: 'nav-left' },
      h('span', { className: 'nav-logo' }, 'GEOPOL'),
    ),
    linksContainer,
  );

  // Sync active link on route change (covers both navigate() and popstate)
  window.addEventListener('route-changed', updateActive);
  window.addEventListener('popstate', updateActive);

  // Set initial active state
  updateActive();

  return nav;
}
