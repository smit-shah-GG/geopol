/**
 * Client-side router using pushState / popstate for SPA screen navigation.
 *
 * Supports View Transition API where available (Chrome 111+), with
 * immediate DOM swap fallback on other browsers.
 *
 * The router owns a single container element. On each transition it unmounts
 * the previous screen and mounts the new one into that container.
 */

export interface Route {
  path: string;
  mount: (container: HTMLElement) => void | Promise<void>;
  unmount: () => void;
}

/**
 * Minimal hash-free SPA router.
 *
 * Usage:
 *   const router = new Router(screenContainer);
 *   router.addRoute({ path: '/dashboard', mount, unmount });
 *   router.resolve();  // mount initial screen from current URL
 */
export class Router {
  private routes: Route[] = [];
  private currentRoute: Route | null = null;
  private readonly container: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
    window.addEventListener('popstate', () => {
      void this.resolve();
    });
  }

  /** Register a screen route. */
  addRoute(route: Route): void {
    this.routes.push(route);
  }

  /** Programmatic navigation. Pushes history state and resolves. */
  navigate(path: string): void {
    if (window.location.pathname === path) return;
    window.history.pushState(null, '', path);
    void this.resolve();
  }

  /** Resolve current pathname to a route and perform the screen swap. */
  async resolve(): Promise<void> {
    const path = window.location.pathname;
    const route = this.routes.find((r) => r.path === path)
      ?? this.routes.find((r) => r.path === '/dashboard')
      ?? this.routes[0];

    if (!route || route === this.currentRoute) {
      window.dispatchEvent(new CustomEvent('route-changed', { detail: { path } }));
      return;
    }

    const doSwap = async (): Promise<void> => {
      if (this.currentRoute) {
        this.currentRoute.unmount();
      }
      this.container.innerHTML = '';
      await route.mount(this.container);
      this.currentRoute = route;
    };

    // Use View Transition API if available for a smooth crossfade
    const doc = document as Document & {
      startViewTransition?: (cb: () => Promise<void> | void) => void;
    };
    if (typeof doc.startViewTransition === 'function') {
      doc.startViewTransition(() => doSwap());
    } else {
      await doSwap();
    }

    window.dispatchEvent(new CustomEvent('route-changed', { detail: { path } }));
  }

  /** Return current pathname. */
  getCurrentPath(): string {
    return window.location.pathname;
  }
}
