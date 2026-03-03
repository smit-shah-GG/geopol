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

    // Set currentRoute eagerly BEFORE the async swap. This prevents a
    // re-entrant resolve() (triggered by rapid navigation or popstate during
    // an in-flight startViewTransition callback) from seeing stale state and
    // short-circuiting via the `route === this.currentRoute` guard above.
    const prevRoute = this.currentRoute;
    this.currentRoute = route;

    const doSwap = async (): Promise<void> => {
      try {
        if (prevRoute) {
          prevRoute.unmount();
        }
      } catch (e) {
        console.error('[Router] Unmount failed:', e);
      }
      this.container.innerHTML = '';
      await route.mount(this.container);
    };

    // Use View Transition API if available for a smooth crossfade.
    // Await updateCallbackDone so doSwap completes before we dispatch route-changed.
    const doc = document as Document & {
      startViewTransition?: (cb: () => Promise<void> | void) => {
        updateCallbackDone: Promise<void>;
        finished: Promise<void>;
      };
    };
    try {
      if (typeof doc.startViewTransition === 'function') {
        const transition = doc.startViewTransition(() => doSwap());
        await transition.updateCallbackDone;
      } else {
        await doSwap();
      }
    } catch (e) {
      // If transition or mount fails, ensure we still have a usable state.
      // Fall back to direct swap without transition animation.
      console.error('[Router] Screen transition failed:', e);
      try {
        this.container.innerHTML = '';
        await route.mount(this.container);
      } catch (retryErr) {
        console.error('[Router] Fallback mount also failed:', retryErr);
      }
    }

    window.dispatchEvent(new CustomEvent('route-changed', { detail: { path } }));
  }

  /** Return current pathname. */
  getCurrentPath(): string {
    return window.location.pathname;
  }
}
