import type { Panel } from '@/components/Panel';
import type { RefreshScheduler } from '@/app/refresh-scheduler';

/**
 * Lifecycle interface for modules registered in the app context.
 * Each module has init/destroy to manage resources.
 */
export interface AppModule {
  init(): void | Promise<void>;
  destroy(): void;
}

/**
 * GeoPolAppContext -- global application state singleton.
 *
 * This is NOT WM's 100+ field monster. Geopol has a small, focused context:
 * forecasts, country risk, health status, and the panel registry.
 *
 * The context is created once at boot and destroyed on teardown. It provides
 * the shared state bus that panels and the refresh scheduler read/write.
 */
export interface GeoPolAppContext {
  /** Root DOM container (#app element) */
  readonly container: HTMLElement;

  /** Whether the app has been torn down */
  isDestroyed: boolean;

  /** In-flight request tracker (prevents duplicate refreshes) */
  inFlight: Set<string>;

  /** Panel registry: panelId -> Panel instance */
  panels: Record<string, Panel>;

  /** Refresh scheduler for periodic data polling */
  scheduler: RefreshScheduler | null;

  /** Destroy all registered modules and mark context as dead */
  destroy(): void;
}

/**
 * Create the global app context.
 *
 * @param container The #app root element
 * @returns Initialized GeoPolAppContext
 */
export function createAppContext(container: HTMLElement): GeoPolAppContext {
  const ctx: GeoPolAppContext = {
    container,
    isDestroyed: false,
    inFlight: new Set(),
    panels: {},
    scheduler: null,

    destroy() {
      if (ctx.isDestroyed) return;
      ctx.isDestroyed = true;

      // Destroy all panels
      for (const panel of Object.values(ctx.panels)) {
        panel.destroy();
      }
      ctx.panels = {};

      // Destroy scheduler
      if (ctx.scheduler) {
        ctx.scheduler.destroy();
        ctx.scheduler = null;
      }

      ctx.inFlight.clear();
    },
  };

  return ctx;
}
