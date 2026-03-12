/**
 * main.ts -- Geopol SPA bootstrap.
 *
 * Router-driven lifecycle:
 *  1. Apply dark theme
 *  2. Create Router + NavBar
 *  3. Register 3 screen routes (/dashboard, /globe, /forecasts)
 *  4. Resolve initial screen from current URL
 *  5. Visibility-aware background throttling
 */

import { applyStoredTheme } from '@/utils/theme-manager';
import { createAppContext } from '@/app/app-context';
import { Router } from '@/app/router';
import { createNavBar } from '@/components/NavBar';
import { h } from '@/utils/dom-utils';

// Screen modules
import { mountDashboard, unmountDashboard } from '@/screens/dashboard-screen';
import { mountGlobe, unmountGlobe } from '@/screens/globe-screen';
import { mountForecasts, unmountForecasts } from '@/screens/forecasts-screen';
import { mountAdmin, unmountAdmin } from '@/screens/admin-screen';

// Styles (CesiumJS widgets loaded in CesiumMap, not here)
import '@/styles/main.css';
import '@/styles/panels.css';

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

async function boot(): Promise<void> {
  applyStoredTheme();

  const app = document.getElementById('app');
  if (!app) throw new Error('Missing #app container');

  const ctx = createAppContext(app);

  // -- Layout: NavBar + screen container --
  app.innerHTML = '';

  const screenContainer = h('div', { className: 'screen-container' });

  const router = new Router(screenContainer);
  const navBar = createNavBar(router);

  app.appendChild(navBar);
  app.appendChild(screenContainer);

  // -- Register routes --
  router.addRoute({
    path: '/dashboard',
    mount: (container) => mountDashboard(container, ctx),
    unmount: () => unmountDashboard(ctx),
  });

  router.addRoute({
    path: '/globe',
    mount: (container) => mountGlobe(container, ctx),
    unmount: () => unmountGlobe(ctx),
  });

  router.addRoute({
    path: '/forecasts',
    mount: (container) => mountForecasts(container, ctx),
    unmount: () => unmountForecasts(ctx),
  });

  router.addRoute({
    path: '/admin',
    mount: (container) => mountAdmin(container),
    unmount: () => unmountAdmin(),
  });

  // -- Resolve initial screen based on current URL --
  await router.resolve();

  // -- Visibility-aware background throttling --
  document.addEventListener('visibilitychange', () => {
    if (!ctx.scheduler) return;
    if (document.visibilityState === 'hidden') {
      ctx.scheduler.setHiddenSince(Date.now());
    } else {
      ctx.scheduler.flushStaleRefreshes();
    }
  });

  // -- Debug access --
  if (import.meta.env.DEV) {
    (window as unknown as Record<string, unknown>)['__geopol'] = { ctx, router };
  }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

boot().catch((err) => {
  console.error('[Geopol] Boot failed:', err);
  const app = document.getElementById('app');
  if (app) {
    app.innerHTML = `
      <div style="
        display: flex; align-items: center; justify-content: center;
        height: 100vh; color: #e63946; font-family: monospace;
        font-size: 14px; text-align: center; padding: 20px;
      ">
        GEOPOL BOOT FAILURE<br/><br/>
        ${String(err)}
      </div>
    `;
  }
});
