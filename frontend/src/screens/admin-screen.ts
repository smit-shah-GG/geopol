/**
 * Admin screen -- auth gate + dynamic import of admin layout.
 *
 * This file is statically imported in main.ts but is intentionally tiny.
 * All heavy admin code (layout, panels, styles) is dynamically imported
 * inside mountAdmin() to ensure code splitting -- zero admin bytes in
 * the public bundle.
 *
 * Auth flow:
 *   1. Check sessionStorage for stored admin key
 *   2. If found, validate via AdminClient.verify()
 *   3. If missing or invalid, show AuthModal overlay
 *   4. On success, dynamically import admin layout + styles
 *   5. Render two-column admin shell
 */

import type { AdminLayout } from '@/admin/admin-layout';

const SESSION_KEY = 'geopol_admin_key';

let layout: AdminLayout | null = null;

export async function mountAdmin(container: HTMLElement): Promise<void> {
  let adminKey: string | null = null;

  // 1. Check sessionStorage for persisted key
  const storedKey = sessionStorage.getItem(SESSION_KEY);
  if (storedKey) {
    // Validate stored key before trusting it
    try {
      const { AdminClient } = await import('@/admin/admin-client');
      const client = new AdminClient(storedKey);
      await client.verify();
      adminKey = storedKey;
    } catch {
      // Stored key is stale or invalid -- clear it
      sessionStorage.removeItem(SESSION_KEY);
    }
  }

  // 2. If no valid key, show auth modal (load styles FIRST so modal is styled)
  if (!adminKey) {
    await import('@/admin/admin-styles.css');
    const { AuthModal } = await import('@/admin/components/AuthModal');
    const modal = new AuthModal();
    adminKey = await modal.waitForAuth();
  } else {
    // Key was valid from sessionStorage — still need styles for the layout
    await import('@/admin/admin-styles.css');
  }

  // 3. Dynamically import admin layout (code splitting boundary)
  const { createAdminLayout } = await import('@/admin/admin-layout');

  // 4. Create the admin layout
  layout = createAdminLayout(container, adminKey);
}

export function unmountAdmin(): void {
  if (layout) {
    layout.destroy();
    layout = null;
  }

  // Remove dynamically injected admin stylesheet to prevent CSS bleed
  const adminLinks = document.querySelectorAll<HTMLLinkElement>(
    'link[rel="stylesheet"][href*="admin"], style[data-vite-dev-id*="admin"]',
  );
  for (const link of adminLinks) {
    link.remove();
  }
}
