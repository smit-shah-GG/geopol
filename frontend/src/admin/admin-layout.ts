/**
 * Admin layout builder -- two-column grid with sidebar nav + content area.
 *
 * Creates the admin shell that Plan 03's panels plug into.
 * For now, each section renders a placeholder div.
 */

import { h } from '@/utils/dom-utils';
import { AdminSidebar } from '@/admin/components/AdminSidebar';
import type { AdminSection } from '@/admin/admin-types';

export interface AdminLayout {
  /** The admin key for panel API calls. */
  adminKey: string;
  destroy(): void;
}

const SECTION_TITLES: Record<AdminSection, string> = {
  processes: 'PROCESSES',
  config: 'CONFIGURATION',
  logs: 'SYSTEM LOGS',
  sources: 'DATA SOURCES',
};

/**
 * Build the admin layout inside the given container.
 *
 * Returns an AdminLayout handle for cleanup on route unmount.
 */
export function createAdminLayout(
  container: HTMLElement,
  adminKey: string,
): AdminLayout {
  const wrapper = h('div', { className: 'admin-layout' });
  const contentArea = h('div', { className: 'admin-content' });

  let currentSection: AdminSection | null = null;

  const sidebar = new AdminSidebar((section: AdminSection) => {
    if (section === currentSection) return;
    currentSection = section;
    renderSection(section);
  });

  wrapper.appendChild(sidebar.getElement());
  wrapper.appendChild(contentArea);
  container.appendChild(wrapper);

  // Default to processes
  sidebar.setActive('processes');
  currentSection = 'processes';
  renderSection('processes');

  function renderSection(section: AdminSection): void {
    contentArea.innerHTML = '';

    const header = h('div', { className: 'admin-header' }, SECTION_TITLES[section]);
    const placeholder = h('div', { className: 'section-placeholder' },
      `${section.toUpperCase()} panel -- coming in Plan 03`,
    );

    contentArea.appendChild(header);
    contentArea.appendChild(placeholder);
  }

  return {
    adminKey,
    destroy(): void {
      sidebar.destroy();
      wrapper.remove();
    },
  };
}
