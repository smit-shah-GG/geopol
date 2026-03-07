/**
 * Admin layout builder -- two-column grid with sidebar nav + content area.
 *
 * Creates the admin shell and wires section navigation to real panels:
 * ProcessTable, ConfigEditor, LogViewer, SourceManager.
 * On section change, the current panel is destroyed and the new one mounted.
 */

import { h } from '@/utils/dom-utils';
import { AdminClient } from '@/admin/admin-client';
import { AdminSidebar } from '@/admin/components/AdminSidebar';
import { ProcessTable } from '@/admin/panels/ProcessTable';
import { ConfigEditor } from '@/admin/panels/ConfigEditor';
import { LogViewer } from '@/admin/panels/LogViewer';
import { SourceManager } from '@/admin/panels/SourceManager';
import { AccuracyPanel } from '@/admin/panels/AccuracyPanel';
import { BacktestingPanel } from '@/admin/panels/BacktestingPanel';
import type { AdminSection } from '@/admin/admin-types';
import type { AdminPanel } from '@/admin/panels/ProcessTable';

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
  accuracy: 'POLYMARKET ACCURACY',
  backtesting: 'HISTORICAL BACKTESTING',
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
  const client = new AdminClient(adminKey);

  let currentSection: AdminSection | null = null;
  let currentPanel: AdminPanel | null = null;

  const sidebar = new AdminSidebar((section: AdminSection) => {
    if (section === currentSection) return;
    currentSection = section;
    void renderSection(section);
  });

  wrapper.appendChild(sidebar.getElement());
  wrapper.appendChild(contentArea);
  container.appendChild(wrapper);

  // Default to processes
  sidebar.setActive('processes');
  currentSection = 'processes';
  void renderSection('processes');

  async function renderSection(section: AdminSection): Promise<void> {
    // Destroy previous panel
    if (currentPanel) {
      currentPanel.destroy();
      currentPanel = null;
    }
    contentArea.innerHTML = '';

    const header = h('div', { className: 'admin-header' }, SECTION_TITLES[section]);
    contentArea.appendChild(header);

    // Instantiate the panel for this section
    const panel = createPanel(section, client);
    currentPanel = panel;
    await panel.mount(contentArea);
  }

  return {
    adminKey,
    destroy(): void {
      if (currentPanel) {
        currentPanel.destroy();
        currentPanel = null;
      }
      sidebar.destroy();
      wrapper.remove();
    },
  };
}

function createPanel(section: AdminSection, client: AdminClient): AdminPanel {
  switch (section) {
    case 'processes': return new ProcessTable(client);
    case 'config': return new ConfigEditor(client);
    case 'logs': return new LogViewer(client);
    case 'sources': return new SourceManager(client);
    case 'accuracy': return new AccuracyPanel(client);
    case 'backtesting': return new BacktestingPanel(client);
  }
}
