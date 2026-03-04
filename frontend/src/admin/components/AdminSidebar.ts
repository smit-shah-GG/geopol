/**
 * AdminSidebar -- vertical navigation for admin sections.
 *
 * Four sections: Processes, Config, Logs, Sources.
 * Active item gets a red accent background via `.active` class.
 * Calls back to the layout on section change.
 */

import { h } from '@/utils/dom-utils';
import type { AdminSection } from '@/admin/admin-types';

interface NavItem {
  section: AdminSection;
  label: string;
  icon: string; // unicode character
}

const NAV_ITEMS: NavItem[] = [
  { section: 'processes', label: 'Processes', icon: '\u2699' },  // gear
  { section: 'config',    label: 'Config',    icon: '\u2692' },  // hammer and pick
  { section: 'logs',      label: 'Logs',      icon: '\u2263' },  // triple bar
  { section: 'sources',   label: 'Sources',   icon: '\u26A1' },  // lightning
];

export class AdminSidebar {
  private readonly el: HTMLElement;
  private readonly items: Map<AdminSection, HTMLElement> = new Map();
  private activeSection: AdminSection | null = null;

  constructor(private readonly onNavigate: (section: AdminSection) => void) {
    this.el = h('nav', { className: 'admin-sidebar' });

    const title = h('div', { className: 'sidebar-title' }, 'GEOPOL ADMIN');
    this.el.appendChild(title);

    for (const item of NAV_ITEMS) {
      const el = h('div', { className: 'nav-item', dataset: { section: item.section } },
        h('span', { className: 'nav-icon' }, item.icon),
        h('span', { className: 'nav-label' }, item.label),
      );

      el.addEventListener('click', () => {
        this.setActive(item.section);
        this.onNavigate(item.section);
      });

      this.items.set(item.section, el);
      this.el.appendChild(el);
    }
  }

  /** Update active state visually. */
  setActive(section: AdminSection): void {
    if (this.activeSection === section) return;
    this.activeSection = section;

    for (const [s, el] of this.items) {
      el.classList.toggle('active', s === section);
    }
  }

  /** Return the sidebar DOM element. */
  getElement(): HTMLElement {
    return this.el;
  }

  /** Clean up event listeners. */
  destroy(): void {
    this.el.remove();
  }
}
