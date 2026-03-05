/**
 * ConfigEditor -- grouped runtime configuration form with inline validation,
 * dangerous-change confirmation, and Revert to Defaults.
 *
 * Settings grouped by prefix (GDELT, RSS, Gemini, Polymarket, etc.).
 * Each field validates on blur, saves only changed fields, and prompts
 * a confirmation dialog for dangerous changes.
 */

import { h, clearChildren } from '@/utils/dom-utils';
import { showToast } from '@/admin/admin-toast';
import type { AdminClient } from '@/admin/admin-client';
import type { ConfigEntry } from '@/admin/admin-types';
import type { AdminPanel } from '@/admin/panels/ProcessTable';

/** Groups for organizing config keys by prefix. */
const GROUP_PREFIXES: [string, string][] = [
  ['gdelt_', 'GDELT'],
  ['rss_', 'RSS'],
  ['gemini_', 'Gemini'],
  ['polymarket_', 'Polymarket'],
  ['acled_', 'ACLED'],
  ['calibration_', 'Calibration'],
  ['smtp_', 'Monitoring'],
  ['alert_', 'Monitoring'],
  ['feed_', 'Monitoring'],
  ['drift_', 'Monitoring'],
  ['disk_', 'Monitoring'],
];

function groupName(key: string): string {
  for (const [prefix, name] of GROUP_PREFIXES) {
    if (key.startsWith(prefix)) return name;
  }
  return 'Other';
}

/** Convert "gdelt_poll_interval" to "Gdelt Poll Interval". */
function humanize(key: string): string {
  return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

/** Cast a form value back to the appropriate type. */
function castValue(raw: string, type: string): unknown {
  switch (type) {
    case 'int': return parseInt(raw, 10);
    case 'float': return parseFloat(raw);
    case 'bool': return raw === 'true';
    case 'list': {
      try { return JSON.parse(raw) as unknown; }
      catch { return raw; }
    }
    default: return raw;
  }
}

/** Validate a raw string against type constraints. Returns error msg or null. */
function validate(raw: string, type: string): string | null {
  switch (type) {
    case 'int': {
      if (!/^-?\d+$/.test(raw.trim())) return 'Must be an integer';
      return null;
    }
    case 'float': {
      if (isNaN(parseFloat(raw.trim()))) return 'Must be a number';
      return null;
    }
    case 'list': {
      try { JSON.parse(raw); return null; }
      catch { return 'Must be valid JSON array'; }
    }
    default:
      return null;
  }
}

export class ConfigEditor implements AdminPanel {
  private el: HTMLElement | null = null;
  private original: Map<string, unknown> = new Map();
  private current: Map<string, string> = new Map();
  private errors: Map<string, string> = new Map();
  private entries: ConfigEntry[] = [];
  private saveBtn: HTMLButtonElement | null = null;
  private formArea: HTMLElement | null = null;

  constructor(private readonly client: AdminClient) {}

  async mount(container: HTMLElement): Promise<void> {
    this.el = h('div', { className: 'config-editor' });
    container.appendChild(this.el);
    await this.load();
  }

  destroy(): void {
    this.el?.remove();
    this.el = null;
    this.formArea = null;
    this.saveBtn = null;
  }

  private async load(): Promise<void> {
    if (!this.el) return;
    try {
      this.entries = await this.client.getConfig();
    } catch {
      this.el.textContent = 'Failed to load configuration.';
      return;
    }

    this.original.clear();
    this.current.clear();
    this.errors.clear();
    for (const e of this.entries) {
      const strVal = e.type === 'list' ? JSON.stringify(e.value) : String(e.value ?? '');
      this.original.set(e.key, e.value);
      this.current.set(e.key, strVal);
    }

    this.render();
  }

  private render(): void {
    if (!this.el) return;
    clearChildren(this.el);

    this.formArea = h('div', { className: 'config-form' });

    // Group entries
    const groups = new Map<string, ConfigEntry[]>();
    for (const e of this.entries) {
      const g = groupName(e.key);
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g)!.push(e);
    }

    for (const [name, entries] of groups) {
      this.formArea.appendChild(this.renderGroup(name, entries));
    }

    // Buttons
    const btnRow = h('div', { className: 'config-btn-row' });

    this.saveBtn = h('button', {
      className: 'save-btn',
      disabled: true,
    }, 'SAVE CHANGES') as HTMLButtonElement;
    this.saveBtn.addEventListener('click', () => { void this.handleSave(); });

    const revertBtn = h('button', { className: 'revert-btn' }, 'REVERT TO DEFAULTS') as HTMLButtonElement;
    revertBtn.addEventListener('click', () => { void this.handleRevert(); });

    btnRow.appendChild(this.saveBtn);
    btnRow.appendChild(revertBtn);

    this.el.appendChild(this.formArea);
    this.el.appendChild(btnRow);
  }

  private renderGroup(name: string, entries: ConfigEntry[]): HTMLElement {
    const group = h('div', { className: 'config-group' });
    const header = h('div', { className: 'config-group-header' }, name);
    const body = h('div', { className: 'config-group-body' });

    let collapsed = false;
    header.addEventListener('click', () => {
      collapsed = !collapsed;
      body.style.display = collapsed ? 'none' : '';
      header.classList.toggle('collapsed', collapsed);
    });

    for (const entry of entries) {
      body.appendChild(this.renderField(entry));
    }

    group.appendChild(header);
    group.appendChild(body);
    return group;
  }

  private renderField(entry: ConfigEntry): HTMLElement {
    const cls = [
      'config-field',
      entry.dangerous ? 'dangerous' : '',
      this.errors.has(entry.key) ? 'invalid' : '',
    ].filter(Boolean).join(' ');

    const field = h('div', { className: cls, dataset: { key: entry.key } });
    const label = h('label', { className: 'config-label' }, humanize(entry.key));
    const desc = h('div', { className: 'config-desc' }, entry.description || '');

    field.appendChild(label);

    if (!entry.editable) {
      // Read-only / secret -- show masked
      const readOnly = h('div', { className: 'config-readonly' }, '********');
      field.appendChild(readOnly);
    } else {
      const input = this.createInput(entry);
      field.appendChild(input);
    }

    field.appendChild(desc);

    // Error slot
    const errorEl = h('div', { className: 'config-error' },
      this.errors.get(entry.key) ?? '',
    );
    field.appendChild(errorEl);

    return field;
  }

  private createInput(entry: ConfigEntry): HTMLElement {
    const currentVal = this.current.get(entry.key) ?? '';

    if (entry.type === 'bool') {
      const wrapper = h('label', { className: 'config-checkbox-label' });
      const cb = h('input', {
        type: 'checkbox',
        className: 'config-checkbox',
      }) as HTMLInputElement;
      cb.checked = currentVal === 'true';
      cb.addEventListener('change', () => {
        this.current.set(entry.key, cb.checked ? 'true' : 'false');
        this.errors.delete(entry.key);
        this.updateSaveState();
      });
      wrapper.appendChild(cb);
      wrapper.appendChild(document.createTextNode(cb.checked ? 'Enabled' : 'Disabled'));
      return wrapper;
    }

    if (entry.type === 'list') {
      const ta = h('textarea', {
        className: 'config-input config-textarea',
        rows: '3',
      }) as HTMLTextAreaElement;
      ta.value = currentVal;
      ta.addEventListener('blur', () => this.onFieldBlur(entry.key, ta.value, entry.type));
      ta.addEventListener('input', () => {
        this.current.set(entry.key, ta.value);
        this.updateSaveState();
      });
      return ta;
    }

    const inputType = entry.type === 'int' || entry.type === 'float' ? 'number' : 'text';
    const input = h('input', {
      type: inputType,
      className: 'config-input',
      value: currentVal,
    }) as HTMLInputElement;

    if (entry.type === 'int') input.step = '1';
    if (entry.type === 'float') input.step = 'any';

    input.addEventListener('blur', () => this.onFieldBlur(entry.key, input.value, entry.type));
    input.addEventListener('input', () => {
      this.current.set(entry.key, input.value);
      this.updateSaveState();
    });

    return input;
  }

  private onFieldBlur(key: string, raw: string, type: string): void {
    const err = validate(raw, type);
    if (err) {
      this.errors.set(key, err);
    } else {
      this.errors.delete(key);
    }
    // Re-render the field error indicator
    const fieldEl = this.formArea?.querySelector(`[data-key="${key}"]`);
    if (fieldEl) {
      fieldEl.classList.toggle('invalid', !!err);
      const errorEl = fieldEl.querySelector('.config-error');
      if (errorEl) errorEl.textContent = err ?? '';
    }
    this.updateSaveState();
  }

  private getChanges(): Record<string, unknown> {
    const changes: Record<string, unknown> = {};
    for (const entry of this.entries) {
      if (!entry.editable) continue;
      const cur = this.current.get(entry.key);
      const origStr = entry.type === 'list'
        ? JSON.stringify(this.original.get(entry.key))
        : String(this.original.get(entry.key) ?? '');
      if (cur !== origStr) {
        changes[entry.key] = castValue(cur ?? '', entry.type);
      }
    }
    return changes;
  }

  private updateSaveState(): void {
    if (!this.saveBtn) return;
    const changes = this.getChanges();
    const hasChanges = Object.keys(changes).length > 0;
    const hasErrors = this.errors.size > 0;
    this.saveBtn.disabled = !hasChanges || hasErrors;
  }

  private async handleSave(): Promise<void> {
    const changes = this.getChanges();
    if (Object.keys(changes).length === 0) return;

    // Check for dangerous fields
    const dangerousKeys = Object.keys(changes).filter(k =>
      this.entries.find(e => e.key === k)?.dangerous,
    );

    if (dangerousKeys.length > 0) {
      const confirmed = await this.showConfirm(
        `Changing dangerous settings: ${dangerousKeys.join(', ')}. This may affect system stability. Continue?`,
      );
      if (!confirmed) return;
    }

    try {
      await this.client.updateConfig(changes);
      showToast('Configuration saved');
      await this.load(); // Re-fetch to show persisted state
    } catch {
      showToast('Failed to save configuration', true);
    }
  }

  private async handleRevert(): Promise<void> {
    const confirmed = await this.showConfirm(
      'This will reset ALL runtime config overrides to their default values. Continue?',
    );
    if (!confirmed) return;

    try {
      await this.client.resetConfig();
      showToast('Configuration reset to defaults');
      await this.load();
    } catch {
      showToast('Failed to reset configuration', true);
    }
  }

  private showConfirm(message: string): Promise<boolean> {
    return new Promise(resolve => {
      const overlay = h('div', { className: 'confirm-dialog' });
      const card = h('div', { className: 'confirm-card' });
      const msg = h('div', { className: 'confirm-message' }, message);
      const btnRow = h('div', { className: 'confirm-btns' });

      const cancelBtn = h('button', { className: 'confirm-cancel' }, 'CANCEL');
      const confirmBtn = h('button', { className: 'confirm-ok' }, 'CONFIRM');

      cancelBtn.addEventListener('click', () => { overlay.remove(); resolve(false); });
      confirmBtn.addEventListener('click', () => { overlay.remove(); resolve(true); });

      btnRow.appendChild(cancelBtn);
      btnRow.appendChild(confirmBtn);
      card.appendChild(msg);
      card.appendChild(btnRow);
      overlay.appendChild(card);
      document.body.appendChild(overlay);
    });
  }
}

