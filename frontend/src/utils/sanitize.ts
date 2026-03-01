const ENTITY_MAP: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
};

/**
 * Escape HTML special characters to prevent XSS when injecting user-controlled
 * strings into innerHTML. Prefer DOM text nodes (h() helper) over this when possible.
 */
export function escapeHtml(str: string): string {
  return str.replace(/[&<>"']/g, (ch) => ENTITY_MAP[ch] ?? ch);
}

/**
 * Sanitize a URL to prevent javascript:, data:, and vbscript: protocol injection.
 * Returns the URL unchanged if safe, empty string if dangerous.
 *
 * Allowed protocols: http, https, mailto, tel, relative paths (/, #, ?).
 */
export function sanitizeUrl(url: string): string {
  const trimmed = url.trim();
  if (trimmed === '') return '';

  // Relative paths and fragment/query-only URLs are safe
  if (trimmed.startsWith('/') || trimmed.startsWith('#') || trimmed.startsWith('?')) {
    return trimmed;
  }

  // Check for allowed absolute protocols
  if (/^https?:\/\//i.test(trimmed)) return trimmed;
  if (/^mailto:/i.test(trimmed)) return trimmed;
  if (/^tel:/i.test(trimmed)) return trimmed;

  // Everything else (javascript:, data:, vbscript:, etc.) is blocked
  return '';
}
