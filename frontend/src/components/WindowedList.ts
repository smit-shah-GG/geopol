/**
 * WindowedList<T> -- Chunk-based windowed rendering for variable-height items.
 *
 * Ported from World Monitor's VirtualList.ts (WindowedList class).
 * Renders items in chunks (default 10), only materializing chunks within
 * the viewport plus a configurable buffer. Uses CSS containment for
 * paint isolation and requestAnimationFrame-throttled scroll handling.
 *
 * Design rationale: Variable-height items (article clusters with varying
 * snippet length) make fixed-height virtual scrolling impractical. Chunk-based
 * windowed rendering provides the 80/20 of virtualization without requiring
 * height measurement or ResizeObserver per item.
 */

export interface WindowedListOptions {
  /** Container element (must have overflow-y: auto) */
  container: HTMLElement;
  /** Number of items per chunk (default: 10) */
  chunkSize?: number;
  /** Buffer chunks above/below viewport (default: 1) */
  bufferChunks?: number;
}

export class WindowedList<T> {
  private container: HTMLElement;
  private chunkSize: number;
  private bufferChunks: number;
  private items: T[] = [];
  private renderItem: (item: T, index: number) => string;
  private onRendered?: () => void;

  private renderedChunks = new Set<number>();
  private chunkElements = new Map<number, HTMLElement>();
  private scrollRAF: number | null = null;

  constructor(
    options: WindowedListOptions,
    renderItem: (item: T, index: number) => string,
    onRendered?: () => void,
  ) {
    this.container = options.container;
    this.chunkSize = options.chunkSize ?? 10;
    this.bufferChunks = options.bufferChunks ?? 1;
    this.renderItem = renderItem;
    this.onRendered = onRendered;

    this.container.classList.add('windowed-list');
    this.container.addEventListener('scroll', this.handleScroll, { passive: true });
  }

  /**
   * Replace all items and re-render visible chunks.
   * Previous DOM content is cleared.
   */
  setItems(items: T[]): void {
    this.items = items;
    this.renderedChunks.clear();

    // Remove existing chunk elements
    for (const el of this.chunkElements.values()) {
      el.remove();
    }
    this.chunkElements.clear();

    // Clear container
    this.container.innerHTML = '';

    if (items.length === 0) {
      return;
    }

    // Create placeholder div for each chunk
    const totalChunks = Math.ceil(items.length / this.chunkSize);
    for (let i = 0; i < totalChunks; i++) {
      const placeholder = document.createElement('div');
      placeholder.className = 'windowed-chunk';
      placeholder.dataset['chunk'] = String(i);
      this.container.appendChild(placeholder);
      this.chunkElements.set(i, placeholder);
    }

    // Render chunks currently in viewport
    this.updateVisibleChunks();
  }

  /**
   * Force re-render of currently visible chunks (e.g., after data change).
   */
  refresh(): void {
    const visibleChunks = this.getVisibleChunks();
    for (const chunkIndex of visibleChunks) {
      this.renderChunk(chunkIndex);
    }
    this.onRendered?.();
  }

  /**
   * Clean up scroll listener and internal state.
   */
  destroy(): void {
    if (this.scrollRAF !== null) {
      cancelAnimationFrame(this.scrollRAF);
      this.scrollRAF = null;
    }
    this.container.removeEventListener('scroll', this.handleScroll);
    this.chunkElements.clear();
    this.renderedChunks.clear();
    this.items = [];
  }

  // ---------------------------------------------------------------------------
  // Private
  // ---------------------------------------------------------------------------

  private handleScroll = (): void => {
    if (this.scrollRAF !== null) return;
    this.scrollRAF = requestAnimationFrame(() => {
      this.scrollRAF = null;
      this.updateVisibleChunks();
    });
  };

  private getVisibleChunks(): number[] {
    const scrollTop = this.container.scrollTop;
    const viewportHeight = this.container.clientHeight;
    const chunks: number[] = [];

    for (const [index, element] of this.chunkElements) {
      const rect = element.getBoundingClientRect();
      const containerRect = this.container.getBoundingClientRect();
      const relativeTop = rect.top - containerRect.top + scrollTop;
      const relativeBottom = relativeTop + rect.height;

      // Check if chunk is within viewport (with buffer)
      const bufferPx = viewportHeight * this.bufferChunks;
      if (
        relativeBottom >= scrollTop - bufferPx &&
        relativeTop <= scrollTop + viewportHeight + bufferPx
      ) {
        chunks.push(index);
      }
    }

    return chunks;
  }

  private updateVisibleChunks(): void {
    const visibleChunks = this.getVisibleChunks();

    let needsCallback = false;
    for (const chunkIndex of visibleChunks) {
      if (!this.renderedChunks.has(chunkIndex)) {
        this.renderChunk(chunkIndex);
        needsCallback = true;
      }
    }

    if (needsCallback) {
      this.onRendered?.();
    }
  }

  private renderChunk(chunkIndex: number): void {
    const element = this.chunkElements.get(chunkIndex);
    if (!element) return;

    const startIdx = chunkIndex * this.chunkSize;
    const endIdx = Math.min(startIdx + this.chunkSize, this.items.length);
    const chunkItems = this.items.slice(startIdx, endIdx);

    const html = chunkItems
      .map((item, i) => this.renderItem(item, startIdx + i))
      .join('');

    element.innerHTML = html;
    element.classList.add('rendered');
    this.renderedChunks.add(chunkIndex);
  }
}
