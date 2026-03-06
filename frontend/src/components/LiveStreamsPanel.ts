/**
 * LiveStreamsPanel -- YouTube live stream embeds with region filtering,
 * idle detection, and exclusive-unmute controls.
 *
 * Displays 15-20 curated geopolitical news live streams in a 2-column grid.
 * Region pills filter visible channels. Idle detection pauses all players
 * after 5 minutes of inactivity. Only one player can be unmuted at a time.
 *
 * YouTube IFrame API loaded lazily via dynamic script injection with a
 * Promise wrapper to handle the async onYouTubeIframeAPIReady callback.
 */

import { Panel } from './Panel';
import { h, replaceChildren } from '@/utils/dom-utils';

// ---------------------------------------------------------------------------
// YouTube IFrame API types
// ---------------------------------------------------------------------------

interface YTPlayerInstance {
  playVideo(): void;
  pauseVideo(): void;
  mute(): void;
  unMute(): void;
  isMuted(): boolean;
  destroy(): void;
  getIframe(): HTMLIFrameElement;
}

interface YTPlayerOptions {
  videoId: string;
  host?: string;
  playerVars: Record<string, number | string>;
  events: {
    onReady?: () => void;
    onError?: (event: { data: number }) => void;
    onStateChange?: (event: { data: number }) => void;
  };
}

interface YTNamespace {
  Player: new (el: string | HTMLElement, opts: YTPlayerOptions) => YTPlayerInstance;
  PlayerState: {
    ENDED: number;
    PLAYING: number;
    PAUSED: number;
    BUFFERING: number;
    CUED: number;
  };
}

declare global {
  interface Window {
    YT?: YTNamespace;
    onYouTubeIframeAPIReady?: () => void;
  }
}

// ---------------------------------------------------------------------------
// Channel data
// ---------------------------------------------------------------------------

export interface LiveChannel {
  id: string;
  name: string;
  youtubeId: string;
  region: 'Americas' | 'Europe' | 'Middle East' | 'Asia-Pacific' | 'Africa' | 'Global';
  language?: string;
}

/**
 * Curated list of geopolitical news channels with known live stream video IDs.
 *
 * YouTube live stream IDs are the video IDs of each channel's persistent live
 * stream URL. These are well-known public IDs that rarely change.
 */
const CHANNELS: LiveChannel[] = [
  // Global / Wire
  { id: 'aljazeera', name: 'Al Jazeera English', youtubeId: 'gCNeDWCI0vo', region: 'Middle East', language: 'en' },
  { id: 'bbc-news', name: 'BBC News', youtubeId: 'dp8PhLsUcFE', region: 'Europe', language: 'en' },
  { id: 'cnn', name: 'CNN', youtubeId: 'oJUvTVdTMyY', region: 'Americas', language: 'en' },
  { id: 'sky-news', name: 'Sky News', youtubeId: '9Auq9mYxFEE', region: 'Europe', language: 'en' },
  { id: 'france24', name: 'France 24 English', youtubeId: 'h3MuIUNCCzI', region: 'Europe', language: 'en' },
  { id: 'dw-news', name: 'DW News', youtubeId: 'GE_SfNVNyqk', region: 'Europe', language: 'en' },
  { id: 'wion', name: 'WION', youtubeId: 'A6Ac0BRKK6w', region: 'Asia-Pacific', language: 'en' },
  { id: 'trt-world', name: 'TRT World', youtubeId: 'CV5Fooi8YJI', region: 'Middle East', language: 'en' },
  { id: 'nhk-world', name: 'NHK World', youtubeId: 'f0ldnpPTZbA', region: 'Asia-Pacific', language: 'en' },
  { id: 'i24news', name: 'i24NEWS', youtubeId: 'wJv48EFkVCE', region: 'Middle East', language: 'en' },
  { id: 'euronews', name: 'Euronews', youtubeId: 'pBMJnorhi7Q', region: 'Europe', language: 'en' },
  { id: 'cna', name: 'CNA', youtubeId: 'XWq5kBlakcQ', region: 'Asia-Pacific', language: 'en' },
  { id: 'abc-live', name: 'ABC News Live', youtubeId: 'YMv06_y6XjY', region: 'Americas', language: 'en' },
  { id: 'msnbc', name: 'MSNBC', youtubeId: 'ewDPNMtHLF0', region: 'Americas', language: 'en' },
  { id: 'reuters-tv', name: 'Reuters TV', youtubeId: 'u3Tay-R7Z4U', region: 'Global', language: 'en' },
  { id: 'cgtn', name: 'CGTN', youtubeId: 'eZBxEAlYjlE', region: 'Asia-Pacific', language: 'en' },
];

type RegionFilter = 'All' | 'Americas' | 'Europe' | 'Middle East' | 'Asia-Pacific' | 'Africa';

const REGION_FILTERS: RegionFilter[] = ['All', 'Americas', 'Europe', 'Middle East', 'Asia-Pacific', 'Africa'];

// ---------------------------------------------------------------------------
// YouTube API loader (Promise wrapper)
// ---------------------------------------------------------------------------

let ytApiPromise: Promise<void> | null = null;

/**
 * Load the YouTube IFrame API asynchronously.
 * Sets window.onYouTubeIframeAPIReady BEFORE injecting the script tag
 * to prevent the race condition where the callback fires before we listen.
 * Returns immediately if the API is already loaded.
 */
function loadYouTubeAPI(): Promise<void> {
  if (window.YT?.Player) {
    return Promise.resolve();
  }

  if (ytApiPromise) {
    return ytApiPromise;
  }

  ytApiPromise = new Promise<void>((resolve) => {
    // Set callback BEFORE script injection -- critical ordering
    window.onYouTubeIframeAPIReady = () => {
      resolve();
    };

    const script = document.createElement('script');
    script.src = 'https://www.youtube.com/iframe_api';
    script.async = true;
    document.head.appendChild(script);
  });

  return ytApiPromise;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Idle timeout: 5 minutes of no user interaction. */
const IDLE_TIMEOUT_MS = 300_000;

/** Events that reset the idle timer. */
const ACTIVITY_EVENTS: (keyof DocumentEventMap)[] = ['mousemove', 'keypress', 'click', 'scroll'];

// ---------------------------------------------------------------------------
// LiveStreamsPanel
// ---------------------------------------------------------------------------

export class LiveStreamsPanel extends Panel {
  private regionFilter: RegionFilter = 'All';
  private pillContainer: HTMLElement;
  private gridContainer: HTMLElement;
  private players: Map<string, YTPlayerInstance> = new Map();
  private unmutedPlayerId: string | null = null;
  private idleTimer: ReturnType<typeof setTimeout> | null = null;
  private isIdle = false;
  private activityHandler: (() => void) | null = null;

  constructor() {
    super({ id: 'live-streams', title: 'LIVE STREAMS', showCount: false });

    // Region pill bar
    this.pillContainer = document.createElement('div');
    this.pillContainer.className = 'live-streams-pills';
    this.element.insertBefore(this.pillContainer, this.content);
    this.buildRegionPills();

    // Grid container for players
    this.gridContainer = this.content;
    this.gridContainer.classList.add('live-streams-grid');

    // Remove loading state -- we render immediately
    replaceChildren(this.gridContainer);

    // Setup idle detection
    this.setupIdleDetection();

    // Load API and render players
    this.initPlayers();
  }

  // -------------------------------------------------------------------------
  // Region pills
  // -------------------------------------------------------------------------

  private buildRegionPills(): void {
    for (const region of REGION_FILTERS) {
      const pill = document.createElement('button');
      pill.className = `news-feed-pill${region === 'All' ? ' active' : ''}`;
      pill.textContent = region.toUpperCase();
      pill.addEventListener('click', () => {
        this.regionFilter = region;
        for (const btn of this.pillContainer.querySelectorAll('.news-feed-pill')) {
          btn.classList.toggle('active', btn === pill);
        }
        this.rebuildPlayers();
      });
      this.pillContainer.appendChild(pill);
    }
  }

  // -------------------------------------------------------------------------
  // Filtered channels
  // -------------------------------------------------------------------------

  private getFilteredChannels(): LiveChannel[] {
    if (this.regionFilter === 'All') return CHANNELS;
    return CHANNELS.filter(
      (ch) => ch.region === this.regionFilter || ch.region === 'Global',
    );
  }

  // -------------------------------------------------------------------------
  // Player lifecycle
  // -------------------------------------------------------------------------

  private async initPlayers(): Promise<void> {
    try {
      await loadYouTubeAPI();
      this.rebuildPlayers();
    } catch (err) {
      console.error('[LiveStreamsPanel] YouTube API load failed:', err);
      this.showError('Failed to load YouTube player API');
    }
  }

  /**
   * Destroy all existing players and create new ones for filtered channels.
   * Called on region filter change and initial mount.
   */
  private rebuildPlayers(): void {
    // Destroy existing players
    for (const [, player] of this.players) {
      try { player.destroy(); } catch { /* iframe may already be removed */ }
    }
    this.players.clear();
    this.unmutedPlayerId = null;

    const channels = this.getFilteredChannels();

    if (channels.length === 0) {
      replaceChildren(
        this.gridContainer,
        h('div', { className: 'empty-state' }, 'No streams for this region'),
      );
      return;
    }

    // Clear grid and build player containers
    replaceChildren(this.gridContainer);

    for (const channel of channels) {
      const card = this.createPlayerCard(channel);
      this.gridContainer.appendChild(card);
    }

    // Create YT.Player instances if API is ready
    if (window.YT?.Player) {
      for (const channel of channels) {
        this.createPlayer(channel);
      }
    }
  }

  private createPlayerCard(channel: LiveChannel): HTMLElement {
    const card = h('div', { className: 'live-stream-card', dataset: { channelId: channel.id } });

    // 16:9 aspect ratio container
    const playerWrapper = h('div', { className: 'live-stream-player-wrapper' });
    const playerTarget = h('div', { className: 'live-stream-player-target' });
    playerTarget.id = `yt-player-${channel.id}`;
    playerWrapper.appendChild(playerTarget);

    // Mute toggle overlay
    const muteBtn = h('button', {
      className: 'live-stream-mute-btn',
      'aria-label': 'Unmute',
    }, 'MUTED');
    muteBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.toggleMute(channel.id);
    });
    playerWrapper.appendChild(muteBtn);

    card.appendChild(playerWrapper);

    // Channel info bar
    const infoBar = h('div', { className: 'live-stream-info' },
      h('span', { className: 'live-stream-dot' }),
      h('span', { className: 'live-stream-name' }, channel.name),
    );
    card.appendChild(infoBar);

    return card;
  }

  private createPlayer(channel: LiveChannel): void {
    if (!window.YT?.Player) return;

    const targetEl = document.getElementById(`yt-player-${channel.id}`);
    if (!targetEl) return;

    try {
      const player = new window.YT.Player(targetEl, {
        videoId: channel.youtubeId,
        host: 'https://www.youtube-nocookie.com',
        playerVars: {
          autoplay: 1,
          mute: 1,
          controls: 0,
          modestbranding: 1,
          rel: 0,
          showinfo: 0,
          fs: 0,
          playsinline: 1,
        },
        events: {
          onReady: () => {
            // Player is ready and muted by default
          },
          onError: (event) => {
            console.warn(`[LiveStreamsPanel] Player error for ${channel.name}: code ${event.data}`);
            // Show error state on the card
            const card = this.gridContainer.querySelector(`[data-channel-id="${channel.id}"]`);
            if (card) {
              const wrapper = card.querySelector('.live-stream-player-wrapper');
              if (wrapper) {
                const errorEl = h('div', { className: 'live-stream-offline' }, 'OFFLINE');
                wrapper.appendChild(errorEl);
              }
            }
          },
        },
      });

      this.players.set(channel.id, player);
    } catch (err) {
      console.error(`[LiveStreamsPanel] Failed to create player for ${channel.name}:`, err);
    }
  }

  // -------------------------------------------------------------------------
  // Mute controls (exclusive unmute)
  // -------------------------------------------------------------------------

  private toggleMute(channelId: string): void {
    const player = this.players.get(channelId);
    if (!player) return;

    if (this.unmutedPlayerId === channelId) {
      // Currently unmuted -- mute it
      player.mute();
      this.unmutedPlayerId = null;
      this.updateMuteUI(channelId, true);
    } else {
      // Mute the currently unmuted player first
      if (this.unmutedPlayerId) {
        const prev = this.players.get(this.unmutedPlayerId);
        if (prev) {
          prev.mute();
          this.updateMuteUI(this.unmutedPlayerId, true);
        }
      }
      // Unmute this one
      player.unMute();
      this.unmutedPlayerId = channelId;
      this.updateMuteUI(channelId, false);
    }
  }

  private updateMuteUI(channelId: string, isMuted: boolean): void {
    const card = this.gridContainer.querySelector(`[data-channel-id="${channelId}"]`);
    if (!card) return;
    const btn = card.querySelector('.live-stream-mute-btn');
    if (btn) {
      btn.textContent = isMuted ? 'MUTED' : 'UNMUTED';
      btn.classList.toggle('unmuted', !isMuted);
      btn.setAttribute('aria-label', isMuted ? 'Unmute' : 'Mute');
    }
  }

  // -------------------------------------------------------------------------
  // Idle detection
  // -------------------------------------------------------------------------

  private setupIdleDetection(): void {
    this.activityHandler = () => {
      if (this.isIdle) {
        this.resumeFromIdle();
      }
      this.resetIdleTimer();
    };

    for (const event of ACTIVITY_EVENTS) {
      document.addEventListener(event, this.activityHandler, { passive: true });
    }

    this.resetIdleTimer();
  }

  private resetIdleTimer(): void {
    if (this.idleTimer) {
      clearTimeout(this.idleTimer);
    }
    this.idleTimer = setTimeout(() => {
      this.goIdle();
    }, IDLE_TIMEOUT_MS);
  }

  private goIdle(): void {
    if (this.isIdle) return;
    this.isIdle = true;

    // Pause all playing streams
    for (const [, player] of this.players) {
      try { player.pauseVideo(); } catch { /* player may not be ready */ }
    }
  }

  private resumeFromIdle(): void {
    this.isIdle = false;

    // Resume: play the first visible player
    const channels = this.getFilteredChannels();
    if (channels.length > 0) {
      const firstId = channels[0]!.id;
      const firstPlayer = this.players.get(firstId);
      if (firstPlayer) {
        try { firstPlayer.playVideo(); } catch { /* player may not be ready */ }
      }
    }
  }

  // -------------------------------------------------------------------------
  // Cleanup
  // -------------------------------------------------------------------------

  public override destroy(): void {
    // Clear idle timer
    if (this.idleTimer) {
      clearTimeout(this.idleTimer);
      this.idleTimer = null;
    }

    // Remove idle detection listeners
    if (this.activityHandler) {
      for (const event of ACTIVITY_EVENTS) {
        document.removeEventListener(event, this.activityHandler);
      }
      this.activityHandler = null;
    }

    // Destroy all YouTube players
    for (const [, player] of this.players) {
      try { player.destroy(); } catch { /* best-effort */ }
    }
    this.players.clear();

    super.destroy();
  }
}
