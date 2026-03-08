/**
 * LiveStreamsPanel -- Single YouTube live stream player with channel-name pills
 * and region filtering via gear icon popover.
 *
 * Shows one stream at a time. Channel pills (e.g., "BBC News", "Al Jazeera")
 * act as the primary selector. Gear icon in the panel header opens a region
 * filter popover that narrows which channel pills are visible.
 *
 * YouTube IFrame API loaded lazily via dynamic script injection with a
 * Promise wrapper to handle the async onYouTubeIframeAPIReady callback.
 * Uses loadVideoById() to switch streams without destroying the player.
 *
 * Idle detection pauses the player after 5 minutes of inactivity.
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
  loadVideoById(videoId: string): void;
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

function loadYouTubeAPI(): Promise<void> {
  if (window.YT?.Player) {
    return Promise.resolve();
  }

  if (ytApiPromise) {
    return ytApiPromise;
  }

  ytApiPromise = new Promise<void>((resolve) => {
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

const IDLE_TIMEOUT_MS = 300_000;
const ACTIVITY_EVENTS: (keyof DocumentEventMap)[] = ['mousemove', 'keypress', 'click', 'scroll'];

// ---------------------------------------------------------------------------
// LiveStreamsPanel
// ---------------------------------------------------------------------------

export class LiveStreamsPanel extends Panel {
  private regionFilter: RegionFilter = 'All';
  private activeChannel: LiveChannel;
  private player: YTPlayerInstance | null = null;
  private isMuted = true;

  private gearBtn: HTMLElement;
  private regionPopover: HTMLElement;
  private pillContainer: HTMLElement;
  private playerWrapper: HTMLElement;
  private playerTarget: HTMLElement;
  private channelInfoBar: HTMLElement;
  private muteBtn: HTMLElement;

  private idleTimer: ReturnType<typeof setTimeout> | null = null;
  private isIdle = false;
  private activityHandler: (() => void) | null = null;
  private readonly onClickOutside: (e: MouseEvent) => void;

  constructor() {
    super({ id: 'live-streams', title: 'LIVE STREAMS', showCount: false });

    // Default to first channel
    this.activeChannel = CHANNELS[0]!;

    // Gear button in header (right side)
    this.gearBtn = h('button', {
      className: 'live-streams-gear-btn',
      'aria-label': 'Filter by region',
      title: 'Filter by region',
    });
    this.gearBtn.innerHTML = '&#9881;';
    this.gearBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.regionPopover.classList.toggle('hidden');
    });
    this.header.appendChild(this.gearBtn);

    // Channel pill bar
    this.pillContainer = document.createElement('div');
    this.pillContainer.className = 'live-streams-pills';
    this.element.insertBefore(this.pillContainer, this.content);

    // Region filter popover (hidden by default)
    this.regionPopover = document.createElement('div');
    this.regionPopover.className = 'live-streams-region-popover hidden';
    this.buildRegionPopover();
    this.element.insertBefore(this.regionPopover, this.content);

    // Clear base class skeleton before building player UI
    replaceChildren(this.content);

    // Player area (single stream)
    this.playerWrapper = h('div', { className: 'live-stream-player-wrapper' });
    this.playerTarget = h('div', { className: 'live-stream-player-target' });
    this.playerTarget.id = 'yt-player-active';
    this.playerWrapper.appendChild(this.playerTarget);

    // Mute toggle overlay
    this.muteBtn = h('button', {
      className: 'live-stream-mute-btn',
      'aria-label': 'Unmute',
    }, 'MUTED');
    this.muteBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.toggleMute();
    });
    this.playerWrapper.appendChild(this.muteBtn);

    this.content.appendChild(this.playerWrapper);

    // Channel info bar below player
    this.channelInfoBar = h('div', { className: 'live-stream-info' },
      h('span', { className: 'live-stream-dot' }),
      h('span', { className: 'live-stream-name' }, this.activeChannel.name),
    );
    this.content.appendChild(this.channelInfoBar);

    // Build channel pills
    this.buildChannelPills();

    // Close popover on outside click
    this.onClickOutside = (e: MouseEvent) => {
      if (
        !this.regionPopover.contains(e.target as Node) &&
        !this.gearBtn.contains(e.target as Node)
      ) {
        this.regionPopover.classList.add('hidden');
      }
    };
    document.addEventListener('click', this.onClickOutside);

    // Setup idle detection
    this.setupIdleDetection();

    // Load API and create player
    this.initPlayer();
  }

  // -------------------------------------------------------------------------
  // Region popover
  // -------------------------------------------------------------------------

  private buildRegionPopover(): void {
    for (const region of REGION_FILTERS) {
      const btn = h('button', {
        className: `live-streams-region-option${region === 'All' ? ' active' : ''}`,
      }, region);
      btn.addEventListener('click', () => {
        this.regionFilter = region;
        for (const opt of this.regionPopover.querySelectorAll('.live-streams-region-option')) {
          opt.classList.toggle('active', opt === btn);
        }
        this.regionPopover.classList.add('hidden');
        this.buildChannelPills();
        // If active channel is not in filtered set, switch to first visible
        const filtered = this.getFilteredChannels();
        if (filtered.length > 0 && !filtered.some((ch) => ch.id === this.activeChannel.id)) {
          this.selectChannel(filtered[0]!);
        }
      });
      this.regionPopover.appendChild(btn);
    }
  }

  // -------------------------------------------------------------------------
  // Channel pills
  // -------------------------------------------------------------------------

  private buildChannelPills(): void {
    replaceChildren(this.pillContainer);

    const channels = this.getFilteredChannels();
    for (const channel of channels) {
      const pill = h('button', {
        className: `live-streams-channel-pill${channel.id === this.activeChannel.id ? ' active' : ''}`,
      }, channel.name);
      pill.addEventListener('click', () => this.selectChannel(channel));
      this.pillContainer.appendChild(pill);
    }
  }

  private getFilteredChannels(): LiveChannel[] {
    if (this.regionFilter === 'All') return CHANNELS;
    return CHANNELS.filter(
      (ch) => ch.region === this.regionFilter || ch.region === 'Global',
    );
  }

  // -------------------------------------------------------------------------
  // Channel selection
  // -------------------------------------------------------------------------

  private selectChannel(channel: LiveChannel): void {
    if (this.activeChannel.id === channel.id) return;
    this.activeChannel = channel;

    // Update channel info bar
    const nameEl = this.channelInfoBar.querySelector('.live-stream-name');
    if (nameEl) nameEl.textContent = channel.name;

    // Update pill active state
    for (const pill of this.pillContainer.querySelectorAll('.live-streams-channel-pill')) {
      pill.classList.toggle('active', pill.textContent === channel.name);
    }

    // Switch the stream
    this.switchStream(channel);
  }

  private switchStream(channel: LiveChannel): void {
    if (this.player) {
      try {
        this.player.loadVideoById(channel.youtubeId);
        // Maintain current mute state
        if (this.isMuted) {
          this.player.mute();
        } else {
          this.player.unMute();
        }
        return;
      } catch {
        // loadVideoById failed — fall back to full recreation
      }
    }

    // Recreate player from scratch
    this.destroyPlayer();
    this.playerTarget = h('div', { className: 'live-stream-player-target' });
    this.playerTarget.id = 'yt-player-active';
    // Insert before mute button
    this.playerWrapper.insertBefore(this.playerTarget, this.muteBtn);
    this.createPlayer(channel);
  }

  // -------------------------------------------------------------------------
  // Player lifecycle
  // -------------------------------------------------------------------------

  private async initPlayer(): Promise<void> {
    try {
      await loadYouTubeAPI();
      this.createPlayer(this.activeChannel);
    } catch (err) {
      console.error('[LiveStreamsPanel] YouTube API load failed:', err);
      this.showErrorWithRetry('Failed to load YouTube player API', () => { void this.initPlayer(); });
    }
  }

  private createPlayer(channel: LiveChannel): void {
    if (!window.YT?.Player) return;

    const targetEl = document.getElementById('yt-player-active');
    if (!targetEl) return;

    try {
      this.player = new window.YT.Player(targetEl, {
        videoId: channel.youtubeId,
        host: 'https://www.youtube-nocookie.com',
        playerVars: {
          autoplay: 1,
          mute: 1,
          controls: 1,
          modestbranding: 1,
          rel: 0,
          showinfo: 0,
          playsinline: 1,
        },
        events: {
          onReady: () => {
            this.isMuted = true;
            this.updateMuteUI();
          },
          onError: (event) => {
            console.warn(`[LiveStreamsPanel] Player error for ${channel.name}: code ${event.data}`);
            const errorEl = h('div', { className: 'live-stream-offline' }, 'OFFLINE');
            this.playerWrapper.appendChild(errorEl);
          },
        },
      });
    } catch (err) {
      console.error(`[LiveStreamsPanel] Failed to create player for ${channel.name}:`, err);
    }
  }

  private destroyPlayer(): void {
    if (this.player) {
      try { this.player.destroy(); } catch { /* iframe may already be removed */ }
      this.player = null;
    }
    // Remove old player target and offline overlay
    const oldTarget = document.getElementById('yt-player-active');
    if (oldTarget) oldTarget.remove();
    const offline = this.playerWrapper.querySelector('.live-stream-offline');
    if (offline) offline.remove();
  }

  // -------------------------------------------------------------------------
  // Mute controls
  // -------------------------------------------------------------------------

  private toggleMute(): void {
    if (!this.player) return;

    if (this.isMuted) {
      this.player.unMute();
      this.isMuted = false;
    } else {
      this.player.mute();
      this.isMuted = true;
    }
    this.updateMuteUI();
  }

  private updateMuteUI(): void {
    this.muteBtn.textContent = this.isMuted ? 'MUTED' : 'UNMUTED';
    this.muteBtn.classList.toggle('unmuted', !this.isMuted);
    this.muteBtn.setAttribute('aria-label', this.isMuted ? 'Unmute' : 'Mute');
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
    if (this.player) {
      try { this.player.pauseVideo(); } catch { /* player may not be ready */ }
    }
  }

  private resumeFromIdle(): void {
    this.isIdle = false;
    if (this.player) {
      try { this.player.playVideo(); } catch { /* player may not be ready */ }
    }
  }

  // -------------------------------------------------------------------------
  // Cleanup
  // -------------------------------------------------------------------------

  public override destroy(): void {
    if (this.idleTimer) {
      clearTimeout(this.idleTimer);
      this.idleTimer = null;
    }

    if (this.activityHandler) {
      for (const event of ACTIVITY_EVENTS) {
        document.removeEventListener(event, this.activityHandler);
      }
      this.activityHandler = null;
    }

    document.removeEventListener('click', this.onClickOutside);
    this.destroyPlayer();
    super.destroy();
  }
}
