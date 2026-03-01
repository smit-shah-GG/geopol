"""
Tiered RSS feed configuration for geopolitical news ingestion.

Feeds extracted from World Monitor's feeds.ts (298-domain list), filtered
to geopolitically relevant sources only. Startup/VC, lifestyle, podcast,
and purely tech-focused feeds excluded.

Tier system:
  TIER_1 -- Wire services, major outlets, government/IO sources (~15-min poll)
  TIER_2 -- Regional outlets, think tanks, specialty analysis (~60-min poll)

Feed URLs are direct RSS endpoints (no proxy wrapper). Google News search
fallbacks are used where publishers block cloud IPs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal


class FeedTier(IntEnum):
    """Polling frequency tier. Lower = more authoritative, polled more often."""

    TIER_1 = 1  # 15-minute poll interval
    TIER_2 = 2  # 60-minute poll interval


FeedCategory = Literal[
    "wire",
    "mainstream",
    "government",
    "intl_org",
    "defense",
    "intel",
    "thinktank",
    "crisis",
    "regional",
    "finance",
    "energy",
]


@dataclass(frozen=True, slots=True)
class FeedSource:
    """Immutable RSS feed definition."""

    name: str
    url: str
    tier: FeedTier
    category: FeedCategory
    lang: str = "en"


# ---------------------------------------------------------------------------
# TIER 1: Wire services, major global outlets, government & IO sources
# Polled every 15 minutes.
# ---------------------------------------------------------------------------

TIER_1_FEEDS: list[FeedSource] = [
    # --- Wire services ---
    FeedSource("Reuters World", "https://news.google.com/rss/search?q=site:reuters.com+world&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_1, "wire"),
    FeedSource("Reuters Business", "https://news.google.com/rss/search?q=site:reuters.com+business+markets&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_1, "wire"),
    FeedSource("AP News", "https://news.google.com/rss/search?q=site:apnews.com&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_1, "wire"),
    FeedSource("AFP", "https://news.google.com/rss/search?q=AFP+agence+france+presse&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_1, "wire"),
    FeedSource("Bloomberg", "https://news.google.com/rss/search?q=site:bloomberg.com+when:1d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_1, "wire"),
    FeedSource("ANSA", "https://www.ansa.it/sito/notizie/topnews/topnews_rss.xml", FeedTier.TIER_1, "wire", lang="it"),
    FeedSource("Xinhua", "https://news.google.com/rss/search?q=site:xinhuanet.com+OR+Xinhua+when:1d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_1, "wire"),
    FeedSource("TASS", "https://news.google.com/rss/search?q=site:tass.com+OR+TASS+Russia+when:1d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_1, "wire"),

    # --- Major global outlets ---
    FeedSource("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml", FeedTier.TIER_1, "mainstream"),
    FeedSource("BBC Middle East", "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml", FeedTier.TIER_1, "mainstream"),
    FeedSource("Guardian World", "https://www.theguardian.com/world/rss", FeedTier.TIER_1, "mainstream"),
    FeedSource("CNN World", "http://rss.cnn.com/rss/cnn_world.rss", FeedTier.TIER_1, "mainstream"),
    FeedSource("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml", FeedTier.TIER_1, "mainstream"),
    FeedSource("NPR News", "https://feeds.npr.org/1001/rss.xml", FeedTier.TIER_1, "mainstream"),
    FeedSource("France 24", "https://www.france24.com/en/rss", FeedTier.TIER_1, "mainstream"),
    FeedSource("EuroNews", "https://www.euronews.com/rss?format=xml", FeedTier.TIER_1, "mainstream"),
    FeedSource("DW News", "https://rss.dw.com/xml/rss-en-all", FeedTier.TIER_1, "mainstream"),
    FeedSource("Le Monde", "https://www.lemonde.fr/en/rss/une.xml", FeedTier.TIER_1, "mainstream"),
    FeedSource("Financial Times", "https://www.ft.com/rss/home", FeedTier.TIER_1, "finance"),
    FeedSource("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html", FeedTier.TIER_1, "finance"),

    # --- Government & International Organizations ---
    FeedSource("White House", "https://news.google.com/rss/search?q=site:whitehouse.gov&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_1, "government"),
    FeedSource("State Dept", "https://news.google.com/rss/search?q=site:state.gov+OR+%22State+Department%22&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_1, "government"),
    FeedSource("Pentagon", "https://news.google.com/rss/search?q=site:defense.gov+OR+Pentagon&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_1, "government"),
    FeedSource("UN News", "https://news.un.org/feed/subscribe/en/news/all/rss.xml", FeedTier.TIER_1, "intl_org"),
    FeedSource("IAEA", "https://www.iaea.org/feeds/topnews", FeedTier.TIER_1, "intl_org"),
    FeedSource("WHO", "https://www.who.int/rss-feeds/news-english.xml", FeedTier.TIER_1, "intl_org"),
    FeedSource("CISA", "https://www.cisa.gov/cybersecurity-advisories/all.xml", FeedTier.TIER_1, "government"),

    # --- European tier-1 state broadcasters ---
    FeedSource("Tagesschau", "https://www.tagesschau.de/xml/rss2/", FeedTier.TIER_1, "mainstream", lang="de"),
    FeedSource("NOS Nieuws", "https://feeds.nos.nl/nosnieuwsalgemeen", FeedTier.TIER_1, "mainstream", lang="nl"),
    FeedSource("SVT Nyheter", "https://www.svt.se/nyheter/rss.xml", FeedTier.TIER_1, "mainstream", lang="sv"),

    # --- Tier-1 defense ---
    FeedSource("UK MOD", "https://www.gov.uk/government/organisations/ministry-of-defence.atom", FeedTier.TIER_1, "defense"),
]

# ---------------------------------------------------------------------------
# TIER 2: Regional outlets, think tanks, specialty analysis
# Polled every 60 minutes.
# ---------------------------------------------------------------------------

TIER_2_FEEDS: list[FeedSource] = [
    # --- Think tanks & analysis ---
    FeedSource("Foreign Policy", "https://foreignpolicy.com/feed/", FeedTier.TIER_2, "thinktank"),
    FeedSource("Foreign Affairs", "https://www.foreignaffairs.com/rss.xml", FeedTier.TIER_2, "thinktank"),
    FeedSource("Atlantic Council", "https://www.atlanticcouncil.org/feed/", FeedTier.TIER_2, "thinktank"),
    FeedSource("CSIS", "https://news.google.com/rss/search?q=site:csis.org+when:7d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "thinktank"),
    FeedSource("RAND", "https://www.rand.org/rss/all.xml", FeedTier.TIER_2, "thinktank"),
    FeedSource("Brookings", "https://www.brookings.edu/feed/", FeedTier.TIER_2, "thinktank"),
    FeedSource("Carnegie", "https://carnegieendowment.org/rss/", FeedTier.TIER_2, "thinktank"),
    FeedSource("War on the Rocks", "https://warontherocks.com/feed", FeedTier.TIER_2, "thinktank"),
    FeedSource("AEI", "https://www.aei.org/feed/", FeedTier.TIER_2, "thinktank"),
    FeedSource("Responsible Statecraft", "https://responsiblestatecraft.org/feed/", FeedTier.TIER_2, "thinktank"),
    FeedSource("RUSI", "https://news.google.com/rss/search?q=site:rusi.org+when:3d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "thinktank"),
    FeedSource("FPRI", "https://www.fpri.org/feed/", FeedTier.TIER_2, "thinktank"),
    FeedSource("Jamestown", "https://jamestown.org/feed/", FeedTier.TIER_2, "thinktank"),
    FeedSource("Wilson Center", "https://news.google.com/rss/search?q=site:wilsoncenter.org+when:7d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "thinktank"),
    FeedSource("GMF", "https://news.google.com/rss/search?q=site:gmfus.org+when:7d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "thinktank"),
    FeedSource("Stimson Center", "https://www.stimson.org/feed/", FeedTier.TIER_2, "thinktank"),
    FeedSource("CNAS", "https://news.google.com/rss/search?q=site:cnas.org+when:7d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "thinktank"),
    FeedSource("Lowy Institute", "https://news.google.com/rss/search?q=site:lowyinstitute.org+when:7d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "thinktank"),
    FeedSource("Arms Control Assn", "https://news.google.com/rss/search?q=site:armscontrol.org+when:7d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "thinktank"),
    FeedSource("Bulletin of Atomic Scientists", "https://news.google.com/rss/search?q=site:thebulletin.org+when:7d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "thinktank"),
    FeedSource("Chatham House", "https://news.google.com/rss/search?q=site:chathamhouse.org+when:7d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "thinktank"),

    # --- Defense & OSINT ---
    FeedSource("Defense One", "https://www.defenseone.com/rss/all/", FeedTier.TIER_2, "defense"),
    FeedSource("Breaking Defense", "https://breakingdefense.com/feed/", FeedTier.TIER_2, "defense"),
    FeedSource("The War Zone", "https://news.google.com/rss/search?q=site:thedrive.com+%22war+zone%22+when:7d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "defense"),
    FeedSource("Defense News", "https://www.defensenews.com/arc/outboundfeeds/rss/?outputType=xml", FeedTier.TIER_2, "defense"),
    FeedSource("Janes", "https://news.google.com/rss/search?q=site:janes.com+when:3d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "defense"),
    FeedSource("Military Times", "https://www.militarytimes.com/arc/outboundfeeds/rss/?outputType=xml", FeedTier.TIER_2, "defense"),
    FeedSource("USNI News", "https://news.usni.org/feed", FeedTier.TIER_2, "defense"),
    FeedSource("Oryx OSINT", "https://www.oryxspioenkop.com/feeds/posts/default?alt=rss", FeedTier.TIER_2, "intel"),
    FeedSource("Bellingcat", "https://www.bellingcat.com/feed/", FeedTier.TIER_2, "intel"),
    FeedSource("Krebs Security", "https://krebsonsecurity.com/feed/", FeedTier.TIER_2, "intel"),

    # --- Crisis monitoring ---
    FeedSource("CrisisWatch", "https://www.crisisgroup.org/rss", FeedTier.TIER_2, "crisis"),
    FeedSource("UNHCR", "https://news.google.com/rss/search?q=site:unhcr.org+OR+UNHCR+refugees+when:3d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "crisis"),
    FeedSource("FAO GIEWS", "https://news.google.com/rss/search?q=site:fao.org+GIEWS+food+security+when:30d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "crisis"),

    # --- US Government (tier-2 agencies) ---
    FeedSource("Treasury", "https://news.google.com/rss/search?q=site:treasury.gov+OR+%22Treasury+Department%22&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "government"),
    FeedSource("DOJ", "https://news.google.com/rss/search?q=site:justice.gov+OR+%22Justice+Department%22+DOJ&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "government"),
    FeedSource("Federal Reserve", "https://www.federalreserve.gov/feeds/press_all.xml", FeedTier.TIER_2, "government"),
    FeedSource("SEC", "https://www.sec.gov/news/pressreleases.rss", FeedTier.TIER_2, "government"),
    FeedSource("CDC", "https://news.google.com/rss/search?q=site:cdc.gov+OR+CDC+health&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "government"),
    FeedSource("DHS", "https://news.google.com/rss/search?q=site:dhs.gov+OR+%22Homeland+Security%22&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "government"),
    FeedSource("FEMA", "https://news.google.com/rss/search?q=site:fema.gov+OR+FEMA+emergency&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "government"),

    # --- Regional: Middle East ---
    FeedSource("Guardian ME", "https://www.theguardian.com/world/middleeast/rss", FeedTier.TIER_2, "regional"),
    FeedSource("Al Arabiya", "https://news.google.com/rss/search?q=site:english.alarabiya.net+when:2d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "regional"),
    FeedSource("Iran International", "https://news.google.com/rss/search?q=site:iranintl.com+when:2d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "regional"),
    FeedSource("Haaretz", "https://news.google.com/rss/search?q=site:haaretz.com+when:7d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "regional"),
    FeedSource("Arab News", "https://news.google.com/rss/search?q=site:arabnews.com+when:7d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "regional"),

    # --- Regional: Europe ---
    FeedSource("Politico", "https://news.google.com/rss/search?q=site:politico.com+when:1d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "mainstream"),
    FeedSource("Der Spiegel", "https://www.spiegel.de/schlagzeilen/tops/index.rss", FeedTier.TIER_2, "mainstream", lang="de"),
    FeedSource("Die Zeit", "https://newsfeed.zeit.de/index", FeedTier.TIER_2, "mainstream", lang="de"),
    FeedSource("Corriere della Sera", "https://xml2.corriereobjects.it/rss/incipit.xml", FeedTier.TIER_2, "mainstream", lang="it"),
    FeedSource("El Pais", "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada", FeedTier.TIER_2, "mainstream", lang="es"),
    FeedSource("Kyiv Independent", "https://news.google.com/rss/search?q=site:kyivindependent.com+when:3d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "regional"),
    FeedSource("Moscow Times", "https://www.themoscowtimes.com/rss/news", FeedTier.TIER_2, "regional"),
    FeedSource("Meduza", "https://meduza.io/rss/all", FeedTier.TIER_2, "regional", lang="ru"),

    # --- Regional: Africa ---
    FeedSource("BBC Africa", "https://feeds.bbci.co.uk/news/world/africa/rss.xml", FeedTier.TIER_2, "regional"),
    FeedSource("Africa News", "https://news.google.com/rss/search?q=(Africa+OR+Nigeria+OR+Kenya+OR+%22South+Africa%22+OR+Ethiopia)+when:2d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "regional"),
    FeedSource("Sahel Crisis", "https://news.google.com/rss/search?q=(Sahel+OR+Mali+OR+Niger+OR+%22Burkina+Faso%22+OR+Wagner)+when:3d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "regional"),
    FeedSource("Premium Times", "https://www.premiumtimesng.com/feed", FeedTier.TIER_2, "regional"),

    # --- Regional: Latin America ---
    FeedSource("BBC Latin America", "https://feeds.bbci.co.uk/news/world/latin_america/rss.xml", FeedTier.TIER_2, "regional"),
    FeedSource("Guardian Americas", "https://www.theguardian.com/world/americas/rss", FeedTier.TIER_2, "regional"),
    FeedSource("InSight Crime", "https://insightcrime.org/feed/", FeedTier.TIER_2, "regional"),

    # --- Regional: Asia-Pacific ---
    FeedSource("BBC Asia", "https://feeds.bbci.co.uk/news/world/asia/rss.xml", FeedTier.TIER_2, "regional"),
    FeedSource("The Diplomat", "https://thediplomat.com/feed/", FeedTier.TIER_2, "regional"),
    FeedSource("South China Morning Post", "https://www.scmp.com/rss/91/feed/", FeedTier.TIER_2, "regional"),
    FeedSource("Nikkei Asia", "https://news.google.com/rss/search?q=site:asia.nikkei.com+when:3d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "regional"),
    FeedSource("CNA Singapore", "https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml", FeedTier.TIER_2, "regional"),

    # --- Energy & commodities ---
    FeedSource("Oil & Gas", "https://news.google.com/rss/search?q=(oil+price+OR+OPEC+OR+%22natural+gas%22+OR+pipeline+OR+LNG)+when:2d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "energy"),
    FeedSource("Nuclear Energy", "https://news.google.com/rss/search?q=(%22nuclear+energy%22+OR+%22nuclear+power%22+OR+uranium+OR+IAEA)+when:3d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "energy"),
    FeedSource("Mining & Resources", "https://news.google.com/rss/search?q=(lithium+OR+%22rare+earth%22+OR+cobalt+OR+mining)+when:3d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "energy"),

    # --- Finance (geopolitical-adjacent) ---
    FeedSource("MarketWatch", "https://news.google.com/rss/search?q=site:marketwatch.com+markets+when:1d&hl=en-US&gl=US&ceid=US:en", FeedTier.TIER_2, "finance"),
]


# ---------------------------------------------------------------------------
# Aggregate lists
# ---------------------------------------------------------------------------

ALL_FEEDS: list[FeedSource] = TIER_1_FEEDS + TIER_2_FEEDS
"""Complete feed list, both tiers. Use for iteration/validation."""


def get_feeds_by_tier(tier: FeedTier) -> list[FeedSource]:
    """Return feeds matching the specified tier."""
    return [f for f in ALL_FEEDS if f.tier == tier]


def get_feeds_by_category(category: FeedCategory) -> list[FeedSource]:
    """Return feeds matching the specified category."""
    return [f for f in ALL_FEEDS if f.category == category]


def validate_feeds() -> list[str]:
    """Check feed list invariants. Returns list of error messages (empty = OK)."""
    errors: list[str] = []
    seen_names: set[str] = set()
    seen_urls: set[str] = set()

    for feed in ALL_FEEDS:
        if feed.name in seen_names:
            errors.append(f"Duplicate feed name: {feed.name}")
        seen_names.add(feed.name)

        if feed.url in seen_urls:
            errors.append(f"Duplicate feed URL for {feed.name}: {feed.url}")
        seen_urls.add(feed.url)

        if not feed.url.startswith(("http://", "https://")):
            errors.append(f"Invalid URL for {feed.name}: {feed.url}")

    return errors


# ---------------------------------------------------------------------------
# Propaganda risk metadata (from WM SOURCE_PROPAGANDA_RISK)
# Exposed for downstream weighting/labeling of ingested articles.
# ---------------------------------------------------------------------------

PROPAGANDA_RISK: dict[str, dict[str, str]] = {
    "Xinhua": {"risk": "high", "state": "China", "note": "Official CCP news agency"},
    "TASS": {"risk": "high", "state": "Russia", "note": "Russian state news agency"},
    "Al Jazeera": {"risk": "medium", "state": "Qatar", "note": "Qatari state-funded, independent editorial"},
    "Al Arabiya": {"risk": "medium", "state": "Saudi Arabia", "note": "Saudi-owned, reflects Gulf perspective"},
    "France 24": {"risk": "medium", "state": "France", "note": "French state-funded, editorially independent"},
    "DW News": {"risk": "medium", "state": "Germany", "note": "German state-funded, editorially independent"},
}


def get_propaganda_risk(source_name: str) -> str:
    """Return 'low', 'medium', or 'high' propaganda risk for a source."""
    profile = PROPAGANDA_RISK.get(source_name)
    if profile is None:
        return "low"
    return profile["risk"]
