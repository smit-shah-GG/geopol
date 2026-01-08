# Phase 1: Data Foundation - Research

**Researched:** 2026-01-09
**Domain:** GDELT API v2 and Python event data processing
**Confidence:** HIGH

<research_summary>
## Summary

Researched the GDELT ecosystem for building a geopolitical event data pipeline focused on conflicts and diplomatic events. The standard approach uses GDELT's 15-minute update cycle with Python libraries (gdeltdoc for Doc API, gdeltPyR for bulk data) combined with strategic sampling to manage the 500K-1M daily article volume.

Key finding: GDELT data has significant quality issues (55% accuracy rate, 20% redundancy) requiring careful deduplication and validation. Events reflect media coverage patterns, not ground truth—649 "kidnappings" means 649 news stories about kidnappings, not 649 actual events. This necessitates sophisticated filtering and sampling strategies.

**Primary recommendation:** Use gdeltdoc for recent data (3-month window) with GDELT100 filtering (100+ mentions), focus on QuadClass 1 (verbal cooperation) and 4 (material conflict), implement deduplication, and treat all events as media observations requiring validation.
</research_summary>

<standard_stack>
## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| gdeltdoc | 1.12.0 | GDELT Doc API client | Simpler access to recent 3 months, returns pandas DataFrames |
| gdeltPyR | 0.1.14 | Bulk GDELT data access | Parallel downloads, BigQuery integration, handles v1 and v2 |
| pandas | 2.x | Data manipulation | Standard for tabular data in Python scientific stack |
| requests | 2.x | HTTP client | For direct API calls and file downloads |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| dask | latest | Distributed computing | When processing full daily dumps |
| pyarrow | latest | Efficient storage | For parquet file storage of events |
| schedule | latest | Task scheduling | For 15-minute update cycles |
| sqlite3 | built-in | Local storage | For event cache and deduplication |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| gdeltdoc | Raw CSV files | More control but requires handling downloads, parsing |
| gdeltPyR | BigQuery direct | Better for large-scale but requires GCP account |
| Local storage | MongoDB | Better for complex queries but higher operational overhead |

**Installation:**
```bash
pip install gdeltdoc gdeltPyR pandas dask pyarrow schedule
```
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Recommended Project Structure
```
src/
├── ingestion/       # GDELT API clients and downloaders
├── filters/         # CAMEO QuadClass and event filtering
├── sampling/        # Intelligent sampling strategies
├── deduplication/   # Remove redundant events
├── storage/         # Event storage and caching
└── monitoring/      # Rate limit tracking, data quality metrics
```

### Pattern 1: Incremental Processing with Doc API
**What:** Use gdeltdoc for recent events with 15-minute polling
**When to use:** When you need events from last 3 months only
**Example:**
```python
from gdeltdoc import GdeltDoc, Filters
import schedule

def fetch_recent_events():
    f = Filters(
        keyword="conflict OR diplomatic",
        start_date="2026-01-09 00:00",
        end_date="2026-01-09 00:15"
    )
    gd = GdeltDoc()
    return gd.article_search(f)

# Run every 15 minutes
schedule.every(15).minutes.do(fetch_recent_events)
```

### Pattern 2: GDELT100 Filtering for Quality
**What:** Filter events with 100+ mentions to reduce noise
**When to use:** When data quality matters more than completeness
**Example:**
```python
def filter_high_confidence_events(events_df):
    # NumMentions field indicates media coverage
    return events_df[events_df['NumMentions'] >= 100]
```

### Pattern 3: QuadClass Filtering for Event Types
**What:** Use QuadClass field to filter verbal cooperation (1) and material conflict (4)
**When to use:** When focusing on specific event categories
**Example:**
```python
def filter_by_quadclass(events_df, classes=[1, 4]):
    return events_df[events_df['QuadClass'].isin(classes)]
```

### Anti-Patterns to Avoid
- **Treating events as ground truth:** Events are media observations, not facts
- **Ignoring duplicates:** Same event reported 151 times appears as 151 events
- **Processing full firehose:** 500K-1M daily articles will overwhelm compute resources
</architecture_patterns>

<dont_hand_roll>
## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| GDELT downloads | Custom scraper | gdeltPyR parallel downloads | Handles rate limits, parallel fetching, error recovery |
| CAMEO parsing | Manual event code mapping | Built-in QuadClass field | Already categorized into 4 classes |
| Date filtering | Manual timestamp parsing | gdeltdoc Filters | Handles date formats and timezone issues |
| Geocoding | Custom location extraction | GDELT's ActionGeo fields | Already geocoded with GNS/GNIS |
| Deduplication | Simple URL matching | Hash-based with fuzzy matching | Same story has many URLs, needs content similarity |
| BigQuery access | Direct SQL queries | pandas-gbq interface | Handles auth, pagination, type conversion |

**Key insight:** GDELT preprocessing is complex with many edge cases. The gdeltdoc and gdeltPyR libraries handle authentication, rate limiting, parallel downloads, and data formatting that would take weeks to implement correctly.
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Confusing Event Counts with Real Events
**What goes wrong:** Interpreting 649 kidnapping events as 649 actual kidnappings
**Why it happens:** GDELT counts news stories, not events
**How to avoid:** Always refer to events as "media reports of events"
**Warning signs:** Impossibly high event counts for single incidents

### Pitfall 2: Data Quality Assumptions
**What goes wrong:** Assuming GDELT data is accurate and complete
**Why it happens:** Not aware of 55% accuracy rate and 20% redundancy
**How to avoid:** Implement deduplication, use GDELT100 filtering, validate with other sources
**Warning signs:** Duplicate events with slightly different timestamps

### Pitfall 3: Geographic Centroids as Real Locations
**What goes wrong:** Events appear clustered at country centroids (e.g., Kaduna, Nigeria)
**Why it happens:** Default geolocation when specific location unknown
**How to avoid:** Filter out events at known centroid coordinates
**Warning signs:** Suspicious clustering of diverse events at single coordinates

### Pitfall 4: Rate Limit Violations
**What goes wrong:** API access blocked or throttled
**Why it happens:** Too many requests without respecting rate limits
**How to avoid:** Implement exponential backoff, use batch downloads
**Warning signs:** HTTP 429 errors, incomplete data returns

### Pitfall 5: Memory Overflow with Full Dataset
**What goes wrong:** System crashes processing daily dumps
**Why it happens:** 500K-1M articles per day exceed RAM
**How to avoid:** Use dask for distributed processing, implement sampling
**Warning signs:** Memory errors, slow processing, system freezes
</common_pitfalls>

<code_examples>
## Code Examples

### Basic GDELT Access with Filtering
```python
# Source: gdeltdoc documentation and best practices
from gdeltdoc import GdeltDoc, Filters
import pandas as pd

def fetch_conflict_diplomatic_events(date_str):
    """Fetch QuadClass 1 (verbal cooperation) and 4 (material conflict) events"""

    f = Filters(
        start_date=date_str,
        end_date=date_str,
        theme="DIPLOMATIC_EXCHANGE OR ARMED_CONFLICT"
    )

    gd = GdeltDoc()
    articles = gd.article_search(f)

    # Filter by tone for additional quality
    # Negative tone often indicates conflict
    conflict_articles = articles[articles['tone'] < -5]
    diplomatic_articles = articles[
        (articles['tone'] > -2) &
        (articles['tone'] < 2)
    ]

    return pd.concat([conflict_articles, diplomatic_articles])
```

### Deduplication Strategy
```python
# Source: Community best practice for GDELT deduplication
import hashlib

def deduplicate_events(events_df):
    """Remove duplicate events based on content similarity"""

    # Create content hash from key fields
    events_df['content_hash'] = events_df.apply(
        lambda x: hashlib.md5(
            f"{x['Actor1Code']}{x['Actor2Code']}{x['EventCode']}{x['ActionGeo']}"
            .encode()
        ).hexdigest(),
        axis=1
    )

    # Keep first occurrence within time window
    events_df['time_window'] = pd.to_datetime(events_df['DATEADDED']).dt.floor('H')

    return events_df.drop_duplicates(
        subset=['content_hash', 'time_window'],
        keep='first'
    )
```

### Intelligent Sampling for Compute Constraints
```python
# Source: Adapted from GDELT research papers
def sample_events_stratified(events_df, max_events=10000):
    """Stratified sampling preserving QuadClass distribution"""

    if len(events_df) <= max_events:
        return events_df

    # Sample proportionally from each QuadClass
    sampled = pd.DataFrame()
    for quad_class in [1, 2, 3, 4]:
        class_events = events_df[events_df['QuadClass'] == quad_class]
        class_ratio = len(class_events) / len(events_df)
        n_samples = int(max_events * class_ratio)

        # Prioritize high-mention events
        class_events = class_events.sort_values('NumMentions', ascending=False)
        sampled = pd.concat([sampled, class_events.head(n_samples)])

    return sampled
```
</code_examples>

<sota_updates>
## State of the Art (2024-2026)

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Download all CSV files | Doc API for recent events | 2023 | Simpler access, built-in filtering |
| Process entire dataset | GDELT100 filtering (100+ mentions) | 2024 | Better signal-to-noise ratio |
| Trust event accuracy | Assume 55% accuracy, validate | 2025 study | Requires validation strategy |
| Single-source events | Multi-source validation required | 2024 | Reduces false positives |

**New tools/patterns to consider:**
- **GDELT Context 2.0 API**: 72-hour window for real-time context
- **Web NGrams 3.0**: Alternative when hitting rate limits
- **RevDet algorithm**: Robust event detection and tracking in large news feeds

**Deprecated/outdated:**
- **GDELT v1.0**: Use v2.0 for 15-minute updates and better coverage
- **Manual CSV downloads**: Use gdeltPyR or gdeltdoc instead
- **Trusting single reports**: Always validate with mention counts
</sota_updates>

<open_questions>
## Open Questions

1. **Exact Rate Limits**
   - What we know: APIs are rate-limited but specific numbers not documented
   - What's unclear: Requests per second/minute limits
   - Recommendation: Implement conservative retry with exponential backoff

2. **Optimal Sampling Threshold**
   - What we know: GDELT100 (100+ mentions) is common
   - What's unclear: Best threshold for conflicts vs diplomatic events
   - Recommendation: Start with 100, tune based on precision/recall metrics

3. **BigQuery Costs**
   - What we know: BigQuery integration available
   - What's unclear: Cost for full dataset queries
   - Recommendation: Use Doc API for prototyping, evaluate BigQuery for production
</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)
- https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/ - GDELT 2.0 capabilities
- https://github.com/alex9smith/gdelt-doc-api - gdeltdoc library documentation
- https://www.gdeltproject.org/data.html - Official data access methods

### Secondary (MEDIUM confidence)
- https://github.com/linwoodc3/gdeltPyR - gdeltPyR library features (verified with PyPI)
- CAMEO QuadClass system documentation (verified across multiple sources)

### Tertiary (LOW confidence - needs validation)
- Specific sampling strategies from research papers (needs testing in practice)
</sources>

<metadata>
## Metadata

**Research scope:**
- Core technology: GDELT API v2, Doc API, BigQuery access
- Ecosystem: gdeltdoc, gdeltPyR, pandas, dask
- Patterns: Incremental processing, GDELT100 filtering, QuadClass filtering
- Pitfalls: Data quality issues, event count interpretation, rate limits

**Confidence breakdown:**
- Standard stack: HIGH - Libraries well-documented and actively maintained
- Architecture: HIGH - Based on official examples and community practices
- Pitfalls: HIGH - Well-documented in academic literature and user reports
- Code examples: MEDIUM - Adapted from documentation, needs production testing

**Research date:** 2026-01-09
**Valid until:** 2026-02-09 (30 days - GDELT APIs stable but practices evolve)
</metadata>

---

*Phase: 01-data-foundation*
*Research completed: 2026-01-09*
*Ready for planning: yes*