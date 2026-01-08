"""
Constants for GDELT event filtering and processing.
"""

# QuadClass definitions (CAMEO event categories)
QUADCLASS_VERBAL_COOPERATION = 1  # Diplomatic, negotiations, agreements
QUADCLASS_MATERIAL_COOPERATION = 2  # Aid, trade, material cooperation
QUADCLASS_VERBAL_CONFLICT = 3  # Threats, accusations, criticism
QUADCLASS_MATERIAL_CONFLICT = 4  # Armed conflict, violence, military action

# Quality thresholds
GDELT100_THRESHOLD = 100  # Minimum mentions for high-confidence events
MIN_GOLDSTEIN_CONFLICT = -5.0  # Threshold for conflict classification

# Tone ranges for event classification
TONE_RANGE_DIPLOMATIC = (-2, 2)  # Near-neutral tone for diplomatic events
TONE_RANGE_CONFLICT = (-10, -5)  # Negative tone for conflict events

# Sampling parameters
DEFAULT_SAMPLE_SIZE = 10000  # Default maximum events per batch
MIN_CLASS_REPRESENTATION = 0.1  # Minimum 10% representation for minority class

# Event code mappings (common CAMEO codes)
CONFLICT_EVENT_CODES = {
    '18': 'ASSAULT',
    '180': 'USE_UNCONVENTIONAL_VIOLENCE',
    '181': 'ABDUCT',
    '182': 'PHYSICALLY_ASSAULT',
    '183': 'TORTURE',
    '184': 'KILL',
    '19': 'FIGHT',
    '190': 'USE_UNCONVENTIONAL_MASS_VIOLENCE',
    '191': 'IMPOSE_BLOCKADE',
    '192': 'OCCUPY_TERRITORY',
    '193': 'FIGHT_SMALL_ARMS',
    '194': 'FIGHT_HEAVY_WEAPONS',
    '195': 'EMPLOY_AERIAL_WEAPONS',
    '196': 'VIOLATE_CEASEFIRE',
    '20': 'USE_UNCONVENTIONAL_MASS_VIOLENCE',
}

DIPLOMATIC_EVENT_CODES = {
    '01': 'MAKE_PUBLIC_STATEMENT',
    '010': 'MAKE_STATEMENT',
    '02': 'APPEAL',
    '03': 'EXPRESS_INTENT_TO_COOPERATE',
    '04': 'CONSULT',
    '040': 'CONSULT',
    '041': 'DISCUSS_BY_TELEPHONE',
    '042': 'MAKE_VISIT',
    '043': 'HOST_VISIT',
    '044': 'MEET_AT_THIRD_LOCATION',
    '045': 'MEDIATE',
    '046': 'ENGAGE_IN_NEGOTIATION',
    '05': 'ENGAGE_IN_DIPLOMATIC_COOPERATION',
    '050': 'ENGAGE_IN_DIPLOMATIC_COOPERATION',
    '051': 'PRAISE_OR_ENDORSE',
    '052': 'DEFEND_VERBALLY',
    '053': 'RALLY_SUPPORT',
    '054': 'GRANT_DIPLOMATIC_RECOGNITION',
    '055': 'APOLOGIZE',
    '056': 'FORGIVE',
    '057': 'SIGN_FORMAL_AGREEMENT',
}

# Actor type filters (ISO/CAMEO actor codes)
MAJOR_ACTORS = {
    'USA': 'United States',
    'CHN': 'China',
    'RUS': 'Russia',
    'GBR': 'United Kingdom',
    'FRA': 'France',
    'DEU': 'Germany',
    'JPN': 'Japan',
    'IND': 'India',
    'ISR': 'Israel',
    'IRN': 'Iran',
    'UKR': 'Ukraine',
    'PRK': 'North Korea',
    'KOR': 'South Korea',
    'SAU': 'Saudi Arabia',
    'TUR': 'Turkey',
    'BRA': 'Brazil',
    'MEX': 'Mexico',
    'CAN': 'Canada',
    'AUS': 'Australia',
    'EU': 'European Union',
    'UN': 'United Nations',
    'NATO': 'NATO',
}

# Data quality thresholds
MIN_SOURCES_THRESHOLD = 2  # Minimum number of sources for reliable event
MIN_TONE_CONFIDENCE = -100  # Events with tone below this are likely errors
MAX_TONE_CONFIDENCE = 100  # Events with tone above this are likely errors

# Batch processing limits
MAX_BATCH_SIZE = 50000  # Maximum events to process in single batch
CHUNK_SIZE = 1000  # Size for chunked processing
PARALLEL_WORKERS = 4  # Number of parallel workers for processing