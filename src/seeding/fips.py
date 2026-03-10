"""FIPS 10-4 to ISO 3166-1 alpha-2 conversion.

GDELT stores country codes in the deprecated CIA FIPS 10-4 system.
142 of 251 FIPS codes differ from their ISO equivalents (e.g., FIPS IS = Israel,
ISO IS = Iceland). This module provides the authoritative conversion layer.

Data source: src/seeding/data/fips_to_iso.csv (251 mappings, loaded at import time).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent / "data"
_FIPS_CSV = _DATA_DIR / "fips_to_iso.csv"

# Track warned codes to avoid log spam
_warned_codes: set[str] = set()


def _load_fips_mapping() -> dict[str, str]:
    """Load FIPS 10-4 -> ISO 3166-1 alpha-2 mapping from CSV data file.

    The CSV has a header row + 251 country mappings.
    File is ~1.8KB, loads in <1ms.
    """
    if not _FIPS_CSV.exists():
        raise FileNotFoundError(
            f"FIPS-to-ISO CSV not found at {_FIPS_CSV}. "
            "Ensure src/seeding/data/fips_to_iso.csv is present."
        )

    mapping: dict[str, str] = {}
    with open(_FIPS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            fips = row["fips"].strip().upper()
            iso = row["iso"].strip().upper()
            if len(fips) == 2 and len(iso) == 2:
                mapping[fips] = iso
    logger.info("Loaded %d FIPS-to-ISO mappings from %s", len(mapping), _FIPS_CSV)
    return mapping


# Module-level singleton -- loaded once at import time
FIPS_TO_ISO: dict[str, str] = _load_fips_mapping()


def fips_to_iso(code: str) -> str | None:
    """Convert a FIPS 10-4 code (2-letter) or ISO alpha-3 code (3-letter) to ISO alpha-2.

    Args:
        code: A 2-letter FIPS 10-4 code or 3-letter ISO alpha-3 code.

    Returns:
        ISO 3166-1 alpha-2 code, or None if unmapped.
    """
    code = code.strip().upper()

    if len(code) == 2:
        result = FIPS_TO_ISO.get(code)
        if result is None and code not in _warned_codes:
            _warned_codes.add(code)
            logger.warning("Unmapped FIPS code: %s", code)
        return result

    if len(code) == 3:
        try:
            import pycountry

            country = pycountry.countries.get(alpha_3=code)
            if country is not None:
                return country.alpha_2
        except Exception:
            pass
        if code not in _warned_codes:
            _warned_codes.add(code)
            logger.debug("Unmapped 3-letter code: %s", code)
        return None

    return None


def get_sovereign_isos() -> set[str]:
    """Return ISO alpha-2 codes for ~250 sovereign states + disputed territories.

    Built dynamically from pycountry (249 entries including territories).
    Kosovo (XK) added manually -- user-assigned code, not in ISO 3166-1.
    """
    import pycountry

    codes = {c.alpha_2 for c in pycountry.countries}
    codes.add("XK")  # Kosovo
    return codes
