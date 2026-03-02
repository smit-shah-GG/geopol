"""
CAMEO super-category mapping and literature-derived cold-start alpha priors.

This module provides the static knowledge base that seeds the calibration
subsystem before any prediction-outcome data has been accumulated. It
bridges two key-spaces:

1. CAMEO root codes (01-20): The authoritative event taxonomy used by
   GDELT, ICEWS, and ACLED. Grouped into four super-categories per the
   standard CAMEO quadrant taxonomy.

2. Keyword-inferred categories (diplomatic/economic/conflict): The
   heuristic categories assigned by EnsemblePredictor._infer_category()
   when no CAMEO code is available.

Cold-start priors rationale:
----------------------------
The alpha values represent LLM weight in the ensemble blend:
    P_final = alpha * P_llm + (1 - alpha) * P_tkg

Literature basis (from geopol.md and domain research):
- Verbal cooperation (01-05): LLMs excel at diplomatic language analysis;
  TKG embeddings capture structural patterns less effectively for events
  dominated by rhetorical signaling. alpha=0.65 favors LLM.
- Material cooperation (06-09): Trade, aid, and economic agreements have
  stronger structural signatures in knowledge graphs (actor-relation-actor
  triples), but LLMs still provide useful contextual reasoning. alpha=0.60.
- Verbal conflict (10-14): Threats and accusations are partially structural
  (recurring actor pairs) but heavily context-dependent. Balanced. alpha=0.55.
- Material conflict (15-20): Military actions, sanctions, and physical
  violence have the strongest graph-structural signatures (RE-GCN MRR
  ~0.40+ on conflict triples in ICEWS/GDELT benchmarks). TKG weight
  dominates. alpha=0.50.
- Global fallback: Weighted average across category frequencies. alpha=0.58.

These are defensible starting points. The weekly calibration pipeline
(weight_optimizer.py) will rapidly adjust them as prediction-outcome
pairs accumulate.
"""

from __future__ import annotations

# ---------------------------------------------------------------
# CAMEO root code -> super-category mapping
# Follows the standard CAMEO quadrant taxonomy:
#   Quadrant 1: Verbal Cooperation (01-05)
#   Quadrant 2: Material Cooperation (06-09)
#   Quadrant 3: Verbal Conflict (10-14)
#   Quadrant 4: Material Conflict (15-20)
# ---------------------------------------------------------------
CAMEO_TO_SUPER: dict[str, str] = {
    "01": "verbal_coop",
    "02": "verbal_coop",
    "03": "verbal_coop",
    "04": "verbal_coop",
    "05": "verbal_coop",
    "06": "material_coop",
    "07": "material_coop",
    "08": "material_coop",
    "09": "material_coop",
    "10": "verbal_conflict",
    "11": "verbal_conflict",
    "12": "verbal_conflict",
    "13": "verbal_conflict",
    "14": "verbal_conflict",
    "15": "material_conflict",
    "16": "material_conflict",
    "17": "material_conflict",
    "18": "material_conflict",
    "19": "material_conflict",
    "20": "material_conflict",
}

SUPER_CATEGORIES: list[str] = [
    "verbal_coop",
    "material_coop",
    "verbal_conflict",
    "material_conflict",
]

# ---------------------------------------------------------------
# Literature-derived cold-start alpha priors (LLM weight)
# See module docstring for derivation rationale.
# ---------------------------------------------------------------
COLD_START_PRIORS: dict[str, float] = {
    "verbal_coop": 0.65,
    "material_coop": 0.60,
    "verbal_conflict": 0.55,
    "material_conflict": 0.50,
    "global": 0.58,
}

# ---------------------------------------------------------------
# Keyword-inferred category -> super-category bridge
# Maps the heuristic categories from EnsemblePredictor._infer_category()
# to the CAMEO super-category namespace.
# ---------------------------------------------------------------
KEYWORD_CATEGORY_TO_SUPER: dict[str, str] = {
    "diplomatic": "verbal_coop",
    "economic": "material_coop",
    "conflict": "material_conflict",
}


def infer_super_category(
    cameo_root_code: str | None,
    keyword_category: str | None,
) -> str | None:
    """Resolve super-category from a CAMEO root code or keyword category.

    CAMEO root code takes precedence when both are provided, since it is
    the authoritative taxonomy. Falls back to keyword category mapping
    when no CAMEO code is available.

    Args:
        cameo_root_code: Two-digit CAMEO root code ("01"-"20"), or None.
        keyword_category: Heuristic category from EnsemblePredictor
            ("diplomatic", "economic", "conflict"), or None.

    Returns:
        Super-category string, or None if neither input resolves.
    """
    if cameo_root_code is not None:
        # Normalize: strip whitespace, zero-pad single digits
        code = cameo_root_code.strip().zfill(2)
        if code in CAMEO_TO_SUPER:
            return CAMEO_TO_SUPER[code]

    if keyword_category is not None:
        key = keyword_category.strip().lower()
        if key in KEYWORD_CATEGORY_TO_SUPER:
            return KEYWORD_CATEGORY_TO_SUPER[key]

    return None
