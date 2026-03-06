"""
Unit tests for cross-source event deduplication.

Tests verify:
- CAMEO code to coarse type mapping correctness
- Fingerprint determinism and normalization
- Intra-source events pass through (handled by existing dedup)
- Cross-source collision prefers ACLED over GDELT
- Suppressed duplicates are counted in stats
- Edge cases: empty codes, None country, unknown sources
"""

import pytest

from src.knowledge_graph.cross_source_dedup import (
    CrossSourceDedupFilter,
    cameo_to_coarse_type,
    cross_source_fingerprint,
)


class TestCameoToCoarseType:
    """Test CAMEO code to coarse category mapping."""

    @pytest.mark.parametrize(
        "cameo_code,expected",
        [
            ("01", "cooperation"),
            ("02", "cooperation"),
            ("05", "cooperation"),
            ("042", "cooperation"),  # 3-digit, prefix 04
            ("0512", "cooperation"),  # 4-digit, prefix 05
            ("06", "diplomacy"),
            ("09", "diplomacy"),
            ("081", "diplomacy"),
            ("10", "conflict"),
            ("14", "conflict"),
            ("142", "conflict"),
            ("1424", "conflict"),
            ("15", "force"),
            ("19", "force"),
            ("20", "force"),
            ("184", "force"),
        ],
    )
    def test_valid_codes(self, cameo_code: str, expected: str) -> None:
        assert cameo_to_coarse_type(cameo_code) == expected

    @pytest.mark.parametrize(
        "cameo_code",
        ["", "  ", "abc", "X1", None],
    )
    def test_invalid_codes_return_unknown(self, cameo_code) -> None:
        assert cameo_to_coarse_type(cameo_code) == "unknown"

    def test_prefix_boundary_05_06(self) -> None:
        """05 is cooperation, 06 is diplomacy."""
        assert cameo_to_coarse_type("05") == "cooperation"
        assert cameo_to_coarse_type("06") == "diplomacy"

    def test_prefix_boundary_09_10(self) -> None:
        """09 is diplomacy, 10 is conflict."""
        assert cameo_to_coarse_type("09") == "diplomacy"
        assert cameo_to_coarse_type("10") == "conflict"

    def test_prefix_boundary_14_15(self) -> None:
        """14 is conflict, 15 is force."""
        assert cameo_to_coarse_type("14") == "conflict"
        assert cameo_to_coarse_type("15") == "force"

    def test_out_of_range_code(self) -> None:
        """Codes above 20 have no mapping."""
        assert cameo_to_coarse_type("21") == "unknown"
        assert cameo_to_coarse_type("99") == "unknown"

    def test_zero_prefix(self) -> None:
        """Code 00 is not in any range (ranges start at 1)."""
        assert cameo_to_coarse_type("00") == "unknown"


class TestCrossSourceFingerprint:
    """Test fingerprint generation."""

    def test_deterministic(self) -> None:
        fp1 = cross_source_fingerprint("2026-01-01", "US", "conflict")
        fp2 = cross_source_fingerprint("2026-01-01", "US", "conflict")
        assert fp1 == fp2

    def test_length_32_hex(self) -> None:
        fp = cross_source_fingerprint("2026-01-01", "US", "conflict")
        assert len(fp) == 32
        assert all(c in "0123456789abcdef" for c in fp)

    def test_different_dates_differ(self) -> None:
        fp1 = cross_source_fingerprint("2026-01-01", "US", "conflict")
        fp2 = cross_source_fingerprint("2026-01-02", "US", "conflict")
        assert fp1 != fp2

    def test_different_countries_differ(self) -> None:
        fp1 = cross_source_fingerprint("2026-01-01", "US", "conflict")
        fp2 = cross_source_fingerprint("2026-01-01", "RU", "conflict")
        assert fp1 != fp2

    def test_different_types_differ(self) -> None:
        fp1 = cross_source_fingerprint("2026-01-01", "US", "conflict")
        fp2 = cross_source_fingerprint("2026-01-01", "US", "cooperation")
        assert fp1 != fp2

    def test_none_country_uses_unk(self) -> None:
        fp1 = cross_source_fingerprint("2026-01-01", None, "conflict")
        fp2 = cross_source_fingerprint("2026-01-01", "UNK", "conflict")
        assert fp1 == fp2

    def test_country_case_insensitive(self) -> None:
        fp1 = cross_source_fingerprint("2026-01-01", "us", "conflict")
        fp2 = cross_source_fingerprint("2026-01-01", "US", "conflict")
        assert fp1 == fp2

    def test_type_case_insensitive(self) -> None:
        fp1 = cross_source_fingerprint("2026-01-01", "US", "Conflict")
        fp2 = cross_source_fingerprint("2026-01-01", "US", "CONFLICT")
        assert fp1 == fp2

    def test_truncates_date_to_10_chars(self) -> None:
        fp1 = cross_source_fingerprint("2026-01-01T12:00:00Z", "US", "conflict")
        fp2 = cross_source_fingerprint("2026-01-01", "US", "conflict")
        assert fp1 == fp2


class TestCrossSourceDedupFilter:
    """Test the session-scoped dedup filter."""

    def test_first_event_always_inserted(self) -> None:
        f = CrossSourceDedupFilter()
        assert f.should_insert("2026-01-01", "US", "14", "gdelt", "g1")
        assert f.stats["checked"] == 1
        assert f.stats["suppressed"] == 0

    def test_intra_source_duplicates_pass_through(self) -> None:
        """Same source, same fingerprint -- allowed (handled elsewhere)."""
        f = CrossSourceDedupFilter()
        assert f.should_insert("2026-01-01", "US", "14", "gdelt", "g1")
        assert f.should_insert("2026-01-01", "US", "14", "gdelt", "g2")
        assert f.stats["suppressed"] == 0

    def test_cross_source_gdelt_then_acled_suppresses_gdelt(self) -> None:
        """ACLED has higher priority -- GDELT event gets flagged."""
        f = CrossSourceDedupFilter()
        # GDELT inserted first
        assert f.should_insert("2026-01-01", "US", "14", "gdelt", "g1")
        # ACLED arrives -- higher priority, replaces in tracking, logs suppression
        assert f.should_insert("2026-01-01", "US", "14", "acled", "a1")
        assert f.stats["suppressed"] == 1
        assert "gdelt->acled" in f.stats["by_source_pair"]

    def test_cross_source_acled_then_gdelt_suppresses_gdelt(self) -> None:
        """ACLED already in, GDELT is suppressed."""
        f = CrossSourceDedupFilter()
        assert f.should_insert("2026-01-01", "US", "14", "acled", "a1")
        assert not f.should_insert("2026-01-01", "US", "14", "gdelt", "g1")
        assert f.stats["suppressed"] == 1
        assert "gdelt->acled" in f.stats["by_source_pair"]

    def test_different_fingerprints_no_collision(self) -> None:
        f = CrossSourceDedupFilter()
        # Same date/country, different event type => different fingerprint
        assert f.should_insert("2026-01-01", "US", "05", "gdelt", "g1")  # cooperation
        assert f.should_insert("2026-01-01", "US", "14", "acled", "a1")  # conflict
        assert f.stats["suppressed"] == 0

    def test_stats_by_source_pair_counts(self) -> None:
        f = CrossSourceDedupFilter()
        # Create 3 collisions: gdelt loses to acled each time
        for i in range(3):
            f.should_insert(f"2026-01-0{i+1}", "US", "14", "acled", f"a{i}")
            f.should_insert(f"2026-01-0{i+1}", "US", "14", "gdelt", f"g{i}")
        assert f.stats["suppressed"] == 3
        assert f.stats["by_source_pair"]["gdelt->acled"] == 3

    def test_reset_clears_state(self) -> None:
        f = CrossSourceDedupFilter()
        f.should_insert("2026-01-01", "US", "14", "gdelt", "g1")
        f.reset()
        assert f.stats["checked"] == 0
        assert f.stats["suppressed"] == 0
        # Same fingerprint now accepted (state cleared)
        assert f.should_insert("2026-01-01", "US", "14", "acled", "a1")

    def test_ucdp_beats_gdelt(self) -> None:
        """UCDP has priority 2, GDELT has priority 1."""
        f = CrossSourceDedupFilter()
        assert f.should_insert("2026-01-01", "US", "14", "gdelt", "g1")
        assert f.should_insert("2026-01-01", "US", "14", "ucdp", "u1")
        assert f.stats["suppressed"] == 1

    def test_acled_beats_ucdp(self) -> None:
        """ACLED has priority 3, UCDP has priority 2."""
        f = CrossSourceDedupFilter()
        assert f.should_insert("2026-01-01", "US", "14", "ucdp", "u1")
        assert f.should_insert("2026-01-01", "US", "14", "acled", "a1")
        assert f.stats["suppressed"] == 1

    def test_unknown_source_lowest_priority(self) -> None:
        """Unknown sources get priority 0."""
        f = CrossSourceDedupFilter()
        assert f.should_insert("2026-01-01", "US", "14", "gdelt", "g1")
        assert not f.should_insert("2026-01-01", "US", "14", "newsapi", "n1")
        assert f.stats["suppressed"] == 1
