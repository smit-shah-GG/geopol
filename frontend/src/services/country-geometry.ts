/**
 * Country geometry service -- loads GeoJSON boundaries, normalizes ISO codes,
 * provides centroid lookup and feature-by-ISO retrieval.
 *
 * Adapted from WM's country-geometry.ts (303 lines). Removed: WM country
 * metadata, Tauri cache, WM-specific list management, name-in-text matching.
 * Added: getCentroid(), getFeatureByIso(), isLoaded() API.
 */

import type {
  FeatureCollection,
  Feature,
  Geometry,
  GeoJsonProperties,
  Position,
} from 'geojson';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GEOJSON_URL = '/data/countries.geojson';

/** Political entity overrides: map non-standard codes to canonical ISO-2 */
const POLITICAL_OVERRIDES: Record<string, string> = { 'CN-TW': 'TW' };

/**
 * Common name aliases -> ISO-2. Used when GeoJSON NAME property is
 * the only identifier available for a feature.
 */
const NAME_ALIASES: Record<string, string> = {
  'dr congo': 'CD',
  'democratic republic of the congo': 'CD',
  'czech republic': 'CZ',
  'ivory coast': 'CI',
  "cote d'ivoire": 'CI',
  'uae': 'AE',
  'uk': 'GB',
  'usa': 'US',
  'south korea': 'KR',
  'north korea': 'KP',
  'republic of the congo': 'CG',
  'east timor': 'TL',
  'cape verde': 'CV',
  'swaziland': 'SZ',
  'burma': 'MM',
};

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/**
 * Extract a 2-element [lon, lat] coordinate from a GeoJSON Position array.
 * Returns null if the position is malformed.
 */
function toCoord(point: Position): [number, number] | null {
  if (!Array.isArray(point) || point.length < 2) return null;
  const lon = Number(point[0]);
  const lat = Number(point[1]);
  if (!Number.isFinite(lon) || !Number.isFinite(lat)) return null;
  return [lon, lat];
}

/**
 * Compute the centroid of a polygon/multipolygon as the arithmetic mean
 * of all outer-ring vertices. Not area-weighted -- good enough for marker
 * placement on 110m resolution data.
 */
function computeCentroid(geometry: Geometry | null): [number, number] | null {
  if (!geometry) return null;

  let sumLon = 0;
  let sumLat = 0;
  let count = 0;

  const addRing = (ring: Position[]): void => {
    for (const pos of ring) {
      const c = toCoord(pos);
      if (c) {
        sumLon += c[0];
        sumLat += c[1];
        count++;
      }
    }
  };

  if (geometry.type === 'Polygon') {
    const outer = geometry.coordinates[0];
    if (outer) addRing(outer);
  } else if (geometry.type === 'MultiPolygon') {
    for (const polygon of geometry.coordinates) {
      const outer = polygon[0];
      if (outer) addRing(outer);
    }
  }

  if (count === 0) return null;
  return [sumLon / count, sumLat / count];
}

// ---------------------------------------------------------------------------
// ISO code normalization
// ---------------------------------------------------------------------------

/**
 * Extract a normalized ISO 3166-1 Alpha-2 code from feature properties.
 * Checks multiple property-name conventions used by Natural Earth, GADM, etc.
 */
export function normalizeCode(
  properties: GeoJsonProperties | null | undefined,
): string | null {
  if (!properties) return null;

  // Try standard property names in priority order
  const rawCode =
    (properties['ISO_A2'] as string | undefined) ??
    (properties['ISO3166-1-Alpha-2'] as string | undefined) ??
    (properties['iso_a2'] as string | undefined) ??
    (properties['ISO_A2_EH'] as string | undefined);

  if (typeof rawCode !== 'string') return null;
  const trimmed = rawCode.trim().toUpperCase();
  if (trimmed === '-99' || trimmed === '-1') return null;
  const overridden = POLITICAL_OVERRIDES[trimmed] ?? trimmed;
  return /^[A-Z]{2}$/.test(overridden) ? overridden : null;
}

/**
 * Extract country name from feature properties.
 */
function normalizeName(
  properties: GeoJsonProperties | null | undefined,
): string | null {
  if (!properties) return null;
  const rawName =
    (properties['NAME'] as string | undefined) ??
    (properties['name'] as string | undefined) ??
    (properties['admin'] as string | undefined);
  if (typeof rawName !== 'string') return null;
  const name = rawName.trim();
  return name.length > 0 ? name : null;
}

// ---------------------------------------------------------------------------
// CountryGeometryService
// ---------------------------------------------------------------------------

interface IndexedCountry {
  code: string;
  name: string;
  centroid: [number, number];
  feature: Feature<Geometry>;
}

export class CountryGeometryService {
  private loadPromise: Promise<void> | null = null;
  private loaded = false;
  private geoJson: FeatureCollection<Geometry> | null = null;

  /** ISO-2 -> indexed country data */
  private readonly byCode = new Map<string, IndexedCountry>();
  /** ISO-3 -> ISO-2 lookup */
  private readonly iso3ToIso2 = new Map<string, string>();
  /** Lowercase name -> ISO-2 lookup */
  private readonly nameToCode = new Map<string, string>();

  // -----------------------------------------------------------------------
  // Public API
  // -----------------------------------------------------------------------

  /** Whether the GeoJSON data has been loaded successfully. */
  isLoaded(): boolean {
    return this.loaded;
  }

  /**
   * Load the countries GeoJSON file. Safe to call multiple times --
   * subsequent calls return the same promise.
   */
  async load(): Promise<void> {
    if (this.loaded) return;
    if (this.loadPromise) {
      await this.loadPromise;
      return;
    }

    this.loadPromise = this.doLoad();
    await this.loadPromise;
  }

  /**
   * Get the full FeatureCollection for deck.gl GeoJsonLayer consumption.
   * Returns null if not yet loaded.
   */
  getFeatureCollection(): FeatureCollection<Geometry> | null {
    return this.geoJson;
  }

  /**
   * Get the GeoJSON Feature for a specific country.
   * @param iso ISO 3166-1 Alpha-2 code (case-insensitive)
   */
  getFeatureByIso(iso: string): Feature<Geometry> | null {
    return this.byCode.get(iso.toUpperCase())?.feature ?? null;
  }

  /**
   * Get the centroid [lon, lat] of a country for marker placement.
   * Uses LABEL_X/LABEL_Y from Natural Earth if available, otherwise
   * falls back to computed centroid.
   * @param iso ISO 3166-1 Alpha-2 code (case-insensitive)
   */
  getCentroid(iso: string): [number, number] | null {
    return this.byCode.get(iso.toUpperCase())?.centroid ?? null;
  }

  /**
   * Get the display name for a country.
   * @param iso ISO 3166-1 Alpha-2 code (case-insensitive)
   */
  getNameByIso(iso: string): string | null {
    return this.byCode.get(iso.toUpperCase())?.name ?? null;
  }

  /**
   * Resolve an ISO-3 code to ISO-2.
   */
  iso3ToIso2Code(iso3: string): string | null {
    return this.iso3ToIso2.get(iso3.trim().toUpperCase()) ?? null;
  }

  /**
   * Resolve a country name to ISO-2 code (case-insensitive exact match).
   */
  nameToCountryCode(name: string): string | null {
    return this.nameToCode.get(name.toLowerCase().trim()) ?? null;
  }

  /**
   * Return all loaded ISO-2 codes.
   */
  getAllCodes(): string[] {
    return [...this.byCode.keys()];
  }

  // -----------------------------------------------------------------------
  // Private loading
  // -----------------------------------------------------------------------

  private async doLoad(): Promise<void> {
    try {
      const response = await fetch(GEOJSON_URL);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} loading ${GEOJSON_URL}`);
      }

      const data = (await response.json()) as FeatureCollection<Geometry>;
      if (
        !data ||
        data.type !== 'FeatureCollection' ||
        !Array.isArray(data.features)
      ) {
        console.warn('[country-geometry] Invalid GeoJSON structure');
        return;
      }

      this.geoJson = data;
      this.byCode.clear();
      this.iso3ToIso2.clear();
      this.nameToCode.clear();

      for (const feature of data.features) {
        const code = normalizeCode(feature.properties);
        const name = normalizeName(feature.properties);
        if (!code || !name) continue;

        // ISO-3 -> ISO-2 mapping
        const iso3 = feature.properties?.['ISO_A3'] as string | undefined;
        if (typeof iso3 === 'string' && /^[A-Z]{3}$/i.test(iso3.trim())) {
          this.iso3ToIso2.set(iso3.trim().toUpperCase(), code);
        }

        // Name -> code mapping
        this.nameToCode.set(name.toLowerCase(), code);

        // Centroid: prefer Natural Earth label coordinates, fall back to computed
        const labelX = feature.properties?.['LABEL_X'] as number | undefined;
        const labelY = feature.properties?.['LABEL_Y'] as number | undefined;
        let centroid: [number, number] | null = null;
        if (
          typeof labelX === 'number' &&
          typeof labelY === 'number' &&
          Number.isFinite(labelX) &&
          Number.isFinite(labelY)
        ) {
          centroid = [labelX, labelY];
        } else {
          centroid = computeCentroid(feature.geometry);
        }

        if (!centroid) continue;

        // Avoid duplicates: first feature for a code wins
        if (!this.byCode.has(code)) {
          this.byCode.set(code, { code, name, centroid, feature });
        }
      }

      // Add name aliases
      for (const [alias, code] of Object.entries(NAME_ALIASES)) {
        if (!this.nameToCode.has(alias)) {
          this.nameToCode.set(alias, code);
        }
      }

      this.loaded = true;
    } catch (err) {
      console.warn('[country-geometry] Failed to load:', err);
    }
  }
}

// ---------------------------------------------------------------------------
// Module singleton
// ---------------------------------------------------------------------------

export const countryGeometry = new CountryGeometryService();
