/**
 * Shared type definitions for the Meetolog frontend.
 */

// ---------------------------------------------------------------------------
// Job status types (v1.1 granular progress states)
// ---------------------------------------------------------------------------

/** All valid backend processing states. */
export type JobStatus =
  | "uploading"
  | "transcribing"
  | "extracting"
  | "generating_pdf"
  | "completed"
  | "failed";

/** Metadata for a single progress stage shown in the UI. */
export interface StageInfo {
  /** User-friendly label displayed beneath the progress bar. */
  label: string;
  /** Nominal percentage for this stage (used when the backend progress is 0). */
  pct: number;
}

/**
 * Maps every known status (including legacy values) to a display label and
 * default percentage.  The frontend should prefer the real `progress` field
 * from the API when available; `pct` is a fallback.
 */
export const PROGRESS_MAPPING: Record<string, StageInfo> = {
  uploading: { label: "Uploading Audio…", pct: 10 },
  transcribing: { label: "Transcribing Audio…", pct: 25 },
  extracting: { label: "Extracting Artifacts…", pct: 50 },
  generating_pdf: { label: "Generating PDF…", pct: 75 },
  completed: { label: "Processing Complete!", pct: 100 },
  failed: { label: "Processing Failed", pct: 0 },
  // Legacy fallbacks for cached jobs written before v1.1
  pending: { label: "Uploading Audio…", pct: 10 },
  processing: { label: "Processing…", pct: 25 },
};

// ---------------------------------------------------------------------------
// Badge types
// ---------------------------------------------------------------------------

/** Visual variant for artifact source badges. */
export type BadgeVariant = "explicit" | "inferred" | "default";

/** Confidence-level bucket used for colour coding. */
export type ConfidenceLevel = "high" | "medium" | "low" | "unknown";

/** Props for the reusable ArtifactBadge component. */
export interface ArtifactBadgeProps {
  /** Determines the colour scheme of the badge. */
  variant: BadgeVariant;
  /** Text displayed inside the badge. */
  label: string;
  /** Optional confidence score (0–100) rendered alongside the label. */
  confidenceScore?: number;
}

/** Props for the ConfidenceIndicator component. */
export interface ConfidenceIndicatorProps {
  /** Raw score between 0.0 and 1.0 (backend value). */
  score: number | null | undefined;
}
