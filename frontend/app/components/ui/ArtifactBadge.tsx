import type {
  ArtifactBadgeProps,
  BadgeVariant,
  ConfidenceIndicatorProps,
  ConfidenceLevel,
} from "@/types";
import styles from "./ArtifactBadge.module.css";

/**
 * Map a variant key to its CSS-module class name.
 * Falls back to `default` for any unrecognised value.
 */
const variantClass: Record<BadgeVariant, string> = {
  explicit: styles.explicit,
  inferred: styles.inferred,
  default: styles.default,
};

/**
 * Map a confidence level to its CSS-module class name.
 */
const confidenceClass: Record<ConfidenceLevel, string> = {
  high: styles.confidenceHigh,
  medium: styles.confidenceMedium,
  low: styles.confidenceLow,
  unknown: styles.confidenceUnknown,
};

/**
 * Convert a raw 0.0 – 1.0 score to a discrete confidence level.
 */
function toConfidenceLevel(score: number | null | undefined): ConfidenceLevel {
  if (score == null) return "unknown";
  if (score >= 0.8) return "high";
  if (score >= 0.5) return "medium";
  return "low";
}

/**
 * Small pill badge that visually distinguishes Explicit from Inferred
 * execution tasks.
 *
 * When `confidenceScore` is provided it is appended to the label
 * and also exposed via a `title` tooltip.
 */
export default function ArtifactBadge({
  variant,
  label,
  confidenceScore,
}: ArtifactBadgeProps) {
  const cls = variantClass[variant] ?? variantClass.default;

  const displayLabel =
    confidenceScore !== undefined
      ? `${label} (${Math.round(confidenceScore)}%)`
      : label;

  const tooltip =
    confidenceScore !== undefined
      ? `${label} — confidence ${Math.round(confidenceScore)}%`
      : label;

  return (
    <span className={`${styles.badge} ${cls}`} title={tooltip}>
      {displayLabel}
    </span>
  );
}

/**
 * Colour-coded confidence score pill.
 *
 * - Score >= 0.8  → green
 * - Score >= 0.5  → amber
 * - Score < 0.5   → red
 * - null/undefined → gray "N/A"
 */
export function ConfidenceIndicator({ score }: ConfidenceIndicatorProps) {
  const level = toConfidenceLevel(score);
  const cls = confidenceClass[level];
  const pct = score != null ? `${Math.round(score * 100)}%` : "N/A";
  const tooltip = score != null ? `Confidence: ${Math.round(score * 100)}%` : "Confidence: N/A";

  return (
    <span className={`${styles.badge} ${cls}`} title={tooltip}>
      {pct}
    </span>
  );
}
