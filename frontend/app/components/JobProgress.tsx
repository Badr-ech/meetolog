"use client";

import { PROGRESS_MAPPING } from "@/types";
import type { JobResponse } from "@/lib/api";
import styles from "./JobProgress.module.css";

/** Ordered pipeline stages rendered as dots beneath the progress bar. */
const STAGE_ORDER = [
  "uploading",
  "transcribing",
  "extracting",
  "generating_pdf",
  "completed",
] as const;

/** Abbreviated labels for the dot indicators. */
const STAGE_SHORT_LABELS: Record<string, string> = {
  uploading: "Upload",
  transcribing: "Transcribe",
  extracting: "Extract",
  generating_pdf: "PDF",
  completed: "Done",
};

interface JobProgressProps {
  /** Current job state from the polling response. */
  job: JobResponse;
}

/**
 * Stage-based progress bar driven by the granular v1.1 status model.
 *
 * Displays a filled progress bar, a user-friendly stage label, the
 * numeric percentage, and a row of stage-indicator dots.
 */
export default function JobProgress({ job }: JobProgressProps) {
  const stage = PROGRESS_MAPPING[job.status] ?? PROGRESS_MAPPING["uploading"];
  const pct = job.progress > 0 ? job.progress : stage.pct;
  const label = job.message || stage.label;

  // Index of the current stage in the ordered list (-1 if not found).
  const currentIdx = STAGE_ORDER.indexOf(
    job.status as (typeof STAGE_ORDER)[number],
  );

  return (
    <div className={`card ${styles.wrapper}`}>
      {/* Header: label + percentage */}
      <div className={styles.header}>
        <span className={styles.stageLabel}>{label}</span>
        <span className={styles.pct}>{pct}%</span>
      </div>

      {/* Progress bar */}
      <div className="progress-bar">
        <div
          className="progress-bar-fill"
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Stage dots */}
      <div className={styles.stages}>
        {STAGE_ORDER.map((s, idx) => {
          const isCompleted = currentIdx > idx;
          const isActive = currentIdx === idx;
          const dotCls = [
            styles.dot,
            isCompleted ? styles.dotCompleted : "",
            isActive ? styles.dotActive : "",
          ]
            .filter(Boolean)
            .join(" ");

          const nameCls = [
            styles.stageName,
            isCompleted ? styles.stageNameCompleted : "",
            isActive ? styles.stageNameActive : "",
          ]
            .filter(Boolean)
            .join(" ");

          return (
            <div key={s} className={styles.stage}>
              <span className={dotCls} />
              <span className={nameCls}>{STAGE_SHORT_LABELS[s]}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
