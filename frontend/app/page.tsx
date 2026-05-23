"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import {
  getPresignedUploadUrl,
  uploadToS3WithProgress,
  enqueueJob,
  pollJobStatus,
  cancelJob,
  getPdfDownloadUrl,
  JobResponse,
  MeetingArtifacts,
} from "@/lib/api";
import VoiceRecorder from "./components/recorder/VoiceRecorder";
import ArtifactEditor from "./components/ArtifactEditor";
import JobProgress from "./components/JobProgress";
import styles from "./page.module.css";

export default function Home() {
  const [isUploading, setIsUploading] = useState(false);
  const [s3UploadProgress, setS3UploadProgress] = useState<number | null>(null);
  const [job, setJob] = useState<JobResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isCancelling, setIsCancelling] = useState(false);

  /**
   * Ref that always holds the latest poll-cancellation function so the
   * beforeunload handler can call it synchronously without stale closure issues.
   */
  const cancelPollRef = useRef<(() => void) | null>(null);

  /**
   * Ref that always holds the latest job ID so the beforeunload handler
   * can read it without capturing a stale value from the closure created
   * at addEventListener time.
   */
  const activeJobIdRef = useRef<string | null>(null);

  /**
   * Ref that tracks whether the current job is still in an active
   * (non-terminal) processing state.  Also kept in a ref so the
   * beforeunload handler reads the live value.
   */
  const isProcessingRef = useRef<boolean>(false);

  // Keep both derived refs up to date whenever job state changes.
  useEffect(() => {
    const inFlight =
      job !== null && !["completed", "failed", "cancelled"].includes(job.status);
    isProcessingRef.current = inFlight;
    activeJobIdRef.current = job?.job_id ?? null;
  }, [job]);

  // Stop polling when the component unmounts (e.g. route change within the SPA).
  useEffect(() => {
    return () => {
      cancelPollRef.current?.();
    };
  }, []);

  /**
   * beforeunload handler — fires when the user closes the tab, reloads the
   * page, or navigates away via the browser chrome.  Uses
   * ``navigator.sendBeacon`` so the request is guaranteed to be delivered
   * even though the page is being torn down and no response can be awaited.
   *
   * sendBeacon dispatches a POST with an empty body.  The cancel endpoint
   * requires no request payload, so this is sufficient.
   */
  useEffect(() => {
    const handleBeforeUnload = (): void => {
      const jobId = activeJobIdRef.current;
      const inFlight = isProcessingRef.current;
      if (!jobId || !inFlight) return;

      // sendBeacon is the only reliable fire-and-forget mechanism available
      // in a beforeunload context — fetch and XMLHttpRequest are not guaranteed
      // to complete when the page is unloading.
      navigator.sendBeacon(`/api/jobs/${jobId}/cancel`);
    };

    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => window.removeEventListener("beforeunload", handleBeforeUnload);
  }, []);

  const handleFileReady = useCallback(async (file: File) => {
    setIsUploading(true);
    setError(null);
    setJob(null);
    setS3UploadProgress(null);
    setIsCancelling(false);

    try {
      // Phase 1 — obtain a presigned POST payload from the backend.
      const presignData = await getPresignedUploadUrl(
        file.name,
        file.type || "audio/webm",
        file.size,
      );

      // Phase 2 — upload the file directly to S3 (backend never receives bytes).
      setS3UploadProgress(0);
      await uploadToS3WithProgress(
        presignData.url,
        presignData.fields,
        file,
        setS3UploadProgress,
      );
      setS3UploadProgress(null);

      // Phase 3 — notify the backend to enqueue the transcription job.
      const jobResponse = await enqueueJob(presignData.s3_key, file.name, file.size);
      setJob(jobResponse);

      cancelPollRef.current = pollJobStatus(jobResponse.job_id, (updatedJob) => {
        setJob(updatedJob);
        if (updatedJob.status === "failed") {
          setError(updatedJob.error || "Processing failed");
        }
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setIsUploading(false);
      setS3UploadProgress(null);
    }
  }, []);

  /**
   * Explicit cancel — called when the user clicks the "Cancel Processing"
   * button during an active polling session.
   *
   * Sequence:
   * 1. Stop the client-side poll loop immediately (prevents stale state updates).
   * 2. Optimistically update local state to ``cancelled`` so the UI responds
   *    instantly without waiting for the server round-trip.
   * 3. Call the API; update local state with the authoritative server response.
   * 4. On failure, show an error — the poll is already stopped so the user
   *    must refresh to resume monitoring if they wish.
   */
  const handleCancelJob = useCallback(async () => {
    const jobId = job?.job_id;
    if (!jobId || isCancelling) return;

    // Stop the polling loop before touching state so we do not race the
    // optimistic update with a poll response that overwrites it.
    cancelPollRef.current?.();
    cancelPollRef.current = null;

    setIsCancelling(true);

    // Optimistic update — gives the user immediate visual feedback.
    setJob((prev) =>
      prev
        ? {
            ...prev,
            status: "cancelled",
            message: "Cancelling…",
            progress: prev.progress,
          }
        : prev,
    );

    try {
      const cancelledJob = await cancelJob(jobId);
      // Replace optimistic state with the authoritative server response.
      setJob(cancelledJob);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to cancel job — please refresh the page",
      );
    } finally {
      setIsCancelling(false);
    }
  }, [job?.job_id, isCancelling]);

  /** Reset all state so the user can start a new upload from scratch. */
  const handleReset = useCallback(() => {
    cancelPollRef.current?.();
    cancelPollRef.current = null;
    setJob(null);
    setError(null);
    setIsCancelling(false);
    setS3UploadProgress(null);
  }, []);

  const isProcessing = !!(
    job && !["completed", "failed", "cancelled"].includes(job.status)
  );
  const isCompleted = job?.status === "completed";
  const isCancelled = job?.status === "cancelled";

  return (
    <main className={styles.main}>
      <div className="container">
        <header className={styles.header}>
          <h1 className={styles.title}>Meetolog</h1>
          <p className={styles.subtitle}>
            Transform meeting recordings into structured Agile artifacts
          </p>
        </header>

        <VoiceRecorder
          onFileReady={handleFileReady}
          disabled={isUploading || isProcessing}
        />

        {error && (
          <div className={`card ${styles.statusCard}`}>
            <div className={styles.error}>{error}</div>
          </div>
        )}

        {s3UploadProgress !== null && (
          <div className={`card ${styles.statusCard}`}>
            <div className={styles.progressHeader}>
              <span className={styles.statusText}>Uploading to storage…</span>
              <span className={styles.progressPercent}>{s3UploadProgress}%</span>
            </div>
            <progress
              className={styles.s3ProgressBar}
              value={s3UploadProgress}
              max={100}
            />
          </div>
        )}

        {job && !isCompleted && !isCancelled && !error && s3UploadProgress === null && (
          <>
            <JobProgress job={job} />
            {isProcessing && (
              <div className={styles.cancelWrapper}>
                <button
                  className={styles.cancelButton}
                  onClick={handleCancelJob}
                  disabled={isCancelling}
                  aria-label="Cancel processing"
                >
                  {isCancelling ? "Cancelling…" : "Cancel Processing"}
                </button>
              </div>
            )}
          </>
        )}

        {isCancelled && (
          <div className={`card ${styles.statusCard}`}>
            <div className={styles.cancelledHeader}>Processing Cancelled</div>
            <p className={styles.cancelledBody}>
              The transcription job was stopped before it completed. No artifacts
              were extracted and no PDF was generated. Your audio file remains
              stored in S3 — upload it again to start a fresh processing run.
            </p>
            <button
              className={styles.resetButton}
              onClick={handleReset}
            >
              Start New Processing
            </button>
          </div>
        )}

        {/* Results Section */}
        {isCompleted && job.artifacts && (
          <ArtifactEditor
            artifacts={job.artifacts}
            jobId={job.job_id}
            onArtifactsChange={(updated) =>
              setJob((prev) => (prev ? { ...prev, artifacts: updated } : prev))
            }
          />
        )}
      </div>
    </main>
  );
}
