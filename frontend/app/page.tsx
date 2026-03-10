"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import {
  getPresignedUploadUrl,
  uploadToS3WithProgress,
  enqueueJob,
  pollJobStatus,
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
  const cancelPollRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    return () => {
      cancelPollRef.current?.();
    };
  }, []);

  const handleFileReady = useCallback(async (file: File) => {
    setIsUploading(true);
    setError(null);
    setJob(null);
    setS3UploadProgress(null);

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

  const isProcessing = !!(job && !["completed", "failed"].includes(job.status));
  const isCompleted = job?.status === "completed";

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

        {job && !isCompleted && !error && s3UploadProgress === null && (
          <JobProgress job={job} />
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
