"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { uploadAudio, pollJobStatus, getPdfDownloadUrl, JobResponse, MeetingArtifacts } from "@/lib/api";
import VoiceRecorder from "./components/recorder/VoiceRecorder";
import ArtifactEditor from "./components/ArtifactEditor";
import JobProgress from "./components/JobProgress";
import styles from "./page.module.css";

export default function Home() {
  const [isUploading, setIsUploading] = useState(false);
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

    try {
      const jobResponse = await uploadAudio(file);
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

        {job && !isCompleted && !error && (
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
