"use client";

import { useState, useCallback } from "react";
import { uploadAudio, pollJobStatus, getPdfDownloadUrl, JobResponse, MeetingArtifacts } from "@/lib/api";
import styles from "./page.module.css";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [job, setJob] = useState<JobResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
      setJob(null);
    }
  };

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setIsUploading(true);
    setError(null);

    try {
      // Upload the file
      const jobResponse = await uploadAudio(file);
      setJob(jobResponse);

      // Start polling for status updates
      pollJobStatus(jobResponse.job_id, (updatedJob) => {
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
  }, [file]);

  const isProcessing = job && !["completed", "failed"].includes(job.status);
  const isCompleted = job?.status === "completed";

  return (
    <main className={styles.main}>
      <div className="container">
        {/* Header */}
        <header className={styles.header}>
          <h1 className={styles.title}>📋 Meetolog</h1>
          <p className={styles.subtitle}>
            Transform meeting recordings into structured Agile artifacts
          </p>
        </header>

        {/* Upload Section */}
        <section className={`card ${styles.uploadCard}`}>
          <h2 className={styles.sectionTitle}>Upload Meeting Recording</h2>
          
          <form onSubmit={handleSubmit} className={styles.form}>
            <div className={styles.dropzone}>
              <input
                type="file"
                id="audio-file"
                accept=".mp3,.wav,.m4a,.ogg,.webm"
                onChange={handleFileChange}
                disabled={isUploading || isProcessing}
                className={styles.fileInput}
              />
              <label htmlFor="audio-file" className={styles.dropzoneLabel}>
                <span className={styles.dropzoneIcon}>🎙️</span>
                {file ? (
                  <span className={styles.fileName}>{file.name}</span>
                ) : (
                  <>
                    <span>Drop audio file here or click to browse</span>
                    <span className={styles.hint}>MP3, WAV, M4A, OGG, WebM (max 100MB)</span>
                  </>
                )}
              </label>
            </div>

            <button
              type="submit"
              disabled={!file || isUploading || isProcessing}
              className="btn btn-primary"
            >
              {isUploading ? "Uploading..." : isProcessing ? "Processing..." : "Process Recording"}
            </button>
          </form>

          {/* Error Display */}
          {error && (
            <div className={styles.error}>
              <span>⚠️</span> {error}
            </div>
          )}

          {/* Progress Display */}
          {job && !isCompleted && !error && (
            <div className={styles.progressSection}>
              <div className={styles.progressHeader}>
                <span className={styles.statusText}>{job.message}</span>
                <span className={styles.progressPercent}>{job.progress}%</span>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-bar-fill"
                  style={{ width: `${job.progress}%` }}
                />
              </div>
            </div>
          )}
        </section>

        {/* Results Section */}
        {isCompleted && job.artifacts && (
          <ResultsView artifacts={job.artifacts} jobId={job.job_id} />
        )}
      </div>
    </main>
  );
}

function ResultsView({ artifacts, jobId }: { artifacts: MeetingArtifacts; jobId: string }) {
  return (
    <section className={styles.results}>
      {/* Summary Card */}
      <div className={`card ${styles.summaryCard}`}>
        <div className={styles.summaryHeader}>
          <div>
            <h2 className={styles.meetingTitle}>{artifacts.meeting_title}</h2>
            <p className={styles.meetingMeta}>
              {new Date(artifacts.meeting_date).toLocaleDateString()} • 
              {artifacts.participants.length > 0 
                ? ` ${artifacts.participants.join(", ")}`
                : " No participants identified"}
            </p>
          </div>
          <a
            href={getPdfDownloadUrl(jobId)}
            download
            className="btn btn-primary"
          >
            📥 Download PDF
          </a>
        </div>
        {artifacts.summary && (
          <p className={styles.summary}>{artifacts.summary}</p>
        )}
      </div>

      {/* User Stories */}
      {artifacts.user_stories.length > 0 && (
        <div className="card">
          <h3 className={styles.artifactTitle}>📖 User Stories ({artifacts.user_stories.length})</h3>
          <div className={styles.artifactList}>
            {artifacts.user_stories.map((story) => (
              <div key={story.id} className={styles.storyCard}>
                <div className={styles.storyHeader}>
                  <span className={styles.storyTitle}>{story.title}</span>
                  <div className={styles.storyMeta}>
                    <span className={`badge badge-${story.priority}`}>{story.priority}</span>
                    {story.story_points && (
                      <span className={styles.storyPoints}>{story.story_points} pts</span>
                    )}
                  </div>
                </div>
                <p className={styles.storyFormat}>
                  As a <strong>{story.as_a}</strong>, I want <strong>{story.i_want}</strong>, 
                  so that <strong>{story.so_that}</strong>
                </p>
                {story.acceptance_criteria.length > 0 && (
                  <ul className={styles.criteria}>
                    {story.acceptance_criteria.map((criterion, i) => (
                      <li key={i}>{criterion}</li>
                    ))}
                  </ul>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tasks */}
      {artifacts.tasks.length > 0 && (
        <div className="card">
          <h3 className={styles.artifactTitle}>✅ Tasks ({artifacts.tasks.length})</h3>
          <div className={styles.taskList}>
            {artifacts.tasks.map((task) => (
              <div key={task.id} className={styles.taskCard}>
                <div className={styles.taskHeader}>
                  <span className={styles.taskTitle}>{task.title}</span>
                  <span className={`badge badge-${task.priority}`}>{task.priority}</span>
                </div>
                {task.description && (
                  <p className={styles.taskDesc}>{task.description}</p>
                )}
                <div className={styles.taskMeta}>
                  {task.assignee && <span>👤 {task.assignee}</span>}
                  {task.due_date && <span>📅 {task.due_date}</span>}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Decisions */}
      {artifacts.decisions.length > 0 && (
        <div className="card">
          <h3 className={styles.artifactTitle}>🎯 Decisions ({artifacts.decisions.length})</h3>
          <div className={styles.artifactList}>
            {artifacts.decisions.map((decision) => (
              <div key={decision.id} className={styles.decisionCard}>
                <h4 className={styles.decisionTitle}>{decision.title}</h4>
                <p>{decision.description}</p>
                {decision.rationale && (
                  <p className={styles.rationale}>
                    <em>Rationale: {decision.rationale}</em>
                  </p>
                )}
                {decision.made_by && (
                  <span className={styles.madeBy}>Decision by: {decision.made_by}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Blockers */}
      {artifacts.blockers.length > 0 && (
        <div className="card">
          <h3 className={styles.artifactTitle}>🚧 Blockers ({artifacts.blockers.length})</h3>
          <div className={styles.artifactList}>
            {artifacts.blockers.map((blocker) => (
              <div key={blocker.id} className={styles.blockerCard}>
                <h4 className={styles.blockerTitle}>⚠️ {blocker.title}</h4>
                <p>{blocker.description}</p>
                {blocker.resolution_plan && (
                  <p className={styles.resolution}>
                    <strong>Resolution:</strong> {blocker.resolution_plan}
                  </p>
                )}
                {blocker.owner && <span className={styles.owner}>Owner: {blocker.owner}</span>}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Action Items */}
      {artifacts.action_items.length > 0 && (
        <div className="card">
          <h3 className={styles.artifactTitle}>📌 Action Items ({artifacts.action_items.length})</h3>
          <ul className={styles.actionList}>
            {artifacts.action_items.map((item) => (
              <li key={item.id} className={styles.actionItem}>
                <span>{item.description}</span>
                {item.assignee && <span className={styles.assignee}>({item.assignee})</span>}
                {item.due_date && <span className={styles.dueDate}>Due: {item.due_date}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Raw JSON Toggle */}
      <details className="card">
        <summary className={styles.jsonToggle}>View Raw JSON</summary>
        <pre className={styles.jsonView}>
          {JSON.stringify(artifacts, null, 2)}
        </pre>
      </details>
    </section>
  );
}
