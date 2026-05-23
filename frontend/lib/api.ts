/**
 * API utilities for communicating with the Meetolog backend.
 */

import type { JobStatus } from "@/types";

const API_BASE = "/api";

export interface JobResponse {
  job_id: string;
  status: JobStatus;
  message: string;
  progress: number;
  artifacts: MeetingArtifacts | null;
  pdf_url: string | null;
  error: string | null;
}

export interface MeetingArtifacts {
  meeting_id: string;
  meeting_title: string;
  meeting_date: string;
  duration_minutes: number | null;
  participants: string[];
  summary: string;
  user_stories: UserStory[];
  tasks: Task[];
  decisions: Decision[];
  blockers: Blocker[];
  action_items: ActionItem[];
  execution_tasks: ActionableTask[];
  transcript: string;
}

export interface UserStory {
  id: string;
  title: string;
  as_a: string;
  i_want: string;
  so_that: string;
  acceptance_criteria: string[];
  priority: "low" | "medium" | "high" | "critical";
  story_points: number | null;
  confidence_score?: number | null;
}

export interface Task {
  id: string;
  title: string;
  description: string;
  assignee: string | null;
  priority: "low" | "medium" | "high" | "critical";
  status: "todo" | "in_progress" | "blocked" | "done";
  due_date: string | null;
  confidence_score?: number | null;
}

export interface Decision {
  id: string;
  title: string;
  description: string;
  made_by: string | null;
  rationale: string;
  timestamp: string | null;
  confidence_score?: number | null;
}

export interface Blocker {
  id: string;
  title: string;
  description: string;
  affected_tasks: string[];
  owner: string | null;
  resolution_plan: string;
  confidence_score?: number | null;
}

export interface ActionItem {
  id: string;
  description: string;
  assignee: string | null;
  due_date: string | null;
  confidence_score?: number | null;
}

export interface ActionableTask {
  title: string;
  description: string;
  owner_role: string;
  priority: "High" | "Medium" | "Low";
  task_source: "Explicit" | "Inferred";
  dependencies: string[];
  confidence_score?: number | null;
}

/**
 * Safely parse a JSON response, returning null if the body isn't valid JSON
 * (e.g. HTML error pages from reverse proxies returning 502/503).
 */
async function safeJson<T = unknown>(response: Response): Promise<T | null> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

export async function uploadAudio(file: File): Promise<JobResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await safeJson<{ detail?: string }>(response);
    throw new Error(error?.detail || `Upload failed (HTTP ${response.status})`);
  }

  return response.json();
}

export interface PresignedUploadData {
  url: string;
  fields: Record<string, string>;
  s3_key: string;
}

/**
 * Request a presigned S3 POST payload for a direct browser-to-S3 upload.
 *
 * @param filename  Original filename as reported by the File API.
 * @param fileType  MIME type of the file (e.g. `"audio/mpeg"`).
 * @param fileSize  Byte length of the file.
 */
export async function getPresignedUploadUrl(
  filename: string,
  fileType: string,
  fileSize: number,
): Promise<PresignedUploadData> {
  const response = await fetch(`${API_BASE}/upload/presign`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename, file_type: fileType, file_size: fileSize }),
  });

  if (!response.ok) {
    const error = await safeJson<{ detail?: string }>(response);
    throw new Error(error?.detail || `Failed to get upload URL (HTTP ${response.status})`);
  }

  const data = await safeJson<PresignedUploadData>(response);
  if (!data) throw new Error("Invalid presign response from server");
  return data;
}

/**
 * Upload a file directly to S3 using a presigned POST payload.
 *
 * Progress events are emitted via `onProgress` as integer percentages (0–100).
 * Uses `XMLHttpRequest` rather than `fetch` because the Fetch API does not
 * expose granular upload progress.
 *
 * @param presignedUrl  The S3 endpoint URL from the presign response.
 * @param fields        The pre-signed form fields that must precede the file.
 * @param file          The audio file to upload.
 * @param onProgress    Callback receiving the upload percentage (0–100).
 */
export function uploadToS3WithProgress(
  presignedUrl: string,
  fields: Record<string, string>,
  file: File,
  onProgress: (percent: number) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const formData = new FormData();
    for (const [key, value] of Object.entries(fields)) {
      formData.append(key, value);
    }
    formData.append("file", file);

    const xhr = new XMLHttpRequest();

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    };

    xhr.onload = () => {
      if (xhr.status === 200 || xhr.status === 204) {
        onProgress(100);
        resolve();
      } else {
        reject(new Error(`S3 upload failed (HTTP ${xhr.status})`));
      }
    };

    xhr.onerror = () => reject(new Error("S3 upload failed: network error"));
    xhr.onabort = () => reject(new Error("S3 upload aborted"));

    xhr.open("POST", presignedUrl);
    xhr.send(formData);
  });
}

/**
 * Notify the backend to enqueue a transcription job for a file already
 * uploaded to S3 via the presigned POST flow.
 *
 * @param s3Key    The S3 object key returned by `POST /upload/presign`.
 * @param fileName Original filename for metadata storage.
 * @param fileSize File size in bytes.
 */
export async function enqueueJob(
  s3Key: string,
  fileName: string,
  fileSize: number,
): Promise<JobResponse> {
  const response = await fetch(`${API_BASE}/jobs/enqueue`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ s3_key: s3Key, file_name: fileName, file_size: fileSize }),
  });

  if (!response.ok) {
    const error = await safeJson<{ detail?: string }>(response);
    throw new Error(error?.detail || `Failed to enqueue job (HTTP ${response.status})`);
  }

  const data = await safeJson<JobResponse>(response);
  if (!data) throw new Error("Invalid enqueue response from server");
  return data;
}

export async function getJobStatus(jobId: string): Promise<JobResponse> {
  // Use Vercel proxy to avoid CORS issues when backend is down (502/503 don't include CORS headers)
  const response = await fetch(`${API_BASE}/status/${jobId}`);

  if (!response.ok) {
    const error = await safeJson<{ detail?: string }>(response);
    throw new Error(error?.detail || `Failed to get status (HTTP ${response.status})`);
  }

  const data = await safeJson<JobResponse>(response);
  if (!data) {
    throw new Error("Invalid response from server");
  }
  return data;
}

/**
 * Request cancellation of a queued or in-progress transcription job.
 *
 * The endpoint is idempotent: calling it on a job that is already
 * ``cancelled`` succeeds with the current job state.  It returns HTTP 409
 * for terminal jobs (``completed`` / ``failed``) and HTTP 404 when the
 * job ID is unknown.
 *
 * @param jobId  UUID of the job to cancel.
 * @returns The updated JobResponse reflecting the cancelled state.
 * @throws Error when the server returns a non-2xx status.
 */
export async function cancelJob(jobId: string): Promise<JobResponse> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}/cancel`, {
    method: "POST",
  });

  if (!response.ok) {
    const error = await safeJson<{ detail?: string }>(response);
    throw new Error(
      error?.detail || `Failed to cancel job (HTTP ${response.status})`,
    );
  }

  const data = await safeJson<JobResponse>(response);
  if (!data) throw new Error("Invalid cancel response from server");
  return data;
}

export function getPdfDownloadUrl(jobId: string): string {
  return `${API_BASE}/download/${jobId}`;
}

export function getJiraExportUrl(jobId: string): string {
  return `${API_BASE}/export/jira/${jobId}`;
}

export async function updateArtifacts(
  jobId: string,
  artifacts: MeetingArtifacts,
): Promise<JobResponse> {
  const response = await fetch(`${API_BASE}/artifacts/${jobId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(artifacts),
  });

  if (!response.ok) {
    const error = await safeJson<{ detail?: string }>(response);
    throw new Error(error?.detail || `Failed to update artifacts (HTTP ${response.status})`);
  }

  const data = await safeJson<JobResponse>(response);
  if (!data) {
    throw new Error("Invalid response from server");
  }
  return data;
}

export function pollJobStatus(
  jobId: string,
  onUpdate: (job: JobResponse) => void,
  intervalMs: number = 1000
): () => void {
  let active = true;
  let consecutiveErrors = 0;
  const MAX_CONSECUTIVE_ERRORS = 30; // Stop after 30 consecutive failures (~30s)

  const poll = async () => {
    while (active) {
      try {
        const status = await getJobStatus(jobId);
        consecutiveErrors = 0; // Reset on success
        onUpdate(status);

        if (
          status.status === "completed" ||
          status.status === "failed" ||
          status.status === "cancelled"
        ) {
          break;
        }
      } catch (error) {
        consecutiveErrors++;
        console.error(`Polling error (${consecutiveErrors}/${MAX_CONSECUTIVE_ERRORS}):`, error);

        if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
          // Backend is likely down for good — report failure to UI
          onUpdate({
            job_id: jobId,
            status: "failed",
            message: "Lost connection to server",
            progress: 0,
            artifacts: null,
            pdf_url: null,
            error: "Could not reach the server. The backend may have restarted — please try uploading again.",
          });
          break;
        }
      }

      // Back off slightly on errors: 1s normally, up to 3s during errors
      const delay = consecutiveErrors > 0
        ? Math.min(intervalMs * (1 + consecutiveErrors * 0.5), 3000)
        : intervalMs;
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  };

  poll();

  return () => {
    active = false;
  };
}
