/**
 * API utilities for communicating with the Meetolog backend.
 */

const API_BASE = "/api";

const BACKEND_DIRECT = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface JobResponse {
  job_id: string;
  status: "pending" | "transcribing" | "extracting" | "generating_pdf" | "completed" | "failed";
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
}

export interface Task {
  id: string;
  title: string;
  description: string;
  assignee: string | null;
  priority: "low" | "medium" | "high" | "critical";
  status: "todo" | "in_progress" | "blocked" | "done";
  due_date: string | null;
}

export interface Decision {
  id: string;
  title: string;
  description: string;
  made_by: string | null;
  rationale: string;
  timestamp: string | null;
}

export interface Blocker {
  id: string;
  title: string;
  description: string;
  affected_tasks: string[];
  owner: string | null;
  resolution_plan: string;
}

export interface ActionItem {
  id: string;
  description: string;
  assignee: string | null;
  due_date: string | null;
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

  const response = await fetch(`${BACKEND_DIRECT}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await safeJson<{ detail?: string }>(response);
    throw new Error(error?.detail || `Upload failed (HTTP ${response.status})`);
  }

  return response.json();
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

export function getPdfDownloadUrl(jobId: string): string {
  return `${BACKEND_DIRECT}/download/${jobId}`;
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

        if (status.status === "completed" || status.status === "failed") {
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
