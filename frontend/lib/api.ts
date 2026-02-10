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

export async function uploadAudio(file: File): Promise<JobResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${BACKEND_DIRECT}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Upload failed");
  }

  return response.json();
}

export async function getJobStatus(jobId: string): Promise<JobResponse> {
  const response = await fetch(`${API_BASE}/status/${jobId}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to get status");
  }

  return response.json();
}

export function getPdfDownloadUrl(jobId: string): string {
  return `${API_BASE}/download/${jobId}`;
}

export function pollJobStatus(
  jobId: string,
  onUpdate: (job: JobResponse) => void,
  intervalMs: number = 1000
): () => void {
  let active = true;

  const poll = async () => {
    while (active) {
      try {
        const status = await getJobStatus(jobId);
        onUpdate(status);

        if (status.status === "completed" || status.status === "failed") {
          break;
        }
      } catch (error) {
        console.error("Polling error:", error);
      }

      await new Promise((resolve) => setTimeout(resolve, intervalMs));
    }
  };

  poll();

  return () => {
    active = false;
  };
}
