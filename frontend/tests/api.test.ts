/**
 * Tests for the API client functions in lib/api.ts.
 *
 * All network calls are intercepted via global fetch mocks.
 * No real HTTP requests are made.
 */

import {
  uploadAudio,
  getJobStatus,
  updateArtifacts,
  pollJobStatus,
  getPdfDownloadUrl,
  getJiraExportUrl,
} from "@/lib/api";
import type { JobResponse, MeetingArtifacts } from "@/lib/api";

// -------------------------------------------------------------------------
// Helpers — Response-like objects (jsdom lacks the Response constructor)
// -------------------------------------------------------------------------

const SAMPLE_JOB: JobResponse = {
  job_id: "abc-123",
  status: "uploading",
  message: "Queued",
  progress: 0,
  artifacts: null,
  pdf_url: null,
  error: null,
};

const COMPLETED_JOB: JobResponse = {
  ...SAMPLE_JOB,
  status: "completed",
  progress: 100,
  message: "Done",
};

/** Build a minimal fetch-Response-like object that satisfies what api.ts uses. */
function fakeResponse(body: unknown, status = 200) {
  const jsonBody = JSON.stringify(body);
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (_h: string) => "application/json" },
    json: () => Promise.resolve(JSON.parse(jsonBody)),
    text: () => Promise.resolve(jsonBody),
  };
}

/** Build a Response-like object whose json() rejects (non-JSON body). */
function textResponse(text: string, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (_h: string) => "text/plain" },
    json: () => Promise.reject(new SyntaxError("Unexpected token")),
    text: () => Promise.resolve(text),
  };
}

// -------------------------------------------------------------------------
// Setup
// -------------------------------------------------------------------------

const originalFetch = global.fetch;

beforeEach(() => {
  global.fetch = jest.fn();
});

afterEach(() => {
  global.fetch = originalFetch;
  jest.restoreAllMocks();
});

// -------------------------------------------------------------------------
// uploadAudio
// -------------------------------------------------------------------------

describe("uploadAudio", () => {
  it("sends FormData and returns JobResponse", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce(fakeResponse(SAMPLE_JOB));

    const file = new File([new Uint8Array(64)], "audio.wav", {
      type: "audio/wav",
    });
    const result = await uploadAudio(file);

    expect(global.fetch).toHaveBeenCalledTimes(1);
    expect(result.job_id).toBe("abc-123");
  });

  it("throws on HTTP error with detail", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce(
      fakeResponse({ detail: "Unsupported file type" }, 400),
    );

    const file = new File([new Uint8Array(64)], "doc.pdf", {
      type: "application/pdf",
    });

    await expect(uploadAudio(file)).rejects.toThrow("Unsupported file type");
  });

  it("throws generic message when body is not JSON", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce(
      textResponse("Bad Gateway", 502),
    );

    const file = new File([new Uint8Array(64)], "audio.wav");
    await expect(uploadAudio(file)).rejects.toThrow(/502/);
  });
});

// -------------------------------------------------------------------------
// getJobStatus
// -------------------------------------------------------------------------

describe("getJobStatus", () => {
  it("returns job data on success", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce(fakeResponse(SAMPLE_JOB));

    const result = await getJobStatus("abc-123");
    expect(result.status).toBe("uploading");
  });

  it("throws on 404", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce(
      fakeResponse({ detail: "Job not found" }, 404),
    );

    await expect(getJobStatus("missing")).rejects.toThrow("Job not found");
  });

  it("throws on non-JSON body", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce(textResponse("OK", 200));

    await expect(getJobStatus("abc")).rejects.toThrow("Invalid response");
  });
});

// -------------------------------------------------------------------------
// updateArtifacts
// -------------------------------------------------------------------------

describe("updateArtifacts", () => {
  const minimalArtifacts = {
    meeting_id: "id",
    meeting_title: "Test",
    meeting_date: "2025-01-01",
    duration_minutes: null,
    participants: [],
    summary: "",
    user_stories: [],
    tasks: [],
    decisions: [],
    blockers: [],
    action_items: [],
    execution_tasks: [],
    transcript: "",
  } as MeetingArtifacts;

  it("sends PUT with JSON body", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce(fakeResponse(COMPLETED_JOB));

    const result = await updateArtifacts("abc-123", minimalArtifacts);
    expect(result.status).toBe("completed");

    const [, init] = (global.fetch as jest.Mock).mock.calls[0];
    expect(init.method).toBe("PUT");
    expect(init.headers["Content-Type"]).toBe("application/json");
  });

  it("throws on 422 validation error", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce(
      fakeResponse({ detail: "Validation failed" }, 422),
    );

    await expect(
      updateArtifacts("abc-123", minimalArtifacts),
    ).rejects.toThrow("Validation failed");
  });

  it("throws on non-JSON response body", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce(textResponse("OK", 200));

    await expect(
      updateArtifacts("abc-123", minimalArtifacts),
    ).rejects.toThrow("Invalid response");
  });
});

// -------------------------------------------------------------------------
// URL builders
// -------------------------------------------------------------------------

describe("URL helpers", () => {
  it("getPdfDownloadUrl includes job ID", () => {
    const url = getPdfDownloadUrl("job-42");
    expect(url).toContain("/download/job-42");
  });

  it("getJiraExportUrl includes job ID", () => {
    const url = getJiraExportUrl("job-42");
    expect(url).toContain("/export/jira/job-42");
  });
});

// -------------------------------------------------------------------------
// pollJobStatus  (uses real timers to avoid fake-timer / async-loop issues)
// -------------------------------------------------------------------------

describe("pollJobStatus", () => {
  it("stops polling when status is completed", async () => {
    let callCount = 0;

    (global.fetch as jest.Mock).mockImplementation(async () => {
      callCount++;
      if (callCount < 3) return fakeResponse(SAMPLE_JOB);
      return fakeResponse(COMPLETED_JOB);
    });

    const updates: JobResponse[] = [];
    // Use a very short interval so the test finishes fast with real timers
    const stop = pollJobStatus("abc", (job) => updates.push(job), 5);

    // Wait long enough for several poll cycles
    await new Promise((r) => setTimeout(r, 200));
    stop();

    expect(updates.some((u) => u.status === "completed")).toBe(true);
  });

  it("reports failure after too many consecutive errors", async () => {
    // Mock getJobStatus (called via fetch) to always throw
    (global.fetch as jest.Mock).mockRejectedValue(new Error("Network error"));

    const updates: JobResponse[] = [];
    // Use tiny interval + low max errors to speed up the test
    // The source hard-codes MAX_CONSECUTIVE_ERRORS=30, so we wait enough time
    const stop = pollJobStatus("abc", (job) => updates.push(job), 5);

    // Wait long enough for 30+ error cycles (each cycle: fetch throw + short delay)
    await new Promise((r) => setTimeout(r, 3000));
    stop();

    const failedUpdate = updates.find((u) => u.status === "failed");
    expect(failedUpdate).toBeDefined();
    expect(failedUpdate!.error).toContain("Could not reach the server");
  }, 10000);
});
