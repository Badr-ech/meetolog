/**
 * Tests for the JobProgress component.
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import JobProgress from "@/app/components/JobProgress";
import type { JobResponse } from "@/lib/api";

function makeJob(overrides: Partial<JobResponse> = {}): JobResponse {
  return {
    job_id: "test-job-id",
    status: "uploading",
    message: "",
    progress: 0,
    artifacts: null,
    pdf_url: null,
    error: null,
    ...overrides,
  };
}

describe("JobProgress", () => {
  it("renders uploading stage", () => {
    render(<JobProgress job={makeJob({ status: "uploading", progress: 10 })} />);
    expect(screen.getByText("10%")).toBeInTheDocument();
    expect(screen.getByText("Upload")).toBeInTheDocument();
  });

  it("renders transcribing stage with message", () => {
    render(
      <JobProgress
        job={makeJob({
          status: "transcribing",
          progress: 30,
          message: "Transcribing chunk 2/4...",
        })}
      />,
    );
    expect(screen.getByText("Transcribing chunk 2/4...")).toBeInTheDocument();
    expect(screen.getByText("30%")).toBeInTheDocument();
  });

  it("renders extracting stage", () => {
    render(
      <JobProgress job={makeJob({ status: "extracting", progress: 55 })} />,
    );
    expect(screen.getByText("55%")).toBeInTheDocument();
  });

  it("renders generating_pdf stage", () => {
    render(
      <JobProgress job={makeJob({ status: "generating_pdf", progress: 80 })} />,
    );
    expect(screen.getByText("80%")).toBeInTheDocument();
  });

  it("renders completed at 100%", () => {
    render(
      <JobProgress job={makeJob({ status: "completed", progress: 100 })} />,
    );
    expect(screen.getByText("100%")).toBeInTheDocument();
    expect(screen.getByText("Done")).toBeInTheDocument();
  });

  it("renders failed state", () => {
    render(
      <JobProgress
        job={makeJob({ status: "failed", progress: 0, message: "Processing Failed" })}
      />,
    );
    expect(screen.getByText("Processing Failed")).toBeInTheDocument();
  });

  it("falls back to PROGRESS_MAPPING pct when progress is 0", () => {
    render(
      <JobProgress job={makeJob({ status: "extracting", progress: 0 })} />,
    );
    // PROGRESS_MAPPING.extracting.pct = 50
    expect(screen.getByText("50%")).toBeInTheDocument();
  });

  it("uses real progress over mapping when progress > 0", () => {
    render(
      <JobProgress job={makeJob({ status: "extracting", progress: 60 })} />,
    );
    expect(screen.getByText("60%")).toBeInTheDocument();
  });

  it("falls back to uploading for unknown status", () => {
    render(
      <JobProgress
        job={makeJob({ status: "unknown_status" as never, progress: 0 })}
      />,
    );
    // Falls back to PROGRESS_MAPPING["uploading"].pct = 10
    expect(screen.getByText("10%")).toBeInTheDocument();
  });

  it("uses job.message over default label", () => {
    render(
      <JobProgress
        job={makeJob({ status: "uploading", progress: 10, message: "Custom msg" })}
      />,
    );
    expect(screen.getByText("Custom msg")).toBeInTheDocument();
  });

  it("renders all 5 stage dots", () => {
    render(<JobProgress job={makeJob()} />);
    const labels = ["Upload", "Transcribe", "Extract", "PDF", "Done"];
    labels.forEach((label) => {
      expect(screen.getByText(label)).toBeInTheDocument();
    });
  });
});
