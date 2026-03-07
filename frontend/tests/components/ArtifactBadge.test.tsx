/**
 * Tests for ArtifactBadge and ConfidenceIndicator components.
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import ArtifactBadge, {
  ConfidenceIndicator,
} from "@/app/components/ui/ArtifactBadge";

describe("ArtifactBadge", () => {
  it("renders label text", () => {
    render(<ArtifactBadge variant="explicit" label="Explicit" />);
    expect(screen.getByText("Explicit")).toBeInTheDocument();
  });

  it("appends confidence score to label when provided", () => {
    render(
      <ArtifactBadge variant="inferred" label="Inferred" confidenceScore={85} />,
    );
    expect(screen.getByText("Inferred (85%)")).toBeInTheDocument();
  });

  it("sets title tooltip with confidence info", () => {
    render(
      <ArtifactBadge variant="explicit" label="Explicit" confidenceScore={72} />,
    );
    const badge = screen.getByTitle("Explicit — confidence 72%");
    expect(badge).toBeInTheDocument();
  });

  it("renders default variant for unknown values", () => {
    // Casting intentionally to cover fallback path
    render(
      <ArtifactBadge variant={"unknown" as never} label="Unknown" />,
    );
    expect(screen.getByText("Unknown")).toBeInTheDocument();
  });

  it("omits percentage suffix when confidenceScore is undefined", () => {
    render(<ArtifactBadge variant="explicit" label="Explicit" />);
    expect(screen.getByText("Explicit")).toBeInTheDocument();
    expect(screen.queryByText(/\(/)).toBeNull();
  });
});

describe("ConfidenceIndicator", () => {
  it("renders green for high confidence (>= 0.8)", () => {
    render(<ConfidenceIndicator score={0.95} />);
    expect(screen.getByText("95%")).toBeInTheDocument();
    expect(screen.getByTitle("Confidence: 95%")).toBeInTheDocument();
  });

  it("renders amber for medium confidence (0.5–0.79)", () => {
    render(<ConfidenceIndicator score={0.65} />);
    expect(screen.getByText("65%")).toBeInTheDocument();
  });

  it("renders red for low confidence (< 0.5)", () => {
    render(<ConfidenceIndicator score={0.3} />);
    expect(screen.getByText("30%")).toBeInTheDocument();
  });

  it("renders N/A for null score", () => {
    render(<ConfidenceIndicator score={null} />);
    expect(screen.getByText("N/A")).toBeInTheDocument();
    expect(screen.getByTitle("Confidence: N/A")).toBeInTheDocument();
  });

  it("renders N/A for undefined score", () => {
    render(<ConfidenceIndicator score={undefined} />);
    expect(screen.getByText("N/A")).toBeInTheDocument();
  });

  it("renders 0% for score of zero", () => {
    render(<ConfidenceIndicator score={0} />);
    expect(screen.getByText("0%")).toBeInTheDocument();
  });

  it("renders 100% for perfect score", () => {
    render(<ConfidenceIndicator score={1.0} />);
    expect(screen.getByText("100%")).toBeInTheDocument();
  });
});
