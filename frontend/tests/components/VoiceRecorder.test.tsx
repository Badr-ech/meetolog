import React from "react";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import VoiceRecorder from "@/app/components/recorder/VoiceRecorder";

/* ------------------------------------------------------------------ */
/* MediaRecorder / getUserMedia stubs                                  */
/* ------------------------------------------------------------------ */

class MockMediaRecorder {
  state = "inactive" as "inactive" | "recording" | "paused";
  ondataavailable: ((e: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;

  static isTypeSupported = jest.fn(() => true);

  start = jest.fn(() => {
    this.state = "recording";
  });

  stop = jest.fn(() => {
    this.state = "inactive";
    const chunk = new Blob([new Uint8Array(256)], { type: "audio/webm" });
    this.ondataavailable?.({ data: chunk });
    this.onstop?.();
  });

  addEventListener = jest.fn();
  removeEventListener = jest.fn();
}

function createMockStream(): MediaStream {
  const track = {
    kind: "audio",
    stop: jest.fn(),
    enabled: true,
    getSettings: () => ({ channelCount: 1, sampleRate: 16000 }),
  } as unknown as MediaStreamTrack;

  return {
    getTracks: () => [track],
    getAudioTracks: () => [track],
    getVideoTracks: () => [],
    addTrack: jest.fn(),
    removeTrack: jest.fn(),
    clone: jest.fn(),
    active: true,
    id: "mock-stream",
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
    onaddtrack: null,
    onremovetrack: null,
  } as unknown as MediaStream;
}

/* ------------------------------------------------------------------ */
/* AudioContext stubs                                                   */
/* ------------------------------------------------------------------ */

function createMockAudioContext() {
  const analyser = {
    fftSize: 2048,
    frequencyBinCount: 1024,
    getByteTimeDomainData: jest.fn((arr: Uint8Array) => arr.fill(128)),
    connect: jest.fn(),
    disconnect: jest.fn(),
  };

  const source = {
    connect: jest.fn(),
    disconnect: jest.fn(),
  };

  return {
    createAnalyser: jest.fn(() => analyser),
    createMediaStreamSource: jest.fn(() => source),
    close: jest.fn(),
    state: "running",
    sampleRate: 48000,
    destination: {},
    decodeAudioData: jest.fn(),
  } as unknown as AudioContext;
}

/* ------------------------------------------------------------------ */
/* Globals setup                                                       */
/* ------------------------------------------------------------------ */

beforeAll(() => {
  Object.defineProperty(global, "MediaRecorder", { value: MockMediaRecorder, writable: true });
  Object.defineProperty(global, "AudioContext", {
    value: jest.fn(() => createMockAudioContext()),
    writable: true,
  });
  Object.defineProperty(navigator, "mediaDevices", {
    value: { getUserMedia: jest.fn(() => Promise.resolve(createMockStream())) },
    writable: true,
  });

  HTMLCanvasElement.prototype.getContext = jest.fn(() => ({
    fillStyle: "",
    fillRect: jest.fn(),
    lineWidth: 0,
    strokeStyle: "",
    beginPath: jest.fn(),
    moveTo: jest.fn(),
    lineTo: jest.fn(),
    stroke: jest.fn(),
  })) as unknown as typeof HTMLCanvasElement.prototype.getContext;

  global.requestAnimationFrame = jest.fn((cb) => {
    cb(0);
    return 0;
  }) as unknown as typeof requestAnimationFrame;
  global.cancelAnimationFrame = jest.fn();
});

/* ------------------------------------------------------------------ */
/* Mock audio conversion (no real AudioContext decoding in JSDOM)       */
/* ------------------------------------------------------------------ */

jest.mock("@/lib/audio", () => ({
  downsampleFile: jest.fn(async (file: File) => file),
  convertRecordingToWav: jest.fn(
    async () => new File([new Uint8Array(128)], "recording.wav", { type: "audio/wav" })
  ),
}));

/* ------------------------------------------------------------------ */
/* Tests                                                               */
/* ------------------------------------------------------------------ */

describe("VoiceRecorder", () => {
  const onFileReady = jest.fn();

  beforeEach(() => {
    onFileReady.mockClear();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it("renders Record and Upload File tabs", () => {
    render(<VoiceRecorder onFileReady={onFileReady} />);
    expect(screen.getByText("Record")).toBeInTheDocument();
    expect(screen.getByText("Upload File")).toBeInTheDocument();
  });

  it("shows microphone permission prompt by default", () => {
    render(<VoiceRecorder onFileReady={onFileReady} />);
    expect(screen.getByText(/Microphone access is required/i)).toBeInTheDocument();
    expect(screen.getByText("Enable Microphone")).toBeInTheDocument();
  });

  it("requests mic permission and transitions to recording UI", async () => {
    render(<VoiceRecorder onFileReady={onFileReady} />);

    await act(async () => {
      fireEvent.click(screen.getByText("Enable Microphone"));
    });

    expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledWith(
      expect.objectContaining({ audio: expect.any(Object) })
    );

    expect(screen.getByLabelText(/start recording/i)).toBeInTheDocument();
  });

  it("shows denied message when getUserMedia rejects", async () => {
    (navigator.mediaDevices.getUserMedia as jest.Mock).mockRejectedValueOnce(
      new DOMException("NotAllowedError")
    );

    render(<VoiceRecorder onFileReady={onFileReady} />);

    await act(async () => {
      fireEvent.click(screen.getByText("Enable Microphone"));
    });

    expect(screen.getByText(/Microphone access was denied/i)).toBeInTheDocument();
  });

  it("starts and stops recording, delivering a WAV file", async () => {
    render(<VoiceRecorder onFileReady={onFileReady} />);

    await act(async () => {
      fireEvent.click(screen.getByText("Enable Microphone"));
    });

    const recordBtn = screen.getByLabelText(/start recording/i);

    await act(async () => {
      fireEvent.click(recordBtn);
    });

    expect(screen.getByText("Recording...")).toBeInTheDocument();

    const stopBtn = screen.getByLabelText(/stop recording/i);

    await act(async () => {
      fireEvent.click(stopBtn);
    });

    await waitFor(() => {
      expect(onFileReady).toHaveBeenCalledTimes(1);
    });

    const deliveredFile: File = onFileReady.mock.calls[0][0];
    expect(deliveredFile.name).toBe("recording.wav");
  });

  it("disables controls when disabled prop is true", () => {
    render(<VoiceRecorder onFileReady={onFileReady} disabled />);
    expect(screen.getByText("Enable Microphone")).toBeDisabled();
  });

  it("switches to Upload tab and shows file input", () => {
    render(<VoiceRecorder onFileReady={onFileReady} />);

    fireEvent.click(screen.getByText("Upload File"));

    expect(screen.getByText(/Drop audio file here/i)).toBeInTheDocument();
    expect(screen.getByText("Process Recording")).toBeInTheDocument();
  });

  it("submit button is disabled until a file is selected", () => {
    render(<VoiceRecorder onFileReady={onFileReady} />);
    fireEvent.click(screen.getByText("Upload File"));

    expect(screen.getByText("Process Recording")).toBeDisabled();
  });
});
