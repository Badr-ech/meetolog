"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { downsampleFile, convertRecordingToWav } from "@/lib/audio";
import styles from "./VoiceRecorder.module.css";

type Mode = "record" | "upload";
type MicPermission = "prompt" | "granted" | "denied";

interface VoiceRecorderProps {
  onFileReady: (file: File) => void;
  disabled?: boolean;
}

export default function VoiceRecorder({ onFileReady, disabled }: VoiceRecorderProps) {
  const [mode, setMode] = useState<Mode>("record");
  const [micPermission, setMicPermission] = useState<MicPermission>("prompt");
  const [isRecording, setIsRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [isConverting, setIsConverting] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number>(0);
  const audioCtxRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    return () => {
      stopVisualization();
      clearTimer();
      streamRef.current?.getTracks().forEach((t) => t.stop());
      audioCtxRef.current?.close();
    };
  }, []);

  const clearTimer = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  const stopVisualization = () => {
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = 0;
    }
  };

  const drawVisualization = useCallback(() => {
    const canvas = canvasRef.current;
    const analyser = analyserRef.current;
    if (!canvas || !analyser) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      animFrameRef.current = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);

      ctx.fillStyle = "#f9fafb";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.lineWidth = 2;
      ctx.strokeStyle = "#dc2626";
      ctx.beginPath();

      const sliceWidth = canvas.width / bufferLength;
      let x = 0;
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
    };

    draw();
  }, []);

  const requestMic = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, sampleRate: 16000 },
      });
      streamRef.current = stream;
      setMicPermission("granted");

      const audioCtx = new AudioContext();
      audioCtxRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      analyserRef.current = analyser;
    } catch {
      setMicPermission("denied");
    }
  };

  const startRecording = () => {
    const stream = streamRef.current;
    if (!stream) return;

    chunksRef.current = [];
    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : "audio/webm";

    const recorder = new MediaRecorder(stream, { mimeType });
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    recorder.onstop = async () => {
      const blob = new Blob(chunksRef.current, { type: mimeType });
      setIsConverting(true);
      try {
        const wav = await convertRecordingToWav(blob);
        onFileReady(wav);
      } finally {
        setIsConverting(false);
      }
    };

    recorder.start(250);
    mediaRecorderRef.current = recorder;
    setIsRecording(true);
    setElapsed(0);
    timerRef.current = setInterval(() => setElapsed((s) => s + 1), 1000);
    drawVisualization();
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setIsRecording(false);
    clearTimer();
    stopVisualization();
  };

  const handleRecordToggle = () => {
    if (isRecording) stopRecording();
    else startRecording();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (!selected) return;

    setIsConverting(true);
    try {
      const optimized = await downsampleFile(selected);
      setUploadFile(optimized);
    } finally {
      setIsConverting(false);
    }
  };

  const handleUploadSubmit = () => {
    if (uploadFile) onFileReady(uploadFile);
  };

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m.toString().padStart(2, "0")}:${sec.toString().padStart(2, "0")}`;
  };

  return (
    <section className={`card ${styles.recorder}`}>
      <div className={styles.tabs}>
        <button
          className={`${styles.tab} ${mode === "record" ? styles.tabActive : ""}`}
          onClick={() => setMode("record")}
          disabled={disabled || isRecording}
        >
          Record
        </button>
        <button
          className={`${styles.tab} ${mode === "upload" ? styles.tabActive : ""}`}
          onClick={() => setMode("upload")}
          disabled={disabled || isRecording}
        >
          Upload File
        </button>
      </div>

      {mode === "record" && (
        <div className={styles.recordArea}>
          {micPermission === "prompt" && (
            <div className={styles.permissionPrompt}>
              <p>Microphone access is required to record audio.</p>
              <button className={styles.permissionBtn} onClick={requestMic} disabled={disabled}>
                Enable Microphone
              </button>
            </div>
          )}

          {micPermission === "denied" && (
            <p className={styles.permissionDenied}>
              Microphone access was denied. Please allow access in your browser settings.
            </p>
          )}

          {micPermission === "granted" && (
            <>
              <canvas
                ref={canvasRef}
                className={styles.visualizer}
                width={560}
                height={64}
              />

              <span className={styles.timer}>{formatTime(elapsed)}</span>

              <button
                className={styles.recordBtn}
                onClick={handleRecordToggle}
                disabled={disabled || isConverting}
                aria-label={isRecording ? "Stop recording" : "Start recording"}
              >
                <span
                  className={`${styles.recordIcon} ${isRecording ? styles.recordIconStop : ""}`}
                />
              </button>

              {isRecording && <span className={styles.recordingLabel}>Recording...</span>}
              {isConverting && <span className={styles.converting}>Converting audio...</span>}
            </>
          )}
        </div>
      )}

      {mode === "upload" && (
        <div>
          <div className={styles.dropzone}>
            <input
              type="file"
              id="audio-file"
              accept=".mp3,.wav,.m4a,.ogg,.webm"
              onChange={handleFileChange}
              disabled={disabled || isConverting}
              className={styles.fileInput}
            />
            <label htmlFor="audio-file" className={styles.dropzoneLabel}>
              <span className={styles.dropzoneIcon}>&#x1F399;</span>
              {uploadFile ? (
                <span className={styles.fileName}>{uploadFile.name}</span>
              ) : (
                <>
                  <span>Drop audio file here or click to browse</span>
                  <span className={styles.hint}>MP3, WAV, M4A, OGG, WebM (max 1 GB)</span>
                </>
              )}
            </label>
          </div>

          {isConverting && <p className={styles.converting}>Optimizing audio...</p>}

          <button
            className={`btn btn-primary ${styles.submitBtn}`}
            onClick={handleUploadSubmit}
            disabled={!uploadFile || disabled || isConverting}
          >
            Process Recording
          </button>
        </div>
      )}
    </section>
  );
}
