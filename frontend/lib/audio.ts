/**
 * Client-side audio processing utilities.
 *
 * Downsamples and converts audio to 16kHz mono WAV before upload,
 * minimizing payload size for speech-to-text backends.
 */

const TARGET_SAMPLE_RATE = 16000;
const TARGET_CHANNELS = 1;

/** Encode a Float32Array of PCM samples into a WAV Blob. */
function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  const bitsPerSample = 16;
  const byteRate = sampleRate * TARGET_CHANNELS * (bitsPerSample / 8);
  const blockAlign = TARGET_CHANNELS * (bitsPerSample / 8);
  const dataSize = samples.length * (bitsPerSample / 8);

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, TARGET_CHANNELS, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

/**
 * Downsample an uploaded audio file to 16kHz mono WAV.
 * Uses OfflineAudioContext for non-realtime decoding and resampling.
 * Returns unchanged blob if decoding fails (server-side fallback).
 */
export async function downsampleFile(file: File): Promise<File> {
  try {
    const arrayBuffer = await file.arrayBuffer();
    const audioCtx = new AudioContext();
    const decoded = await audioCtx.decodeAudioData(arrayBuffer);
    await audioCtx.close();

    const duration = decoded.duration;
    const offlineCtx = new OfflineAudioContext(
      TARGET_CHANNELS,
      Math.ceil(duration * TARGET_SAMPLE_RATE),
      TARGET_SAMPLE_RATE
    );

    const source = offlineCtx.createBufferSource();
    source.buffer = decoded;
    source.connect(offlineCtx.destination);
    source.start(0);

    const rendered = await offlineCtx.startRendering();
    const pcm = rendered.getChannelData(0);
    const wav = encodeWav(pcm, TARGET_SAMPLE_RATE);

    const baseName = file.name.replace(/\.[^.]+$/, "");
    return new File([wav], `${baseName}.wav`, { type: "audio/wav" });
  } catch {
    return file;
  }
}

/**
 * Convert a MediaRecorder Blob (WebM/Opus) to 16kHz mono WAV.
 * Falls back to the raw blob if decoding fails.
 */
export async function convertRecordingToWav(blob: Blob): Promise<File> {
  try {
    const arrayBuffer = await blob.arrayBuffer();
    const audioCtx = new AudioContext();
    const decoded = await audioCtx.decodeAudioData(arrayBuffer);
    await audioCtx.close();

    const duration = decoded.duration;
    const offlineCtx = new OfflineAudioContext(
      TARGET_CHANNELS,
      Math.ceil(duration * TARGET_SAMPLE_RATE),
      TARGET_SAMPLE_RATE
    );

    const source = offlineCtx.createBufferSource();
    source.buffer = decoded;
    source.connect(offlineCtx.destination);
    source.start(0);

    const rendered = await offlineCtx.startRendering();
    const pcm = rendered.getChannelData(0);
    const wav = encodeWav(pcm, TARGET_SAMPLE_RATE);

    return new File([wav], "recording.wav", { type: "audio/wav" });
  } catch {
    return new File([blob], "recording.webm", { type: blob.type });
  }
}
