import { type Socket } from "socket.io-client";
import playbackProcessorUrl from "./playback-processor.ts?worker&url";
import microphoneProcessorUrl from "./microphone-processor.ts?worker&url";
import type { LogErrorFn } from "../../types";

// Audio options for microphone
const audioOptions = {
  audio: {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
  },
};

const arrayBufferToBase64 = (buffer: ArrayBufferLike): string => {
  let binary = "";
  const bytes = new Uint8Array(buffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCodePoint(bytes[i]);
  }
  return globalThis.btoa(binary);
};

const base64ToArrayBuffer = (s: string): ArrayBuffer => {
  const binary_string = globalThis.atob(s);
  const len = binary_string.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binary_string.codePointAt(i) ?? 0;
  }
  return bytes.buffer;
};

const MAX_INT16_VALUE = 0x7fff;

const floatToInt16Array = (arr: Float32Array): Int16Array => {
  return Int16Array.from(arr, (x) => x * MAX_INT16_VALUE);
};

const int16ToFloatArray = (arr: Int16Array): Float32Array => {
  return Float32Array.from(arr, (x) => x / MAX_INT16_VALUE);
};

interface Mark {
  id: string;
  bytesToGo: number;
}

export interface AudioQueue {
  marks: Array<Mark>;
  queuedSamples: number;
  socket: Socket;
  enqueue: (newAudio: Float32Array) => void;
  onSamplesPlayed: (samplesPlayed: number) => void;
  attachPlaybackNode: (node: AudioWorkletNode) => void;
  addMarker: (id: string) => void;
  reduceMarkers: (samplesPlayed: number) => void;
  popMarkers: () => void;
  clear: () => void;
}

/**
 * Creates an AudioQueue that pushes audio directly to the playback worklet.
 *
 * Before a playback node is attached (via `attachPlaybackNode`), audio is
 * buffered internally so early bot messages are not lost.
 */
export const createAudioQueue = (socket: Socket): AudioQueue => {
  let playbackNode: AudioWorkletNode | undefined;
  const pendingChunks: Float32Array[] = [];

  const pushToWorklet = (audio: Float32Array) => {
    playbackNode!.port.postMessage(
      { type: "audio", data: audio },
      [audio.buffer],
    );
  };

  const queue: AudioQueue = {
    marks: new Array<Mark>(),
    queuedSamples: 0,
    socket,

    enqueue(newAudio: Float32Array) {
      this.queuedSamples += newAudio.length;
      if (playbackNode) {
        pushToWorklet(newAudio);
      } else {
        pendingChunks.push(newAudio);
      }
    },

    onSamplesPlayed(samplesPlayed: number) {
      if (samplesPlayed <= 0) return;
      this.queuedSamples = Math.max(0, this.queuedSamples - samplesPlayed);
      this.reduceMarkers(samplesPlayed);
      this.popMarkers();
    },

    attachPlaybackNode(node: AudioWorkletNode) {
      playbackNode = node;
      for (const chunk of pendingChunks) {
        pushToWorklet(chunk);
      }
      pendingChunks.length = 0;
    },

    addMarker(id: string) {
      this.marks.push({ id, bytesToGo: this.queuedSamples });
    },

    reduceMarkers(samplesPlayed: number) {
      this.marks = this.marks.map((m) => ({
        id: m.id,
        bytesToGo: m.bytesToGo - samplesPlayed,
      }));
    },

    popMarkers() {
      let popUpTo = 0;
      while (popUpTo < this.marks.length) {
        if (this.marks[popUpTo].bytesToGo <= 0) {
          popUpTo += 1;
        } else {
          break;
        }
      }
      const marksToPop = this.marks.slice(0, popUpTo);
      this.marks = this.marks.slice(popUpTo, this.marks.length);
      marksToPop.forEach((m) => {
        this.socket.emit("user_message", { marker: m.id });
      });
    },

    clear() {
      this.queuedSamples = 0;
      this.marks = [];
      if (playbackNode) {
        playbackNode.port.postMessage({ type: "clear" });
      } else {
        pendingChunks.length = 0;
      }
    },
  };

  return queue;
};

interface MicrophoneStream {
  audioContext: AudioContext;
  audioStream: MediaStream;
  microphoneNode: AudioWorkletNode;
  source: MediaStreamAudioSourceNode;
}

/**
 * Streams microphone audio to server via WebSocket.
 */
export const streamMicrophoneToServer = async (
  socket: Socket,
  logError: LogErrorFn,
  sampleRate: number,
): Promise<MicrophoneStream | undefined> => {
  const audioContext = new AudioContext({ sampleRate });

  try {
    const audioStream = await navigator.mediaDevices.getUserMedia(audioOptions);
    await audioContext.audioWorklet.addModule(microphoneProcessorUrl);

    const microphoneNode = new AudioWorkletNode(
      audioContext,
      "microphone-processor",
    );

    microphoneNode.port.onmessage = (event: MessageEvent) => {
      if (event.data instanceof Float32Array) {
        const audioData = event.data;
        socket.emit("user_message", {
          audio: arrayBufferToBase64(floatToInt16Array(audioData).buffer),
        });
      } else {
        logError(`Received unexpected data type from microphone-processor`, {
          tags: {
            component: "streamMicrophoneToServer",
            action: "microphoneNode.port.onmessage",
          },
          extra: {
            data:
              typeof event.data === "object"
                ? JSON.stringify(event.data)
                : null,
          },
        });
      }
    };
    const source = audioContext.createMediaStreamSource(audioStream);
    source.connect(microphoneNode);

    return {
      audioContext,
      audioStream,
      microphoneNode,
      source,
    };
  } catch (err) {
    if (audioContext.state !== "closed") {
      await audioContext.close();
    }
    logError(err, {
      tags: {
        component: "streamMicrophoneToServer",
        action: "Error streaming microphone to server",
      },
    });
    throw err;
  }
};

/**
 * Stops microphone audio streaming and cleans up resources.
 */
export const stopMicrophoneStream = async (
  microphoneStream: MicrophoneStream | undefined,
): Promise<void> => {
  if (!microphoneStream) return;

  const { audioStream, source, microphoneNode, audioContext } =
    microphoneStream;

  audioStream.getTracks().forEach((track) => track.stop());

  source.disconnect();
  microphoneNode.disconnect();

  if (audioContext.state !== "closed") {
    await audioContext.close();
  }
};

let globalAudioOutputContext: AudioContext | undefined;
let globalPlaybackNode: AudioWorkletNode | undefined;

/**
 * Sets up push-based audio playback with a worklet-side RingBuffer.
 *
 * Accepts an optional `existingQueue` so the caller can create and assign it
 * earlier, allowing bot_message events to buffer data while the worklet loads.
 */
export const setupAudioPlayback = async (
  socket: Socket,
  _logError: LogErrorFn,
  sampleRate: number,
  existingQueue?: AudioQueue,
): Promise<AudioQueue> => {
  const audioQueue = existingQueue ?? createAudioQueue(socket);

  if (globalAudioOutputContext && globalAudioOutputContext.state !== "closed") {
    await globalAudioOutputContext.close();
  }
  if (globalPlaybackNode) {
    globalPlaybackNode.disconnect();
  }

  globalAudioOutputContext = new AudioContext({ sampleRate });

  if (globalAudioOutputContext.state === "suspended") {
    await globalAudioOutputContext.resume();
  }

  await globalAudioOutputContext.audioWorklet.addModule(playbackProcessorUrl);

  globalPlaybackNode = new AudioWorkletNode(
    globalAudioOutputContext,
    "playback-processor",
  );

  audioQueue.attachPlaybackNode(globalPlaybackNode);

  globalPlaybackNode.port.onmessage = (
    event: MessageEvent<{ type: string; samples: number }>,
  ) => {
    if (event.data.type === "played-samples") {
      audioQueue.onSamplesPlayed(event.data.samples);
    }
  };

  globalPlaybackNode.connect(globalAudioOutputContext.destination);

  return audioQueue;
};

export const stopAudioPlayback = async (
  audioQueue: AudioQueue | undefined,
): Promise<void> => {
  if (!audioQueue) return;

  audioQueue.clear();

  if (globalPlaybackNode) {
    globalPlaybackNode.disconnect();
    globalPlaybackNode = undefined;
  }

  if (globalAudioOutputContext && globalAudioOutputContext.state !== "closed") {
    await globalAudioOutputContext.close();
    globalAudioOutputContext = undefined;
  }
};

export const addDataToAudioQueue =
  (audioQueue: AudioQueue) => (message: string) => {
    const parsedData = JSON.parse(message) as Record<string, unknown>;

    if (typeof parsedData !== "object" || parsedData === null) {
      throw new Error("Invalid message format received from server");
    }

    if (parsedData["error"]) {
      throw new Error(
        typeof parsedData["error"] === "string"
          ? parsedData["error"]
          : JSON.stringify(parsedData["error"]),
      );
    }

    if (parsedData["audio"] && typeof parsedData["audio"] === "string") {
      const audioBytes = base64ToArrayBuffer(parsedData["audio"]);
      const int16Data = new Int16Array(audioBytes);
      const audioData = int16ToFloatArray(int16Data);
      audioQueue.enqueue(audioData);
    } else if (
      parsedData["marker"] &&
      typeof parsedData["marker"] === "string"
    ) {
      audioQueue.addMarker(parsedData["marker"]);
    } else if (parsedData["interruptPlayback"]) {
      audioQueue.clear();
    } else {
      throw new Error("Unknown data structure received from server");
    }
  };
