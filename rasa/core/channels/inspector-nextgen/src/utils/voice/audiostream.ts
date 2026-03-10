import { type Socket } from "socket.io-client";
import playbackProcessorUrl from "./playback-processor.ts?worker&url";
import microphoneProcessorUrl from "./microphone-processor.ts?worker&url";
import type { LogErrorFn } from "../../types";

// Buffer size for audio worklet processing
// 128 samples = 16ms @ 8kHz sample rate (optimal for low latency)
const bufferSize = 128;

// Sample rate optimized for voice (8kHz = telephone quality)
// Lower rate reduces bandwidth while maintaining intelligibility
const sampleRate = 8000;

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

const MAX_INT32_VALUE = 0x7fffffff;

const floatToIntArray = (arr: Float32Array): Int32Array => {
  return Int32Array.from(arr, (x) => x * MAX_INT32_VALUE);
};

const intToFloatArray = (arr: Int32Array): Float32Array => {
  return Float32Array.from(arr, (x) => x / MAX_INT32_VALUE);
};

interface Mark {
  id: string;
  bytesToGo: number;
}

export interface AudioQueue {
  buffer: Float32Array;
  marks: Array<Mark>;
  socket: Socket;
  write: (newAudio: Float32Array) => void;
  read: (nSamples: number) => Float32Array;
  length: () => number;
  addMarker: (id: string) => void;
  reduceMarkers: (bytesRead: number) => void;
  popMarkers: () => void;
  clear: () => void;
}

export const createAudioQueue = (socket: Socket): AudioQueue => {
  return {
    buffer: new Float32Array(0),
    marks: new Array<Mark>(),
    socket,

    write: function (newAudio: Float32Array) {
      const currentQLength = this.buffer.length;
      const newBuffer = new Float32Array(currentQLength + newAudio.length);
      newBuffer.set(this.buffer, 0);
      newBuffer.set(newAudio, currentQLength);
      this.buffer = newBuffer;
    },

    read: function (nSamples: number) {
      const samplesToPlay = this.buffer.subarray(0, nSamples);
      this.buffer = this.buffer.subarray(nSamples, this.buffer.length);
      this.reduceMarkers(samplesToPlay.length);
      this.popMarkers();
      return samplesToPlay;
    },

    length: function () {
      return this.buffer.length;
    },

    addMarker: function (id: string) {
      this.marks.push({ id, bytesToGo: this.length() });
    },

    reduceMarkers: function (bytesRead: number) {
      this.marks = this.marks.map((m) => {
        return { id: m.id, bytesToGo: m.bytesToGo - bytesRead };
      });
    },

    popMarkers: function () {
      // marks are ordered
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

    clear: function () {
      this.buffer = new Float32Array(0);
      this.marks = [];
    },
  };
};

interface MicrophoneStream {
  audioContext: AudioContext;
  audioStream: MediaStream;
  microphoneNode: AudioWorkletNode;
  source: MediaStreamAudioSourceNode;
}

/**
 * Streams microphone audio to server via WebSocket.
 *
 * @param socket - Connected Socket.IO socket instance
 * @returns MicrophoneStream object for cleanup, or undefined if permission denied
 * @throws Error if microphone is not available or readable
 */
export const streamMicrophoneToServer = async (
  socket: Socket,
  logError: LogErrorFn,
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
          audio: arrayBufferToBase64(floatToIntArray(audioData).buffer),
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
    // this error can be thrown if the microphone permission is denied
    // https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia#exceptions
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
 *
 * @param microphoneStream - MicrophoneStream object to stop, or undefined
 * @returns Promise that resolves when cleanup is complete
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
 * Sets up audio playback with worklet. Accepts optional existingQueue so the
 * caller can create and assign it earlier, allowing bot_message to buffer data
 * while the worklet loads.
 */
export const setupAudioPlayback = async (
  socket: Socket,
  logError: LogErrorFn,
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

  globalPlaybackNode.port.onmessage = (event: MessageEvent) => {
    if (event.data === "need-more-data") {
      const audioData = audioQueue.length()
        ? audioQueue.read(bufferSize)
        : new Float32Array(bufferSize);

      if (!(audioData instanceof Float32Array)) {
        logError("audioData is invalid, sending silence.", {
          tags: {
            component: "setupAudioPlayback",
            action: "globalPlaybackNode.port.onmessage",
          },
          extra: {
            data:
              typeof audioData === "object" ? JSON.stringify(audioData) : null,
          },
        });
        globalPlaybackNode?.port.postMessage(new Float32Array(bufferSize));
      } else if (audioData.length === bufferSize) {
        globalPlaybackNode?.port.postMessage(audioData);
      } else {
        const padded = new Float32Array(bufferSize);
        padded.set(audioData);
        globalPlaybackNode?.port.postMessage(padded);
      }
    }
  };

  globalPlaybackNode.connect(globalAudioOutputContext.destination);

  return audioQueue;
};

export const stopAudioPlayback = async (
  audioQueue: AudioQueue | undefined,
): Promise<void> => {
  if (!audioQueue) return;

  // Clear the audio buffer
  audioQueue.clear();

  if (globalPlaybackNode) {
    globalPlaybackNode.disconnect();
    globalPlaybackNode = undefined; // Clear reference
  }

  if (globalAudioOutputContext && globalAudioOutputContext.state !== "closed") {
    await globalAudioOutputContext.close();
    globalAudioOutputContext = undefined; // Clear reference
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
      const int32Data = new Int32Array(audioBytes);
      const audioData = intToFloatArray(int32Data);
      audioQueue.write(audioData);
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
