import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  setupAudioPlayback,
  stopAudioPlayback,
  streamMicrophoneToServer,
  stopMicrophoneStream,
  addDataToAudioQueue,
  type AudioQueue,
} from "./audiostream";
import { type Socket } from "socket.io-client";

const mockLogError = vi.fn();

// Mock worker URLs
vi.mock("./playback-processor.ts?worker&url", () => ({
  default: "mock-playback-processor-url",
}));

vi.mock("./microphone-processor.ts?worker&url", () => ({
  default: "mock-microphone-processor-url",
}));

// Create mock socket
const createMockSocket = (): Socket => {
  return {
    emit: vi.fn(),
    on: vi.fn(),
    off: vi.fn(),
  } as unknown as Socket;
};

// Mock Web Audio API
class MockAudioContext {
  state: AudioContextState = "running";
  sampleRate = 8000;
  destination = {};
  audioWorklet = {
    addModule: vi.fn().mockResolvedValue(undefined),
  };

  resume = vi.fn().mockResolvedValue(undefined);
  close = vi.fn().mockResolvedValue(undefined);
  createMediaStreamSource = vi.fn().mockReturnValue({
    connect: vi.fn(),
    disconnect: vi.fn(),
  });
}

class MockAudioWorkletNode {
  port = {
    onmessage: null as ((event: MessageEvent) => void) | null,
    postMessage: vi.fn(),
  };

  connect = vi.fn();
  disconnect = vi.fn();
}

class MockMediaStream {
  getTracks = vi.fn().mockReturnValue([{ stop: vi.fn() }, { stop: vi.fn() }]);
}

// Setup global mocks
beforeEach(() => {
  globalThis.AudioContext = MockAudioContext as never;
  globalThis.AudioWorkletNode = MockAudioWorkletNode as never;
  globalThis.navigator = {
    mediaDevices: {
      getUserMedia: vi.fn(),
    },
  } as never;
});

describe("audiostream", () => {
  describe("utility functions", () => {
    it("converts ArrayBuffer to base64 and back", () => {
      // Create test data
      const originalData = new Uint8Array([1, 2, 3, 4, 5]);
      const arrayBuffer = originalData.buffer;

      // Convert to base64
      let binary = "";
      const bytes = new Uint8Array(arrayBuffer);
      for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCodePoint(bytes[i]);
      }
      const base64 = globalThis.btoa(binary);

      // Convert back to ArrayBuffer
      const binaryString = globalThis.atob(base64);
      const len = binaryString.length;
      const resultBytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        resultBytes[i] = binaryString.codePointAt(i) ?? 0;
      }

      expect(resultBytes).toEqual(originalData);
    });

    it("converts Float32Array to Int32Array using MAX_INT32_VALUE", () => {
      const MAX_INT32_VALUE = 0x7fffffff;
      const floatArray = new Float32Array([0.5, -0.5, 0, 1, -1]);
      const intArray = Int32Array.from(floatArray, (x) => x * MAX_INT32_VALUE);

      // Note: Int32Array automatically truncates decimal values
      expect(intArray[0]).toBe(Math.trunc(0.5 * MAX_INT32_VALUE));
      expect(intArray[1]).toBe(Math.trunc(-0.5 * MAX_INT32_VALUE));
      expect(intArray[2]).toBe(0);
      expect(intArray[3]).toBe(MAX_INT32_VALUE);
      expect(intArray[4]).toBe(-MAX_INT32_VALUE);
    });

    it("converts Int32Array to Float32Array using MAX_INT32_VALUE", () => {
      const MAX_INT32_VALUE = 0x7fffffff;
      const intArray = new Int32Array([MAX_INT32_VALUE, -MAX_INT32_VALUE, 0]);
      const floatArray = Float32Array.from(
        intArray,
        (x) => x / MAX_INT32_VALUE,
      );

      expect(floatArray[0]).toBeCloseTo(1, 5);
      expect(floatArray[1]).toBeCloseTo(-1, 5);
      expect(floatArray[2]).toBe(0);
    });
  });

  describe("setupAudioPlayback", () => {
    let mockSocket: Socket;

    beforeEach(() => {
      vi.clearAllMocks();
      mockSocket = createMockSocket();
    });

    it("creates AudioQueue with empty buffer and marks", async () => {
      const audioQueue = await setupAudioPlayback(mockSocket, mockLogError);

      expect(audioQueue.buffer).toBeInstanceOf(Float32Array);
      expect(audioQueue.buffer.length).toBe(0);
      expect(audioQueue.marks).toEqual([]);
      expect(audioQueue.socket).toBe(mockSocket);
    });

    it("creates AudioContext with correct sample rate", async () => {
      const mockConstructor = vi.fn(() => new MockAudioContext());
      globalThis.AudioContext = mockConstructor as never;

      await setupAudioPlayback(mockSocket, mockLogError);

      expect(mockConstructor).toHaveBeenCalledWith({ sampleRate: 8000 });
    });

    it("adds audio worklet module", async () => {
      const mockContext = new MockAudioContext();
      globalThis.AudioContext = vi.fn(() => mockContext) as never;

      await setupAudioPlayback(mockSocket, mockLogError);

      expect(mockContext.audioWorklet.addModule).toHaveBeenCalledWith(
        "mock-playback-processor-url",
      );
    });

    it("resumes suspended audio context", async () => {
      const suspendedContext = new MockAudioContext();
      suspendedContext.state = "suspended";
      globalThis.AudioContext = vi
        .fn()
        .mockReturnValue(suspendedContext) as never;

      await setupAudioPlayback(mockSocket, mockLogError);

      expect(suspendedContext.resume).toHaveBeenCalledTimes(1);
    });

    it("does not resume running audio context", async () => {
      const runningContext = new MockAudioContext();
      runningContext.state = "running";
      globalThis.AudioContext = vi
        .fn()
        .mockReturnValue(runningContext) as never;

      await setupAudioPlayback(mockSocket, mockLogError);

      expect(runningContext.resume).not.toHaveBeenCalled();
    });
  });

  describe("stopAudioPlayback", () => {
    let mockSocket: Socket;
    let audioQueue: AudioQueue;

    beforeEach(async () => {
      vi.clearAllMocks();
      mockSocket = createMockSocket();
      audioQueue = await setupAudioPlayback(mockSocket, mockLogError);
    });

    it("clears audio queue buffer and marks", async () => {
      audioQueue.write(new Float32Array([1, 2, 3, 4, 5]));
      audioQueue.addMarker("marker1");
      audioQueue.addMarker("marker2");
      expect(audioQueue.length()).toBe(5);
      expect(audioQueue.marks.length).toBe(2);

      await stopAudioPlayback(audioQueue);

      expect(audioQueue.length()).toBe(0);
      expect(audioQueue.marks.length).toBe(0);
    });

    it("handles undefined audioQueue gracefully", async () => {
      await expect(stopAudioPlayback(undefined)).resolves.toBeUndefined();
    });
  });

  describe("streamMicrophoneToServer", () => {
    let mockSocket: Socket;

    beforeEach(() => {
      vi.clearAllMocks();
      mockSocket = createMockSocket();
    });

    /* eslint-disable @typescript-eslint/unbound-method */

    it("creates AudioContext with correct sample rate", async () => {
      const mockConstructor = vi.fn(() => new MockAudioContext());
      globalThis.AudioContext = mockConstructor as never;

      const mockStream = new MockMediaStream();
      vi.mocked(navigator.mediaDevices.getUserMedia).mockResolvedValue(
        mockStream as never,
      );

      await streamMicrophoneToServer(mockSocket, mockLogError);

      expect(mockConstructor).toHaveBeenCalledWith({ sampleRate: 8000 });
    });

    it("requests microphone access with correct audio options", async () => {
      const mockStream = new MockMediaStream();
      vi.mocked(navigator.mediaDevices.getUserMedia).mockResolvedValue(
        mockStream as never,
      );

      await streamMicrophoneToServer(mockSocket, mockLogError);

      expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledWith({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
    });

    it("adds microphone worklet module", async () => {
      const mockContext = new MockAudioContext();
      globalThis.AudioContext = vi.fn(() => mockContext) as never;

      const mockStream = new MockMediaStream();
      vi.mocked(navigator.mediaDevices.getUserMedia).mockResolvedValue(
        mockStream as never,
      );

      await streamMicrophoneToServer(mockSocket, mockLogError);

      expect(mockContext.audioWorklet.addModule).toHaveBeenCalledWith(
        "mock-microphone-processor-url",
      );
    });

    it("throws error when microphone access is denied", async () => {
      const error = new Error("Permission denied");
      vi.mocked(navigator.mediaDevices.getUserMedia).mockRejectedValue(error);

      await expect(streamMicrophoneToServer(mockSocket, mockLogError)).rejects.toThrow(
        "Permission denied",
      );
    });

    it("closes AudioContext when getUserMedia rejects to prevent leak", async () => {
      const mockContext = new MockAudioContext();
      globalThis.AudioContext = vi.fn(() => mockContext) as never;

      const error = new Error("Permission denied");
      vi.mocked(navigator.mediaDevices.getUserMedia).mockRejectedValue(error);

      await expect(streamMicrophoneToServer(mockSocket, mockLogError)).rejects.toThrow(
        "Permission denied",
      );

      expect(mockContext.close).toHaveBeenCalledTimes(1);
    });

    it("closes AudioContext when addModule rejects to prevent leak", async () => {
      const mockContext = new MockAudioContext();
      globalThis.AudioContext = vi.fn(() => mockContext) as never;

      const mockStream = new MockMediaStream();
      vi.mocked(navigator.mediaDevices.getUserMedia).mockResolvedValue(
        mockStream as never,
      );
      vi.mocked(mockContext.audioWorklet.addModule).mockRejectedValue(
        new Error("Failed to load worklet"),
      );

      await expect(streamMicrophoneToServer(mockSocket, mockLogError)).rejects.toThrow(
        "Failed to load worklet",
      );

      expect(mockContext.close).toHaveBeenCalledTimes(1);
    });

    it("returns MicrophoneStream object on success", async () => {
      const mockStream = new MockMediaStream();
      vi.mocked(navigator.mediaDevices.getUserMedia).mockResolvedValue(
        mockStream as never,
      );

      const result = await streamMicrophoneToServer(mockSocket, mockLogError);

      expect(result).toBeDefined();
      expect(result?.audioContext).toBeInstanceOf(MockAudioContext);
      expect(result?.audioStream).toBe(mockStream);
      expect(result?.microphoneNode).toBeInstanceOf(MockAudioWorkletNode);
    });

    /* eslint-enable @typescript-eslint/unbound-method */
  });

  describe("stopMicrophoneStream", () => {
    /* eslint-disable @typescript-eslint/unbound-method, @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access */

    it("handles undefined microphoneStream gracefully", async () => {
      await expect(stopMicrophoneStream(undefined)).resolves.toBeUndefined();
    });

    it("stops all media tracks", async () => {
      const mockSocket = createMockSocket();
      const mockStream = new MockMediaStream();
      vi.mocked(navigator.mediaDevices.getUserMedia).mockResolvedValue(
        mockStream as never,
      );

      const microphoneStream = await streamMicrophoneToServer(mockSocket, mockLogError);
      await stopMicrophoneStream(microphoneStream);

      const tracks = mockStream.getTracks();
      tracks.forEach((track: { stop: () => void }) => {
        expect(track.stop).toHaveBeenCalledTimes(1);
      });
    });

    it("closes audio context if not already closed", async () => {
      const mockSocket = createMockSocket();
      const mockStream = new MockMediaStream();
      vi.mocked(navigator.mediaDevices.getUserMedia).mockResolvedValue(
        mockStream as never,
      );

      const microphoneStream = await streamMicrophoneToServer(mockSocket, mockLogError);
      const audioContext =
        microphoneStream?.audioContext as unknown as MockAudioContext;

      await stopMicrophoneStream(microphoneStream);

      expect(audioContext.close).toHaveBeenCalledTimes(1);
    });

    it("does not close already closed audio context", async () => {
      const mockSocket = createMockSocket();
      const mockStream = new MockMediaStream();
      vi.mocked(navigator.mediaDevices.getUserMedia).mockResolvedValue(
        mockStream as never,
      );

      const microphoneStream = await streamMicrophoneToServer(mockSocket, mockLogError);
      const audioContext =
        microphoneStream?.audioContext as unknown as MockAudioContext;
      audioContext.state = "closed";

      await stopMicrophoneStream(microphoneStream);

      expect(audioContext.close).not.toHaveBeenCalled();
    });

    /* eslint-enable @typescript-eslint/unbound-method, @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access */
  });

  describe("AudioQueue", () => {
    let mockSocket: Socket;
    let audioQueue: AudioQueue;

    beforeEach(async () => {
      vi.clearAllMocks();
      mockSocket = createMockSocket();
      audioQueue = await setupAudioPlayback(mockSocket, mockLogError);
    });

    /* eslint-disable @typescript-eslint/unbound-method */

    describe("write", () => {
      it("appends new audio to empty buffer", () => {
        const newAudio = new Float32Array([1, 2, 3]);
        audioQueue.write(newAudio);

        expect(audioQueue.length()).toBe(3);
        expect(audioQueue.buffer).toEqual(new Float32Array([1, 2, 3]));
      });

      it("appends new audio to existing buffer", () => {
        audioQueue.write(new Float32Array([1, 2, 3]));
        audioQueue.write(new Float32Array([4, 5]));

        expect(audioQueue.length()).toBe(5);
        expect(audioQueue.buffer).toEqual(new Float32Array([1, 2, 3, 4, 5]));
      });

      it("handles empty audio array", () => {
        audioQueue.write(new Float32Array([1, 2]));
        audioQueue.write(new Float32Array([]));

        expect(audioQueue.length()).toBe(2);
        expect(audioQueue.buffer).toEqual(new Float32Array([1, 2]));
      });
    });

    describe("read", () => {
      it("reads requested number of samples from buffer", () => {
        audioQueue.write(new Float32Array([1, 2, 3, 4, 5]));

        const samples = audioQueue.read(3);

        expect(samples).toEqual(new Float32Array([1, 2, 3]));
        expect(audioQueue.length()).toBe(2);
        expect(audioQueue.buffer).toEqual(new Float32Array([4, 5]));
      });

      it("reads all samples when requested more than available", () => {
        audioQueue.write(new Float32Array([1, 2, 3]));

        const samples = audioQueue.read(10);

        expect(samples).toEqual(new Float32Array([1, 2, 3]));
        expect(audioQueue.length()).toBe(0);
      });

      it("returns empty array when buffer is empty", () => {
        const samples = audioQueue.read(5);

        expect(samples).toEqual(new Float32Array([]));
        expect(audioQueue.length()).toBe(0);
      });
    });

    describe("length", () => {
      it("returns 0 for empty buffer", () => {
        expect(audioQueue.length()).toBe(0);
      });

      it("returns correct length after write", () => {
        audioQueue.write(new Float32Array([1, 2, 3, 4, 5]));

        expect(audioQueue.length()).toBe(5);
      });

      it("returns updated length after read", () => {
        audioQueue.write(new Float32Array([1, 2, 3, 4, 5]));
        audioQueue.read(2);

        expect(audioQueue.length()).toBe(3);
      });
    });

    describe("markers", () => {
      describe("addMarker", () => {
        it("adds marker with current buffer length as bytesToGo", () => {
          audioQueue.write(new Float32Array([1, 2, 3, 4, 5]));
          audioQueue.addMarker("marker1");

          expect(audioQueue.marks).toHaveLength(1);
          expect(audioQueue.marks[0]).toEqual({ id: "marker1", bytesToGo: 5 });
        });

        it("adds multiple markers with correct bytesToGo", () => {
          audioQueue.write(new Float32Array([1, 2, 3]));
          audioQueue.addMarker("marker1");

          audioQueue.write(new Float32Array([4, 5]));
          audioQueue.addMarker("marker2");

          expect(audioQueue.marks).toHaveLength(2);
          expect(audioQueue.marks[0]).toEqual({ id: "marker1", bytesToGo: 3 });
          expect(audioQueue.marks[1]).toEqual({ id: "marker2", bytesToGo: 5 });
        });

        it("adds marker with 0 bytesToGo when buffer is empty", () => {
          audioQueue.addMarker("marker1");

          expect(audioQueue.marks[0]).toEqual({ id: "marker1", bytesToGo: 0 });
        });
      });

      describe("reduceMarkers", () => {
        it("reduces bytesToGo for all markers", () => {
          audioQueue.write(new Float32Array([1, 2, 3, 4, 5]));
          audioQueue.addMarker("marker1");
          audioQueue.addMarker("marker2");

          audioQueue.reduceMarkers(2);

          expect(audioQueue.marks[0].bytesToGo).toBe(3);
          expect(audioQueue.marks[1].bytesToGo).toBe(3);
        });

        it("allows bytesToGo to become negative", () => {
          audioQueue.write(new Float32Array([1, 2]));
          audioQueue.addMarker("marker1");

          audioQueue.reduceMarkers(5);

          expect(audioQueue.marks[0].bytesToGo).toBe(-3);
        });
      });

      describe("popMarkers", () => {
        it("emits markers with bytesToGo <= 0 to socket", () => {
          audioQueue.addMarker("marker1");
          audioQueue.addMarker("marker2");

          audioQueue.popMarkers();

          expect(mockSocket.emit).toHaveBeenCalledTimes(2);
          expect(mockSocket.emit).toHaveBeenCalledWith("user_message", {
            marker: "marker1",
          });
          expect(mockSocket.emit).toHaveBeenCalledWith("user_message", {
            marker: "marker2",
          });
        });

        it("removes emitted markers from marks array", () => {
          audioQueue.addMarker("marker1");
          audioQueue.addMarker("marker2");

          audioQueue.popMarkers();

          expect(audioQueue.marks).toHaveLength(0);
        });

        it("keeps markers with positive bytesToGo", () => {
          audioQueue.write(new Float32Array([1, 2, 3, 4, 5]));
          audioQueue.addMarker("marker1");
          audioQueue.addMarker("marker2");

          audioQueue.reduceMarkers(2);
          audioQueue.popMarkers();

          expect(audioQueue.marks).toHaveLength(2);
          expect(mockSocket.emit).not.toHaveBeenCalled();
        });

        it("pops only markers with bytesToGo <= 0", () => {
          audioQueue.write(new Float32Array([1, 2, 3, 4, 5]));
          audioQueue.addMarker("marker1");
          audioQueue.write(new Float32Array([6, 7, 8]));
          audioQueue.addMarker("marker2");

          audioQueue.reduceMarkers(6);
          audioQueue.popMarkers();

          expect(audioQueue.marks).toHaveLength(1);
          expect(audioQueue.marks[0].id).toBe("marker2");
          expect(mockSocket.emit).toHaveBeenCalledTimes(1);
          expect(mockSocket.emit).toHaveBeenCalledWith("user_message", {
            marker: "marker1",
          });
        });
      });

      describe("read with markers", () => {
        it("reduces markers and pops them when reading", () => {
          audioQueue.write(new Float32Array([1, 2, 3, 4, 5]));
          audioQueue.addMarker("marker1");

          audioQueue.read(5);

          expect(mockSocket.emit).toHaveBeenCalledWith("user_message", {
            marker: "marker1",
          });
          expect(audioQueue.marks).toHaveLength(0);
        });
      });
    });

    describe("clear", () => {
      it("clears buffer and marks", () => {
        audioQueue.write(new Float32Array([1, 2, 3, 4, 5]));
        audioQueue.addMarker("marker1");
        audioQueue.addMarker("marker2");

        audioQueue.clear();

        expect(audioQueue.length()).toBe(0);
        expect(audioQueue.marks).toHaveLength(0);
      });

      it("handles already empty buffer and marks", () => {
        audioQueue.clear();

        expect(audioQueue.length()).toBe(0);
        expect(audioQueue.marks).toHaveLength(0);
      });
    });

    /* eslint-enable @typescript-eslint/unbound-method */
  });

  describe("addDataToAudioQueue", () => {
    let mockSocket: Socket;
    let audioQueue: AudioQueue;
    let addData: ReturnType<typeof addDataToAudioQueue>;

    beforeEach(async () => {
      vi.clearAllMocks();
      mockSocket = createMockSocket();
      audioQueue = await setupAudioPlayback(mockSocket, mockLogError);
      addData = addDataToAudioQueue(audioQueue);
    });

    describe("audio data", () => {
      it("adds audio data to queue", () => {
        const audioData = new Float32Array([0.5, -0.5, 0.25]);
        const MAX_INT32_VALUE = 0x7fffffff;
        const intArray = Int32Array.from(audioData, (x) => x * MAX_INT32_VALUE);
        const base64Audio = Buffer.from(intArray.buffer).toString("base64");

        addData(JSON.stringify({ audio: base64Audio }));

        expect(audioQueue.length()).toBeGreaterThan(0);
      });

      it("throws on invalid base64 audio data", () => {
        expect(() =>
          addData(JSON.stringify({ audio: "invalid-base64", extra: "data" })),
        ).toThrow();
      });
    });

    describe("marker data", () => {
      it("adds marker to queue", () => {
        addData(JSON.stringify({ marker: "marker1" }));

        expect(audioQueue.marks).toHaveLength(1);
        expect(audioQueue.marks[0].id).toBe("marker1");
      });

      it("adds multiple markers in order", () => {
        addData(JSON.stringify({ marker: "marker1" }));
        addData(JSON.stringify({ marker: "marker2" }));
        addData(JSON.stringify({ marker: "marker3" }));

        expect(audioQueue.marks).toHaveLength(3);
        expect(audioQueue.marks[0].id).toBe("marker1");
        expect(audioQueue.marks[1].id).toBe("marker2");
        expect(audioQueue.marks[2].id).toBe("marker3");
      });
    });

    describe("interruptPlayback", () => {
      it("clears audio queue on interrupt", () => {
        audioQueue.write(new Float32Array([1, 2, 3, 4, 5]));
        audioQueue.addMarker("marker1");

        addData(JSON.stringify({ interruptPlayback: true }));

        expect(audioQueue.length()).toBe(0);
        expect(audioQueue.marks).toHaveLength(0);
      });
    });

    describe("error handling", () => {
      it("throws on invalid JSON", () => {
        expect(() => addData("not valid json{")).toThrow();
      });

      it("throws on non-object parsed data", () => {
        expect(() => addData(JSON.stringify("string value"))).toThrow(
          "Invalid message format",
        );
      });

      it("throws on null parsed data", () => {
        expect(() => addData(JSON.stringify(null))).toThrow(
          "Invalid message format",
        );
      });

      it("throws on error property in data", () => {
        expect(() =>
          addData(JSON.stringify({ error: "Something went wrong" })),
        ).toThrow("Something went wrong");
      });

      it("throws on unknown data structure", () => {
        expect(() => addData(JSON.stringify({ unknown: "property" }))).toThrow(
          "Unknown data structure",
        );
      });

      it("throws on empty object", () => {
        expect(() => addData(JSON.stringify({}))).toThrow(
          "Unknown data structure",
        );
      });
    });
  });
});
