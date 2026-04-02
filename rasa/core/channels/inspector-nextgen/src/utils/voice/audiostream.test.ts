import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  setupAudioPlayback,
  stopAudioPlayback,
  streamMicrophoneToServer,
  stopMicrophoneStream,
  addDataToAudioQueue,
  createAudioQueue,
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
  sampleRate = 48000;
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
  static instances: MockAudioWorkletNode[] = [];

  port = {
    onmessage: null as ((event: MessageEvent) => void) | null,
    postMessage: vi.fn(),
  };

  connect = vi.fn();
  disconnect = vi.fn();

  constructor() {
    MockAudioWorkletNode.instances.push(this);
  }
}

class MockMediaStream {
  getTracks = vi.fn().mockReturnValue([{ stop: vi.fn() }, { stop: vi.fn() }]);
}

// Setup global mocks
beforeEach(() => {
  MockAudioWorkletNode.instances = [];
  globalThis.AudioContext = MockAudioContext as never;
  globalThis.AudioWorkletNode = MockAudioWorkletNode as never;
  globalThis.navigator = {
    mediaDevices: {
      getUserMedia: vi.fn(),
    },
  } as never;
});

/** Helper: get the playback worklet node created by setupAudioPlayback. */
const getPlaybackNode = (): MockAudioWorkletNode => {
  const node = MockAudioWorkletNode.instances[
    MockAudioWorkletNode.instances.length - 1
  ];
  expect(node).toBeDefined();
  return node;
};

describe("audiostream", () => {
  describe("utility functions", () => {
    it("converts ArrayBuffer to base64 and back", () => {
      const originalData = new Uint8Array([1, 2, 3, 4, 5]);
      const arrayBuffer = originalData.buffer;

      let binary = "";
      const bytes = new Uint8Array(arrayBuffer);
      for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCodePoint(bytes[i]);
      }
      const base64 = globalThis.btoa(binary);

      const binaryString = globalThis.atob(base64);
      const len = binaryString.length;
      const resultBytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        resultBytes[i] = binaryString.codePointAt(i) ?? 0;
      }

      expect(resultBytes).toEqual(originalData);
    });

    it("converts Float32Array to Int16Array using MAX_INT16_VALUE", () => {
      const MAX_INT16_VALUE = 0x7fff;
      const floatArray = new Float32Array([0.5, -0.5, 0, 1, -1]);
      const intArray = Int16Array.from(floatArray, (x) => x * MAX_INT16_VALUE);

      expect(intArray[0]).toBe(Math.trunc(0.5 * MAX_INT16_VALUE));
      expect(intArray[1]).toBe(Math.trunc(-0.5 * MAX_INT16_VALUE));
      expect(intArray[2]).toBe(0);
      expect(intArray[3]).toBe(MAX_INT16_VALUE);
      expect(intArray[4]).toBe(-MAX_INT16_VALUE);
    });

    it("converts Int16Array to Float32Array using MAX_INT16_VALUE", () => {
      const MAX_INT16_VALUE = 0x7fff;
      const intArray = new Int16Array([MAX_INT16_VALUE, -MAX_INT16_VALUE, 0]);
      const floatArray = Float32Array.from(
        intArray,
        (x) => x / MAX_INT16_VALUE,
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

    it("creates AudioQueue with zero queuedSamples and empty marks", async () => {
      const audioQueue = await setupAudioPlayback(
        mockSocket,
        mockLogError,
        48000,
      );

      expect(audioQueue.queuedSamples).toBe(0);
      expect(audioQueue.marks).toEqual([]);
      expect(audioQueue.socket).toBe(mockSocket);
    });

    it("creates AudioContext with correct sample rate", async () => {
      const mockConstructor = vi.fn(() => new MockAudioContext());
      globalThis.AudioContext = mockConstructor as never;

      await setupAudioPlayback(mockSocket, mockLogError, 48000);

      expect(mockConstructor).toHaveBeenCalledWith({ sampleRate: 48000 });
    });

    it("adds audio worklet module", async () => {
      const mockContext = new MockAudioContext();
      globalThis.AudioContext = vi.fn(() => mockContext) as never;

      await setupAudioPlayback(mockSocket, mockLogError, 48000);

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

      await setupAudioPlayback(mockSocket, mockLogError, 48000);

      expect(suspendedContext.resume).toHaveBeenCalledTimes(1);
    });

    it("does not resume running audio context", async () => {
      const runningContext = new MockAudioContext();
      runningContext.state = "running";
      globalThis.AudioContext = vi
        .fn()
        .mockReturnValue(runningContext) as never;

      await setupAudioPlayback(mockSocket, mockLogError, 48000);

      expect(runningContext.resume).not.toHaveBeenCalled();
    });

    it("flushes pending audio when worklet attaches", async () => {
      const queue = createAudioQueue(mockSocket);
      queue.enqueue(new Float32Array([1, 2, 3]));
      queue.enqueue(new Float32Array([4, 5]));

      await setupAudioPlayback(mockSocket, mockLogError, 48000, queue);

      const node = getPlaybackNode();
      expect(node.port.postMessage).toHaveBeenCalledTimes(2);
      /* eslint-disable @typescript-eslint/no-unsafe-assignment */
      expect(node.port.postMessage).toHaveBeenCalledWith(
        { type: "audio", data: expect.any(Float32Array) },
        expect.any(Array),
      );
      /* eslint-enable @typescript-eslint/no-unsafe-assignment */
      expect(queue.queuedSamples).toBe(5);
    });
  });

  describe("stopAudioPlayback", () => {
    let mockSocket: Socket;
    let audioQueue: AudioQueue;

    beforeEach(async () => {
      vi.clearAllMocks();
      mockSocket = createMockSocket();
      audioQueue = await setupAudioPlayback(mockSocket, mockLogError, 48000);
    });

    it("clears audio queue queuedSamples and marks", async () => {
      audioQueue.enqueue(new Float32Array([1, 2, 3, 4, 5]));
      audioQueue.addMarker("marker1");
      audioQueue.addMarker("marker2");
      expect(audioQueue.queuedSamples).toBe(5);
      expect(audioQueue.marks.length).toBe(2);

      await stopAudioPlayback(audioQueue);

      expect(audioQueue.queuedSamples).toBe(0);
      expect(audioQueue.marks.length).toBe(0);
    });

    it("sends clear message to worklet", async () => {
      const node = getPlaybackNode();
      node.port.postMessage.mockClear();

      await stopAudioPlayback(audioQueue);

      expect(node.port.postMessage).toHaveBeenCalledWith({ type: "clear" });
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

      await streamMicrophoneToServer(mockSocket, mockLogError, 48000);

      expect(mockConstructor).toHaveBeenCalledWith({ sampleRate: 48000 });
    });

    it("requests microphone access with correct audio options", async () => {
      const mockStream = new MockMediaStream();
      vi.mocked(navigator.mediaDevices.getUserMedia).mockResolvedValue(
        mockStream as never,
      );

      await streamMicrophoneToServer(mockSocket, mockLogError, 48000);

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

      await streamMicrophoneToServer(mockSocket, mockLogError, 48000);

      expect(mockContext.audioWorklet.addModule).toHaveBeenCalledWith(
        "mock-microphone-processor-url",
      );
    });

    it("throws error when microphone access is denied", async () => {
      const error = new Error("Permission denied");
      vi.mocked(navigator.mediaDevices.getUserMedia).mockRejectedValue(error);

      await expect(
        streamMicrophoneToServer(mockSocket, mockLogError, 48000),
      ).rejects.toThrow("Permission denied");
    });

    it("closes AudioContext when getUserMedia rejects to prevent leak", async () => {
      const mockContext = new MockAudioContext();
      globalThis.AudioContext = vi.fn(() => mockContext) as never;

      const error = new Error("Permission denied");
      vi.mocked(navigator.mediaDevices.getUserMedia).mockRejectedValue(error);

      await expect(
        streamMicrophoneToServer(mockSocket, mockLogError, 48000),
      ).rejects.toThrow("Permission denied");

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

      await expect(
        streamMicrophoneToServer(mockSocket, mockLogError, 48000),
      ).rejects.toThrow("Failed to load worklet");

      expect(mockContext.close).toHaveBeenCalledTimes(1);
    });

    it("returns MicrophoneStream object on success", async () => {
      const mockStream = new MockMediaStream();
      vi.mocked(navigator.mediaDevices.getUserMedia).mockResolvedValue(
        mockStream as never,
      );

      const result = await streamMicrophoneToServer(
        mockSocket,
        mockLogError,
        48000,
      );

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

      const microphoneStream = await streamMicrophoneToServer(
        mockSocket,
        mockLogError,
        48000,
      );
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

      const microphoneStream = await streamMicrophoneToServer(
        mockSocket,
        mockLogError,
        48000,
      );
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

      const microphoneStream = await streamMicrophoneToServer(
        mockSocket,
        mockLogError,
        48000,
      );
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
      MockAudioWorkletNode.instances = [];
      mockSocket = createMockSocket();
      audioQueue = await setupAudioPlayback(mockSocket, mockLogError, 48000);
    });

    /* eslint-disable @typescript-eslint/unbound-method */

    describe("enqueue", () => {
      it("pushes audio to worklet and updates queuedSamples", () => {
        const node = getPlaybackNode();
        node.port.postMessage.mockClear();

        audioQueue.enqueue(new Float32Array([1, 2, 3]));

        expect(audioQueue.queuedSamples).toBe(3);
        /* eslint-disable @typescript-eslint/no-unsafe-assignment */
        expect(node.port.postMessage).toHaveBeenCalledWith(
          { type: "audio", data: expect.any(Float32Array) },
          expect.any(Array),
        );
        /* eslint-enable @typescript-eslint/no-unsafe-assignment */
      });

      it("accumulates queuedSamples across multiple enqueues", () => {
        audioQueue.enqueue(new Float32Array([1, 2, 3]));
        audioQueue.enqueue(new Float32Array([4, 5]));

        expect(audioQueue.queuedSamples).toBe(5);
      });

      it("handles empty audio array", () => {
        audioQueue.enqueue(new Float32Array([1, 2]));
        audioQueue.enqueue(new Float32Array([]));

        expect(audioQueue.queuedSamples).toBe(2);
      });

      it("buffers audio before playback node is attached", () => {
        const earlyQueue = createAudioQueue(mockSocket);
        earlyQueue.enqueue(new Float32Array([1, 2, 3]));
        earlyQueue.enqueue(new Float32Array([4, 5]));

        expect(earlyQueue.queuedSamples).toBe(5);
      });
    });

    describe("onSamplesPlayed", () => {
      it("decrements queuedSamples", () => {
        audioQueue.enqueue(new Float32Array([1, 2, 3, 4, 5]));

        audioQueue.onSamplesPlayed(3);

        expect(audioQueue.queuedSamples).toBe(2);
      });

      it("does not go below zero", () => {
        audioQueue.enqueue(new Float32Array([1, 2]));

        audioQueue.onSamplesPlayed(10);

        expect(audioQueue.queuedSamples).toBe(0);
      });

      it("ignores non-positive values", () => {
        audioQueue.enqueue(new Float32Array([1, 2, 3]));

        audioQueue.onSamplesPlayed(0);
        audioQueue.onSamplesPlayed(-1);

        expect(audioQueue.queuedSamples).toBe(3);
      });

      it("reduces and pops markers", () => {
        audioQueue.enqueue(new Float32Array([1, 2, 3, 4, 5]));
        audioQueue.addMarker("marker1");

        audioQueue.onSamplesPlayed(5);

        expect(mockSocket.emit).toHaveBeenCalledWith("user_message", {
          marker: "marker1",
        });
        expect(audioQueue.marks).toHaveLength(0);
      });
    });

    describe("markers", () => {
      describe("addMarker", () => {
        it("adds marker with current queuedSamples as bytesToGo", () => {
          audioQueue.enqueue(new Float32Array([1, 2, 3, 4, 5]));
          audioQueue.addMarker("marker1");

          expect(audioQueue.marks).toHaveLength(1);
          expect(audioQueue.marks[0]).toEqual({
            id: "marker1",
            bytesToGo: 5,
          });
        });

        it("adds multiple markers with correct bytesToGo", () => {
          audioQueue.enqueue(new Float32Array([1, 2, 3]));
          audioQueue.addMarker("marker1");

          audioQueue.enqueue(new Float32Array([4, 5]));
          audioQueue.addMarker("marker2");

          expect(audioQueue.marks).toHaveLength(2);
          expect(audioQueue.marks[0]).toEqual({
            id: "marker1",
            bytesToGo: 3,
          });
          expect(audioQueue.marks[1]).toEqual({
            id: "marker2",
            bytesToGo: 5,
          });
        });

        it("adds marker with 0 bytesToGo when queue is empty", () => {
          audioQueue.addMarker("marker1");

          expect(audioQueue.marks[0]).toEqual({
            id: "marker1",
            bytesToGo: 0,
          });
        });
      });

      describe("reduceMarkers", () => {
        it("reduces bytesToGo for all markers", () => {
          audioQueue.enqueue(new Float32Array([1, 2, 3, 4, 5]));
          audioQueue.addMarker("marker1");
          audioQueue.addMarker("marker2");

          audioQueue.reduceMarkers(2);

          expect(audioQueue.marks[0].bytesToGo).toBe(3);
          expect(audioQueue.marks[1].bytesToGo).toBe(3);
        });

        it("allows bytesToGo to become negative", () => {
          audioQueue.enqueue(new Float32Array([1, 2]));
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
          audioQueue.enqueue(new Float32Array([1, 2, 3, 4, 5]));
          audioQueue.addMarker("marker1");
          audioQueue.addMarker("marker2");

          audioQueue.reduceMarkers(2);
          audioQueue.popMarkers();

          expect(audioQueue.marks).toHaveLength(2);
          expect(mockSocket.emit).not.toHaveBeenCalled();
        });

        it("pops only markers with bytesToGo <= 0", () => {
          audioQueue.enqueue(new Float32Array([1, 2, 3, 4, 5]));
          audioQueue.addMarker("marker1");
          audioQueue.enqueue(new Float32Array([6, 7, 8]));
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

      describe("onSamplesPlayed with markers", () => {
        it("reduces markers and pops them when samples are played", () => {
          audioQueue.enqueue(new Float32Array([1, 2, 3, 4, 5]));
          audioQueue.addMarker("marker1");

          audioQueue.onSamplesPlayed(5);

          expect(mockSocket.emit).toHaveBeenCalledWith("user_message", {
            marker: "marker1",
          });
          expect(audioQueue.marks).toHaveLength(0);
        });
      });
    });

    describe("clear", () => {
      it("resets queuedSamples and marks", () => {
        audioQueue.enqueue(new Float32Array([1, 2, 3, 4, 5]));
        audioQueue.addMarker("marker1");
        audioQueue.addMarker("marker2");

        audioQueue.clear();

        expect(audioQueue.queuedSamples).toBe(0);
        expect(audioQueue.marks).toHaveLength(0);
      });

      it("sends clear message to worklet", () => {
        const node = getPlaybackNode();
        node.port.postMessage.mockClear();

        audioQueue.clear();

        expect(node.port.postMessage).toHaveBeenCalledWith({ type: "clear" });
      });

      it("handles already empty queue", () => {
        audioQueue.clear();

        expect(audioQueue.queuedSamples).toBe(0);
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
      MockAudioWorkletNode.instances = [];
      mockSocket = createMockSocket();
      audioQueue = await setupAudioPlayback(mockSocket, mockLogError, 48000);
      addData = addDataToAudioQueue(audioQueue);
    });

    describe("audio data", () => {
      it("adds audio data to queue", () => {
        const audioData = new Float32Array([0.5, -0.5, 0.25]);
        const MAX_INT16_VALUE = 0x7fff;
        const intArray = Int16Array.from(
          audioData,
          (x) => x * MAX_INT16_VALUE,
        );
        const base64Audio = Buffer.from(intArray.buffer).toString("base64");

        addData(JSON.stringify({ audio: base64Audio }));

        expect(audioQueue.queuedSamples).toBeGreaterThan(0);
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
        audioQueue.enqueue(new Float32Array([1, 2, 3, 4, 5]));
        audioQueue.addMarker("marker1");

        addData(JSON.stringify({ interruptPlayback: true }));

        expect(audioQueue.queuedSamples).toBe(0);
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
