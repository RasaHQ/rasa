/**
 * Dynamic ring buffer for audio sample storage.
 * Uses efficient ring buffer operations but grows when needed to avoid data loss.
 */
class RingBuffer {
  private buffer: Float32Array;
  private capacity: number;
  private readIndex: number;
  private writeIndex: number;
  public length: number;

  constructor(initialCapacity: number) {
    this.buffer = new Float32Array(initialCapacity);
    this.capacity = initialCapacity;
    this.readIndex = 0;
    this.writeIndex = 0;
    this.length = 0;
  }

  grow(minCapacity: number): void {
    const newCapacity = Math.max(minCapacity, this.capacity * 2);
    const newBuffer = new Float32Array(newCapacity);

    if (this.length > 0) {
      if (this.readIndex < this.writeIndex) {
        newBuffer.set(this.buffer.subarray(this.readIndex, this.writeIndex));
      } else {
        const firstPart = this.buffer.subarray(this.readIndex, this.capacity);
        const secondPart = this.buffer.subarray(0, this.writeIndex);
        newBuffer.set(firstPart);
        newBuffer.set(secondPart, firstPart.length);
      }
    }

    this.buffer = newBuffer;
    this.capacity = newCapacity;
    this.readIndex = 0;
    this.writeIndex = this.length;
  }

  write(samples: Float32Array): void {
    const samplesToWrite = samples.length;
    const requiredCapacity = this.length + samplesToWrite;
    if (requiredCapacity > this.capacity) {
      this.grow(requiredCapacity);
    }

    const firstChunk = Math.min(samplesToWrite, this.capacity - this.writeIndex);
    this.buffer.set(samples.subarray(0, firstChunk), this.writeIndex);

    if (firstChunk < samplesToWrite) {
      this.buffer.set(samples.subarray(firstChunk), 0);
    }

    this.writeIndex = (this.writeIndex + samplesToWrite) % this.capacity;
    this.length += samplesToWrite;
  }

  read(destination: Float32Array): number {
    const samplesToRead = Math.min(destination.length, this.length);

    if (samplesToRead === 0) {
      return 0;
    }

    const firstChunk = Math.min(
      samplesToRead,
      this.capacity - this.readIndex,
    );
    destination.set(
      this.buffer.subarray(this.readIndex, this.readIndex + firstChunk),
    );

    if (firstChunk < samplesToRead) {
      destination.set(
        this.buffer.subarray(0, samplesToRead - firstChunk),
        firstChunk,
      );
    }

    this.readIndex = (this.readIndex + samplesToRead) % this.capacity;
    this.length -= samplesToRead;

    return samplesToRead;
  }

  clear(): void {
    this.readIndex = 0;
    this.writeIndex = 0;
    this.length = 0;
  }
}

// ~10 seconds at 48 kHz
const DEFAULT_BUFFER_CAPACITY = 48000 * 10;

class PlaybackProcessor extends AudioWorkletProcessor {
  private ringBuffer: RingBuffer;

  constructor() {
    super();
    this.ringBuffer = new RingBuffer(DEFAULT_BUFFER_CAPACITY);

    this.port.onmessage = (
      event: MessageEvent<{ type: string; data?: Float32Array }>,
    ) => {
      const message = event.data;
      if (message.type === "clear") {
        this.ringBuffer.clear();
        return;
      }
      if (message.type === "audio" && message.data) {
        this.ringBuffer.write(message.data);
      }
    };
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const channelData = outputs[0][0];
    const samplesRead = this.ringBuffer.read(channelData);

    if (samplesRead < channelData.length) {
      channelData.fill(0, samplesRead);
    }

    this.port.postMessage({ type: "played-samples", samples: samplesRead });
    return true;
  }
}

registerProcessor("playback-processor", PlaybackProcessor);
