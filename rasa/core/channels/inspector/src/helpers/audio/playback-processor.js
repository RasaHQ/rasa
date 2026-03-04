/**
 * Dynamic ring buffer for audio sample storage.
 * Uses efficient ring buffer operations but grows when needed to avoid data loss.
 */
class RingBuffer {
  constructor(initialCapacity) {
    this.buffer = new Float32Array(initialCapacity)
    this.capacity = initialCapacity
    this.readIndex = 0
    this.writeIndex = 0
    this.length = 0
  }

  /**
   * Grow the buffer to accommodate more samples.
   * Linearizes existing data in the process.
   */
  grow(minCapacity) {
    const newCapacity = Math.max(minCapacity, this.capacity * 2)
    const newBuffer = new Float32Array(newCapacity)

    // Copy existing data in order (linearize the ring)
    if (this.length > 0) {
      if (this.readIndex < this.writeIndex) {
        // Data is contiguous
        newBuffer.set(this.buffer.subarray(this.readIndex, this.writeIndex))
      } else {
        // Data wraps around
        const firstPart = this.buffer.subarray(this.readIndex, this.capacity)
        const secondPart = this.buffer.subarray(0, this.writeIndex)
        newBuffer.set(firstPart)
        newBuffer.set(secondPart, firstPart.length)
      }
    }

    this.buffer = newBuffer
    this.capacity = newCapacity
    this.readIndex = 0
    this.writeIndex = this.length
  }

  /**
   * Write samples into the ring buffer.
   * Buffer grows if needed - no samples are ever discarded.
   */
  write(samples) {
    const samplesToWrite = samples.length

    // Grow buffer if needed
    const requiredCapacity = this.length + samplesToWrite
    if (requiredCapacity > this.capacity) {
      this.grow(requiredCapacity)
    }

    // Write samples, handling wrap-around
    const firstChunk = Math.min(samplesToWrite, this.capacity - this.writeIndex)
    this.buffer.set(samples.subarray(0, firstChunk), this.writeIndex)

    if (firstChunk < samplesToWrite) {
      // Wrap around to beginning
      this.buffer.set(samples.subarray(firstChunk), 0)
    }

    this.writeIndex = (this.writeIndex + samplesToWrite) % this.capacity
    this.length += samplesToWrite
  }

  /**
   * Read samples from the ring buffer into the destination array.
   * Returns the number of samples actually read.
   */
  read(destination) {
    const samplesToRead = Math.min(destination.length, this.length)

    if (samplesToRead === 0) {
      return 0
    }

    // Read samples, handling wrap-around
    const firstChunk = Math.min(samplesToRead, this.capacity - this.readIndex)
    destination.set(this.buffer.subarray(this.readIndex, this.readIndex + firstChunk))

    if (firstChunk < samplesToRead) {
      // Wrap around to beginning
      destination.set(this.buffer.subarray(0, samplesToRead - firstChunk), firstChunk)
    }

    this.readIndex = (this.readIndex + samplesToRead) % this.capacity
    this.length -= samplesToRead

    return samplesToRead
  }

  /**
   * Clear all buffered samples.
   */
  clear() {
    this.readIndex = 0
    this.writeIndex = 0
    this.length = 0
  }
}

// Default buffer capacity: ~10 seconds at 48kHz sample rate
const DEFAULT_BUFFER_CAPACITY = 48000 * 10

class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super()
    this.ringBuffer = new RingBuffer(DEFAULT_BUFFER_CAPACITY)

    // Push-based audio feed from main thread.
    this.port.onmessage = (event) => {
      const message = event.data
      if (message?.type === 'clear') {
        this.ringBuffer.clear()
        return
      }
      if (message?.type === 'audio' && message.data) {
        this.ringBuffer.write(message.data)
      }
    }
  }

  process(_, outputs) {
    const output = outputs[0]
    const channelData = output[0]

    const samplesRead = this.ringBuffer.read(channelData)

    // Fill remaining samples with silence if buffer didn't have enough
    if (samplesRead < channelData.length) {
      channelData.fill(0, samplesRead)
    }

    // Tell main thread how much queued audio was actually consumed.
    this.port.postMessage({ type: 'played-samples', samples: samplesRead })

    return true
  }
}

registerProcessor('playback-processor', PlaybackProcessor)
