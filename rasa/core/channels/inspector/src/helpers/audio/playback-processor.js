class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super()
    this.audioBuffer = new Float32Array(0)

    // Set up message handling from the main thread
    this.port.onmessage = (event) => {
      this.audioBuffer = event.data
    }

    // Request initial audio data
    this.port.postMessage('need-more-data')
  }

  process(_, outputs) {
    const output = outputs[0]
    const channelData = output[0]

    const availableSamples = this.audioBuffer.length
    const requestedSamples = channelData.length
    const samplesToCopy = Math.min(availableSamples, requestedSamples)

    if (samplesToCopy > 0) {
      channelData.set(this.audioBuffer.subarray(0, samplesToCopy))
      this.audioBuffer = this.audioBuffer.subarray(samplesToCopy)
    } else {
      channelData.fill(0)
    }

    this.port.postMessage('need-more-data')

    return true
  }
}

registerProcessor('playback-processor', PlaybackProcessor)
