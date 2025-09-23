const bufferSize = 128
const sampleRate = 8000
const audioOptions = {
  audio: {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
  },
}

const arrayBufferToBase64 = (buffer: ArrayBufferLike): string => {
  let binary = ''
  const bytes = new Uint8Array(buffer)
  const len = bytes.byteLength
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i])
  }
  return window.btoa(binary)
}

const base64ToArrayBuffer = (s: string): ArrayBuffer => {
  const binary_string = window.atob(s)
  const len = binary_string.length
  const bytes = new Uint8Array(len)
  for (let i = 0; i < len; i++) {
    bytes[i] = binary_string.charCodeAt(i)
  }
  return bytes.buffer
}

const floatToIntArray = (arr: Float32Array): Int32Array => {
  // Convert Float Array [-1, 1] to full range int array
  return Int32Array.from(arr, (x) => x * 0x7fffffff)
}

const intToFloatArray = (arr: Int32Array): Float32Array => {
  return Float32Array.from(arr, (x) => x / 0x7fffffff)
}

interface Mark {
  id: string
  bytesToGo: number
}

interface AudioQueue {
  buffer: Float32Array
  marks: Array<Mark>
  socket: WebSocket
  write: (newAudio: Float32Array) => void
  read: (nSamples: number) => Float32Array
  length: () => number
  addMarker: (id: string) => void
  reduceMarkers: (bytesRead: number) => void
  popMarkers: () => void
  clear: () => void
}

const createAudioQueue = (socket: WebSocket): AudioQueue => {
  return {
    buffer: new Float32Array(0),
    marks: new Array<Mark>(),
    socket,

    write: function (newAudio: Float32Array) {
      const currentQLength = this.buffer.length
      const newBuffer = new Float32Array(currentQLength + newAudio.length)
      newBuffer.set(this.buffer, 0)
      newBuffer.set(newAudio, currentQLength)
      this.buffer = newBuffer
    },

    read: function (nSamples: number) {
      const samplesToPlay = this.buffer.subarray(0, nSamples)
      this.buffer = this.buffer.subarray(nSamples, this.buffer.length)
      this.reduceMarkers(samplesToPlay.length)
      this.popMarkers()
      return samplesToPlay
    },

    length: function () {
      return this.buffer.length
    },

    addMarker: function (id: string) {
      this.marks.push({ id, bytesToGo: this.length() })
    },

    reduceMarkers: function (bytesRead: number) {
      this.marks = this.marks.map((m) => {
        return { id: m.id, bytesToGo: m.bytesToGo - bytesRead }
      })
    },

    popMarkers: function () {
      // marks are ordered
      let popUpTo = 0
      while (popUpTo < this.marks.length) {
        if (this.marks[popUpTo].bytesToGo <= 0) {
          popUpTo += 1
        } else {
          break
        }
      }
      const marksToPop = this.marks.slice(0, popUpTo)
      this.marks = this.marks.slice(popUpTo, this.marks.length)
      marksToPop.forEach((m) => {
        if (this.socket.readyState === WebSocket.OPEN) {
          this.socket.send(JSON.stringify({ marker: m.id }))
        }
      })
    },

    /**
     * Clears the audio queue, removing all buffered audio and markers.
     */
    clear: function () {
      this.buffer = new Float32Array(0)
      this.marks = []
    },
  }
}

const streamMicrophoneToServer = async (socket: WebSocket) => {
  const audioContext = new AudioContext({ sampleRate })

  try {
    const audioStream = await navigator.mediaDevices.getUserMedia(audioOptions)
    await audioContext.audioWorklet.addModule(
      new URL('./microphone-processor.js', import.meta.url).href,
    )

    const microphoneNode = new AudioWorkletNode(
      audioContext,
      'microphone-processor',
    )
    microphoneNode.port.onmessage = (event: MessageEvent) => {
      const audioData = event.data as Float32Array
      const message = JSON.stringify({
        audio: arrayBufferToBase64(floatToIntArray(audioData).buffer),
      })
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(message)
      }
    }
    const source = audioContext.createMediaStreamSource(audioStream)
    source.connect(microphoneNode).connect(audioContext.destination)
  } catch (err) {
    console.error(err)
  }
}

const setupAudioPlayback = async (socket: WebSocket): Promise<AudioQueue> => {
  const audioQueue = createAudioQueue(socket)
  const audioOutputContext = new AudioContext({ sampleRate })

  // Resume the audio context (browsers often start it in suspended state)
  if (audioOutputContext.state === 'suspended') {
    await audioOutputContext.resume()
  }

  await audioOutputContext.audioWorklet.addModule(
    new URL('./playback-processor.js', import.meta.url).href,
  )

  const playbackNode = new AudioWorkletNode(
    audioOutputContext,
    'playback-processor',
  )

  playbackNode.port.onmessage = (event: MessageEvent) => {
    if (event.data === 'need-more-data') {
      // Request more audio data from the queue
      const audioData = audioQueue.length()
        ? audioQueue.read(bufferSize)
        : new Float32Array(bufferSize)

      if (!(audioData instanceof Float32Array)) {
        console.error('audioData is invalid, sending silence.')
        playbackNode.port.postMessage(new Float32Array(bufferSize))
      } else if (audioData.length !== bufferSize) {
        console.warn(
          `audioData is too short (${audioData.length} samples), padding with silence`,
        )
        const padded = new Float32Array(bufferSize)
        padded.set(audioData)
        playbackNode.port.postMessage(padded)
      } else {
        playbackNode.port.postMessage(audioData)
      }
    }
  }

  playbackNode.connect(audioOutputContext.destination)

  return audioQueue
}

const addDataToAudioQueue =
  (audioQueue: AudioQueue, onLatencyUpdate?: (latency: any) => void) =>
  (message: MessageEvent<any>) => {
    try {
      const data = JSON.parse(message.data.toString())
      if (data['error']) {
        console.error('Error from server:', data['error'])
      }
      if (data['audio']) {
        const audioBytes = base64ToArrayBuffer(data['audio'])
        const int32Data = new Int32Array(audioBytes)
        const audioData = intToFloatArray(int32Data)
        audioQueue.write(audioData)
      } else if (data['marker']) {
        if (data['latency'] && onLatencyUpdate) {
          onLatencyUpdate(data['latency'])
        }
        console.log('Voice Latency Metrics:', data['latency'])
        audioQueue.addMarker(data['marker'])
      } else if (data['interruptPlayback']) {
        // User interrupted the bot, immediately clear the audio queue
        audioQueue.clear()
        console.log('Audio queue cleared due to user interruption.')
      }
    } catch (error) {
      console.error('Error processing server incoming audio data:', error)
    }
  }

/**
 * Constructs a WebSocket URL for browser audio from a base HTTP/HTTPS URL
 *
 * @param baseUrl - The base URL (e.g., "https://example.com" or "http://localhost:5005")
 * @returns WebSocket URL for browser audio endpoint
 *
 * @example
 * getWebSocketUrl("https://example.com")
 * // Returns: "wss://example.com/webhooks/browser_audio/websocket"
 *
 * getWebSocketUrl("http://localhost:5005")
 * // Returns: "ws://localhost:5005/webhooks/browser_audio/websocket"
 *
 * @throws {TypeError} If baseUrl is not a valid URL
 */
function getWebSocketUrl(baseUrl: string) {
  const url = new URL(baseUrl)
  const wsProtocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${wsProtocol}//${url.host}/webhooks/browser_audio/websocket`
}

/**
 * Creates a WebSocket connection for browser audio and streams microphone input to the server
 *
 * @param baseUrl - The base URL (e.g., "https://example.com" or "http://localhost:5005")
 * @param onLatencyUpdate - Optional callback function to receive latency updates
 */
export async function createAudioConnection(
  baseUrl: string,
  onLatencyUpdate?: (latency: any) => void,
) {
  const websocketURL = getWebSocketUrl(baseUrl)
  const socket = new WebSocket(websocketURL)

  socket.onopen = async () => {
    await streamMicrophoneToServer(socket)
  }

  const audioQueue = await setupAudioPlayback(socket)
  socket.onmessage = addDataToAudioQueue(audioQueue, onLatencyUpdate)
}
