const audioOptions = {
  audio: {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
  },
}
interface SocketMessageRouter {
  waitForHandshake: () => Promise<number>
  setHandler: (handler: (event: MessageEvent<any>) => void) => void
  dispose: () => void
}

const createSocketMessageRouter = (socket: WebSocket): SocketMessageRouter => {
  let handshakeSampleRate: number | null = null
  let resolveHandshake: ((sampleRate: number) => void) | null = null
  let messageHandler: ((event: MessageEvent<any>) => void) | null = null
  const bufferedMessages: MessageEvent<any>[] = []

  const onMessage = (event: MessageEvent<any>) => {
    let data: any = null
    try {
      data = JSON.parse(event.data)
    } catch {
      // Ignore parse errors here; handler will do full validation later.
    }

    if (data?.type === 'handshake') {
      const sampleRate = Number(data.sample_rate)
      if (Number.isFinite(sampleRate) && sampleRate > 0) {
        handshakeSampleRate = sampleRate
        if (resolveHandshake) {
          resolveHandshake(sampleRate)
          resolveHandshake = null
        }
      }
      return
    }

    if (messageHandler) {
      messageHandler(event)
      return
    }

    bufferedMessages.push(event)
  }

  socket.addEventListener('message', onMessage)

  return {
    waitForHandshake: () =>
      new Promise<number>((resolve) => {
        if (handshakeSampleRate !== null) {
          resolve(handshakeSampleRate)
          return
        }
        resolveHandshake = resolve
      }),
    setHandler: (handler: (event: MessageEvent<any>) => void) => {
      messageHandler = handler
      while (bufferedMessages.length) {
        const event = bufferedMessages.shift()
        if (event) {
          messageHandler(event)
        }
      }
    },
    dispose: () => {
      socket.removeEventListener('message', onMessage)
      bufferedMessages.length = 0
      messageHandler = null
      resolveHandshake = null
    },
  }
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

const floatToIntArray = (arr: Float32Array): Int16Array => {
  // Convert Float Array [-1, 1] to full range int array
  // Full range of Int16 is -0x8000 to 0x7fff
  // so we multiply by 0x7fff to scale the float to the int range
  return Int16Array.from(arr, (x) => x * 0x7fff)
}

const intToFloatArray = (arr: Int16Array): Float32Array => {
  // Convert full range int array to Float Array [-1, 1]
  // Full range of Int16 is -0x8000 to 0x7fff
  // so we divide by 0x7fff to scale the int to the float range
  return Float32Array.from(arr, (x) => x / 0x7fff)
}

interface Mark {
  id: string
  bytesToGo: number
}

interface AudioQueue {
  marks: Array<Mark>
  queuedSamples: number
  socket: WebSocket
  enqueue: (newAudio: Float32Array) => void
  onSamplesPlayed: (samplesPlayed: number) => void
  addMarker: (id: string) => void
  reduceMarkers: (samplesPlayed: number) => void
  popMarkers: () => void
  clear: () => void
}

const createAudioQueue = (
  socket: WebSocket,
  playbackNode: AudioWorkletNode,
): AudioQueue => {
  return {
    marks: new Array<Mark>(),
    queuedSamples: 0,
    socket,

    enqueue: function (newAudio: Float32Array) {
      this.queuedSamples += newAudio.length
      playbackNode.port.postMessage(
        { type: 'audio', data: newAudio },
        [newAudio.buffer],
      )
    },

    addMarker: function (id: string) {
      this.marks.push({ id, bytesToGo: this.queuedSamples })
    },

    onSamplesPlayed: function (samplesPlayed: number) {
      if (samplesPlayed <= 0) {
        return
      }
      this.queuedSamples = Math.max(0, this.queuedSamples - samplesPlayed)
      this.reduceMarkers(samplesPlayed)
      this.popMarkers()
    },

    reduceMarkers: function (samplesPlayed: number) {
      this.marks = this.marks.map((m) => {
        return { id: m.id, bytesToGo: m.bytesToGo - samplesPlayed }
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
      this.queuedSamples = 0
      this.marks = []
      playbackNode.port.postMessage({ type: 'clear' })
    },
  }
}

const streamMicrophoneToServer = async (socket: WebSocket, sampleRate: number) => {
  console.log("Setting up microphone stream with sample rate:", sampleRate)
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

const setupAudioPlayback = async (socket: WebSocket, sampleRate: number): Promise<AudioQueue> => {
  console.log("Setting up audio playback with sample rate:", sampleRate)
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
  const audioQueue = createAudioQueue(socket, playbackNode)

  playbackNode.port.onmessage = (event: MessageEvent) => {
    if (event.data?.type === 'played-samples') {
      const samplesPlayed = Number(event.data.samples) || 0
      audioQueue.onSamplesPlayed(samplesPlayed)
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
        const audioData = intToFloatArray(new Int16Array(audioBytes))
        audioQueue.enqueue(audioData)
      } else if (data['marker']) {
        if (data['latency'] && onLatencyUpdate) {
          onLatencyUpdate(data['latency'])
        }
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
 * @param params - Optional query params (e.g. { language: "de" })
 * @returns WebSocket URL for browser audio endpoint
 */
function getWebSocketUrl(baseUrl: string, params?: { language?: string }) {
  const url = new URL(baseUrl)
  const wsProtocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  let path = `${wsProtocol}//${url.host}/webhooks/browser_audio/websocket`
  if (params?.language) {
    path += `?language=${encodeURIComponent(params.language)}`
  }
  return path
}

/**
 * Fetches supported languages for voice from the running agent.
 *
 * @param baseUrl - The base URL (e.g., "https://example.com" or "http://localhost:5005")
 * @returns List of language codes supported by the model
 */
export async function fetchSupportedLanguages(baseUrl: string): Promise<string[]> {
  const url = new URL('/webhooks/browser_audio/supported_languages', baseUrl).href
  const res = await fetch(url)
  if (!res.ok) return []
  const data = await res.json()
  return Array.isArray(data?.languages) ? data.languages : []
}

/**
 * Creates a WebSocket connection for browser audio and streams microphone input to the server
 *
 * @param baseUrl - The base URL (e.g., "https://example.com" or "http://localhost:5005")
 * @param onLatencyUpdate - Optional callback function to receive latency updates
 * @param language - Optional language code (passed as query param to the WebSocket URL)
 */
export async function createAudioConnection(
  baseUrl: string,
  onLatencyUpdate?: (latency: any) => void,
  language?: string,
) {
  const websocketURL = getWebSocketUrl(baseUrl, language ? { language } : undefined)
  const socket = new WebSocket(websocketURL)
  const messageRouter = createSocketMessageRouter(socket)

  // Wait for handshake, reply with preferred sample rate, then set up audio
  // Audio Format: Linear PCM, 16-bit, Mono, with the sample rate determined by the handshake
  const sampleRate = await messageRouter.waitForHandshake()

  await streamMicrophoneToServer(socket, sampleRate)
  const audioQueue = await setupAudioPlayback(socket, sampleRate)
  messageRouter.setHandler(addDataToAudioQueue(audioQueue, onLatencyUpdate))

  socket.addEventListener('close', () => {
    messageRouter.dispose()
  })
}
