const bufferSize = 4096
const sampleRate = 8000
const audioOptions = {
  audio: {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true
  }
}

const arrayBufferToBase64 = ( buffer: ArrayBuffer ): string => {
    let binary = '';
    const bytes = new Uint8Array( buffer );
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode( bytes[ i ] );
    }
    return window.btoa( binary );
}

const base64ToArrayBuffer = ( s: string ): ArrayBuffer => {
    const binary_string =  window.atob(s);
    const len = binary_string.length;
    const bytes = new Uint8Array( len );
    for (let i = 0; i < len; i++)        {
        bytes[i] = binary_string.charCodeAt(i);
    }
    return bytes.buffer;
}

const floatToIntArray = (arr: Float32Array): Int32Array => {
  // Convert Float Array [-1, 1] to full range int array
  return Int32Array.from(arr, x => x * 0x7fffffff)
}

const intToFloatArray = (arr: Int32Array): Float32Array => {
  return Float32Array.from(arr, x => (x / 0x7fffffff))
}

interface Mark {
    id: string
    bytesToGo: number
}

interface AudioQueue {
  buffer: Float32Array;
  marks: Array<Mark>
  socket: WebSocket,
  write: (newAudio: Float32Array) => void;
  read: (nSamples: number) => Float32Array;
  length: () => number;
  addMarker: (id: string) => void;
  reduceMarkers: (bytesRead: number) => void;
  popMarkers: () => void;
}


const createAudioQueue = (socket: WebSocket) : AudioQueue => {
  return {
    buffer: new Float32Array(0),
    marks: new Array<Mark>(),
    socket,

    write: function(newAudio: Float32Array) {
      const currentQLength = this.buffer.length;
      const newBuffer = new Float32Array(currentQLength + newAudio.length);
      newBuffer.set(this.buffer, 0);
      newBuffer.set(newAudio, currentQLength);
      this.buffer = newBuffer;
    },

    read: function(nSamples: number) {
      const samplesToPlay = this.buffer.subarray(0, nSamples);
      this.buffer = this.buffer.subarray(nSamples, this.buffer.length);
      this.reduceMarkers(samplesToPlay.length)
      this.popMarkers()
      return samplesToPlay;
    },

    length: function() {
      return this.buffer.length;
    },

    addMarker: function(id: string) {
        this.marks.push({id, bytesToGo: this.length()})
    },

    reduceMarkers: function(bytesRead: number) {
        this.marks = this.marks.map((m) => {
            return {id: m.id, bytesToGo: m.bytesToGo - bytesRead}
        })
    },

    popMarkers: function() {
        // marks are ordered
        let popUpTo = 0;
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
            this.socket.send(JSON.stringify({marker: m.id}))
        })
    }

  };
}

const streamMicrophoneToServer = async (socket: WebSocket) => {
  let audioStream = null;
  const audioContext = new AudioContext({sampleRate});

  try {
    audioStream = await navigator.mediaDevices.getUserMedia(audioOptions);
    const audioInput = audioContext.createMediaStreamSource(audioStream)
    const sender = audioContext.createScriptProcessor(bufferSize, 1, 1)
    sender.onaudioprocess = function(event) {
      const message = JSON.stringify({
        "audio": arrayBufferToBase64(floatToIntArray(event.inputBuffer.getChannelData(0)).buffer)
      })
      socket.send(message)
    }
    audioInput.connect(sender)
    sender.connect(audioContext.destination)
  } catch (err) {
    console.error(err);
  }
}

const setupAudioPlayback = (socket: WebSocket): AudioQueue => {
  const audioQueue = createAudioQueue(socket)
  const silence = new Float32Array(bufferSize)
  const audioOutputContext = new AudioContext({sampleRate})
  const scriptNode = audioOutputContext.createScriptProcessor(bufferSize, 1, 1);
  scriptNode.onaudioprocess = function(e) {
    const audioData = audioQueue.length() ? audioQueue.read(bufferSize) : silence
    e.outputBuffer.getChannelData(0).set(audioData);
  }
  scriptNode.connect(audioOutputContext.destination)
  return audioQueue
}

const addDataToAudioQueue = (audioQueue: AudioQueue) => (message: MessageEvent<any>) => {
    const data = JSON.parse(message.data.toString())
    if (data["audio"]) {
      const audioBytes = base64ToArrayBuffer(data["audio"])
      const audioData = intToFloatArray(new Int32Array(audioBytes))
      audioQueue.write(audioData);
    } else if (data["marker"]) {
        audioQueue.addMarker(data["marker"])
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
export function getWebSocketUrl(baseUrl: string) {
  const url = new URL(baseUrl);
  const wsProtocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${wsProtocol}//${url.host}/webhooks/browser_audio/websocket`;
}

/**
 * Creates a WebSocket connection for browser audio and streams microphone input to the server
 * 
 * @param baseUrl - The base URL (e.g., "https://example.com" or "http://localhost:5005")
 */
export async function createAudioConnection(baseUrl: string) {
  const websocketURL = getWebSocketUrl(baseUrl)
  const socket = new WebSocket(websocketURL)
  socket.onopen = async () => { await streamMicrophoneToServer(socket)}
  const audioQueue = setupAudioPlayback(socket)
  socket.onmessage = addDataToAudioQueue(audioQueue)
}
