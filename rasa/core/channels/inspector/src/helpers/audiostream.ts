const bufferSize = 4096
const sampleRate = 8000
const audioOptions = {
  audio: {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true
  }
}


const floatToIntArray = (arr: Float32Array): Int32Array => {
  // Convert Float Array [-1, 1] to full range int array
  return Int32Array.from(arr, x => x * 0x7fffffff)
}

const intToFloatArray = (arr: Int32Array): Float32Array => {
  return Float32Array.from(arr, x => (x / 0x7fffffff))
}

interface AudioQueue {
  buffer: Float32Array;
  write: (newAudio: Float32Array) => void;
  read: (nSamples: number) => Float32Array;
  length: () => number;
}


const createAudioQueue = () : AudioQueue => {
  return {
    buffer: new Float32Array(0),

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
      return samplesToPlay;
    },

    length: function() {
      return this.buffer.length;
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
      socket.send(floatToIntArray(event.inputBuffer.getChannelData(0)))
    }
    audioInput.connect(sender)
    sender.connect(audioContext.destination)
  } catch (err) {
    console.error(err);
  }
}

const setupAudioPlayback = (): AudioQueue => {
  const audioQueue = createAudioQueue()
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
  if (message.data instanceof Blob) {
    const reader = new FileReader();
    reader.onload = function () {
      const result = reader.result
      if (result && result instanceof ArrayBuffer) {
        const audioData = intToFloatArray(new Int32Array(result))
        audioQueue.write(audioData);
      }
    };
    reader.readAsArrayBuffer(message.data);
  }
}

export async function createAudioConnection() {
  const websocketURL  = "ws://localhost:5005/webhooks/browser_audio/websocket"
  const socket = new WebSocket(websocketURL)
  socket.onopen = async () => { await streamMicrophoneToServer(socket)}
  const audioQueue = setupAudioPlayback()
  socket.onmessage = addDataToAudioQueue(audioQueue)
}
