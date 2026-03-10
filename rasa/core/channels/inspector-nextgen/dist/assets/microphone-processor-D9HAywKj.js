(function(){"use strict";class o extends AudioWorkletProcessor{process(e){const s=e[0];if(s.length>0){const r=s[0];this.port.postMessage(r)}return!0}}registerProcessor("microphone-processor",o)})();
