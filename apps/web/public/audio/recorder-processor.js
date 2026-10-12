class RecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = [];
    this.recording = false;
    // sampleRate is a global in AudioWorkletGlobalScope
    // eslint-disable-next-line no-undef
    this.sampleRate = sampleRate;
    this.maxSamples = 30 * this.sampleRate; // 30 seconds max
    this.streamInterval = Math.floor(0.1 * this.sampleRate); // ~100ms
    this.samplesSinceStream = 0;
    this.port.onmessage = (e) => {
      if (e.data === "start") {
        this.buffer = [];
        this.recording = true;
        this.samplesSinceStream = 0;
      } else if (e.data === "stop") {
        this.recording = false;
        this.flush();
      }
    };
  }
  process(inputs) {
    if (!this.recording) return true;

    const input = inputs[0];
    if (input.length > 0) {
      const channel = input[0];
      for (let i = 0; i < channel.length; i++) {
        if (this.buffer.length < this.maxSamples) {
          this.buffer.push(channel[i]);
        }
        this.samplesSinceStream++;
      }

      // Stream partial buffer every ~100ms for live visualization
      if (this.samplesSinceStream >= this.streamInterval) {
        this.stream();
        this.samplesSinceStream = 0;
      }

      if (this.buffer.length >= this.maxSamples) {
        this.recording = false;
        this.flush();
      }
    }
    return true;
  }

  stream() {
    if (this.buffer.length > 0) {
      this.port.postMessage({ type: "stream", data: new Float32Array(this.buffer) });
    }
  }

  flush() {
    if (this.buffer.length > 0) {
      this.port.postMessage({ type: "complete", data: new Float32Array(this.buffer) });
      this.buffer = [];
    }
  }
}

registerProcessor("recorder-processor", RecorderProcessor);