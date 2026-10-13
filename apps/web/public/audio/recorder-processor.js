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
    /**
     * How many samples the last "stream" message already covered. The
     * consumer needs this: every snapshot starts at index 0, so without an
     * offset it would re-score old frames on every message, which stalls
     * calibration and double-counts coughs.
     */
    this.sentSamples = 0;
    this.port.onmessage = (e) => {
      if (e.data === "start") {
        this.buffer = [];
        this.recording = true;
        this.samplesSinceStream = 0;
        this.sentSamples = 0;
      } else if (e.data === "stop") {
        this.recording = false;
        this.flush();
      }
    };
  }
  process(inputs) {
    if (!this.recording) return true;

    const input = inputs[0];
    // A device that publishes an empty capture delivers frames with no
    // channels at all. Append silence for them so the 100 ms stream cadence
    // (the recorder's proof the graph is running) still fires and the UI can
    // calibrate or report a quiet take instead of hanging on "0%".
    const channel = input.length > 0 ? input[0] : null;
    const frames = channel ? channel.length : 128;
    for (let i = 0; i < frames; i++) {
      if (this.buffer.length < this.maxSamples) {
        this.buffer.push(channel ? channel[i] : 0);
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

    return true;
  }

  stream() {
    if (this.buffer.length > 0) {
      this.port.postMessage({
        type: "stream",
        data: new Float32Array(this.buffer),
        offset: this.sentSamples,
      });
      this.sentSamples = this.buffer.length;
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