class MicForwarderProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0]
    if (input && input.length > 0 && input[0] && input[0].length > 0) {
      const mono = input[0]
      this.port.postMessage(new Float32Array(mono))
    }
    return true
  }
}

registerProcessor('mic-forwarder', MicForwarderProcessor)
