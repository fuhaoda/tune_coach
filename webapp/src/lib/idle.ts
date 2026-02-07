export function shouldAutoPause(lastActivityAtMs: number, nowMs: number, idlePauseMs: number): boolean {
  if (idlePauseMs <= 0) {
    return false
  }
  return nowMs - lastActivityAtMs >= idlePauseMs
}

export function isVoiceActivity(voiced: boolean, rms: number, threshold = 0.02): boolean {
  if (voiced) {
    return true
  }
  return rms >= threshold
}
