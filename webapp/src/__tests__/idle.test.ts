import { describe, expect, it } from 'vitest'

import { isVoiceActivity, shouldAutoPause } from '../lib/idle'

describe('idle helpers', () => {
  it('triggers auto pause when idle timeout is reached', () => {
    expect(shouldAutoPause(1_000, 61_000, 60_000)).toBe(true)
    expect(shouldAutoPause(1_000, 60_999, 60_000)).toBe(false)
  })

  it('detects activity from voiced or rms threshold', () => {
    expect(isVoiceActivity(true, 0.001)).toBe(true)
    expect(isVoiceActivity(false, 0.03)).toBe(true)
    expect(isVoiceActivity(false, 0.01)).toBe(false)
  })
})
