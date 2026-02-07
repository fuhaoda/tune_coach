import { describe, expect, it } from 'vitest'

import { degreeToFrequency, shiftDegree } from '../lib/audioMath'

describe('audio math', () => {
  it('computes a valid major scale degree frequency', () => {
    const f = degreeToFrequency(5, 0, 130.8, 'Just Intonation', 0)
    expect(f).toBeGreaterThan(190)
    expect(f).toBeLessThan(200)
  })

  it('wraps degree shift across octaves', () => {
    expect(shiftDegree(7, 1)).toEqual({ degree: 1, octaveShift: 1 })
    expect(shiftDegree(1, -1)).toEqual({ degree: 7, octaveShift: -1 })
  })
})
