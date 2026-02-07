import { describe, expect, it } from 'vitest'

import { resolveOctave, TOUCH_KEYPAD_GRID } from '../lib/keypad'

describe('keypad layout', () => {
  it('matches 7 _ _ / 4 5 6 / 1 2 3', () => {
    expect(TOUCH_KEYPAD_GRID).toEqual([7, null, null, 4, 5, 6, 1, 2, 3])
  })

  it('prefers Oct+ when Oct+ and Oct- are both pressed', () => {
    expect(resolveOctave(true, true)).toBe(1)
    expect(resolveOctave(true, false)).toBe(1)
    expect(resolveOctave(false, true)).toBe(-1)
    expect(resolveOctave(false, false)).toBe(0)
  })
})
