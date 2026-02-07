export type Tuning = 'Equal Temperament' | 'Just Intonation'

const MAJOR_SCALE_SEMITONES = [0, 2, 4, 5, 7, 9, 11]
const JUST_INTONATION_RATIOS = [1, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 15 / 8]

export function degreeToFrequency(
  degree: number,
  octave: number,
  doHz: number,
  tuning: Tuning,
  keySemitone: number
): number {
  if (degree < 1 || degree > 7 || doHz <= 0) {
    return 0
  }
  const idx = degree - 1
  const rootHz = doHz * 2 ** (keySemitone / 12)
  if (tuning === 'Just Intonation') {
    return rootHz * JUST_INTONATION_RATIOS[idx] * 2 ** octave
  }
  const semitone = MAJOR_SCALE_SEMITONES[idx] + octave * 12
  return rootHz * 2 ** (semitone / 12)
}

export function shiftDegree(degree: number, steps: number): { degree: number; octaveShift: number } {
  const idx = degree - 1 + steps
  const octaveShift = Math.floor(idx / 7)
  const wrapped = ((idx % 7) + 7) % 7
  return { degree: wrapped + 1, octaveShift }
}

export function clampBpm(value: number): number {
  return Math.max(50, Math.min(160, value))
}
