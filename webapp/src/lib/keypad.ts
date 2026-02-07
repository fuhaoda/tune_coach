export type KeypadCell = number | null

export const TOUCH_KEYPAD_GRID: KeypadCell[] = [
  7,
  null,
  null,
  4,
  5,
  6,
  1,
  2,
  3
]

export function resolveOctave(shiftPressed: boolean, controlPressed: boolean): number {
  if (shiftPressed) {
    return 1
  }
  if (controlPressed) {
    return -1
  }
  return 0
}
