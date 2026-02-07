import { TOUCH_KEYPAD_GRID } from '../lib/keypad'

type Props = {
  shiftPressed: boolean
  controlPressed: boolean
  onShiftDown: () => void
  onShiftUp: () => void
  onControlDown: () => void
  onControlUp: () => void
  onDigitDown: (digit: number, pointerId: number) => void
  onDigitUp: (digit: number, pointerId: number) => void
}

export default function TouchPad({
  shiftPressed,
  controlPressed,
  onShiftDown,
  onShiftUp,
  onControlDown,
  onControlUp,
  onDigitDown,
  onDigitUp
}: Props): JSX.Element {
  return (
    <div className="touch-pad" aria-label="Touch keyboard">
      <div className="modifiers">
        <button
          type="button"
          className={`modifier ${shiftPressed ? 'active' : ''}`}
          onPointerDown={(e) => {
            e.preventDefault()
            onShiftDown()
          }}
          onPointerUp={(e) => {
            e.preventDefault()
            onShiftUp()
          }}
          onPointerCancel={() => onShiftUp()}
          onPointerLeave={(e) => {
            if (e.buttons === 0) {
              onShiftUp()
            }
          }}
        >
          Oct+
        </button>
        <button
          type="button"
          className={`modifier ${controlPressed ? 'active' : ''}`}
          onPointerDown={(e) => {
            e.preventDefault()
            onControlDown()
          }}
          onPointerUp={(e) => {
            e.preventDefault()
            onControlUp()
          }}
          onPointerCancel={() => onControlUp()}
          onPointerLeave={(e) => {
            if (e.buttons === 0) {
              onControlUp()
            }
          }}
        >
          Oct-
        </button>
      </div>

      <div className="digits-grid">
        {TOUCH_KEYPAD_GRID.map((value, index) => {
          if (value === null) {
            return <div key={`empty-${index}`} className="digit empty" aria-hidden="true" />
          }
          return (
            <button
              key={value}
              type="button"
              className="digit"
              onPointerDown={(e) => {
                e.preventDefault()
                onDigitDown(value, e.pointerId)
              }}
              onPointerUp={(e) => {
                e.preventDefault()
                onDigitUp(value, e.pointerId)
              }}
              onPointerCancel={(e) => {
                e.preventDefault()
                onDigitUp(value, e.pointerId)
              }}
              onPointerLeave={(e) => {
                if (e.buttons === 0) {
                  onDigitUp(value, e.pointerId)
                }
              }}
            >
              {value}
            </button>
          )
        })}
      </div>
    </div>
  )
}
