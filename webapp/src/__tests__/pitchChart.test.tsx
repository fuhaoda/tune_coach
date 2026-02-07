import { fireEvent, render } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import PitchChart, { computeTimeWindow, mapPitchYToDisplay } from '../components/PitchChart'

describe('pitch chart helpers', () => {
  it('computes scrolling time window from nowSec', () => {
    expect(computeTimeWindow(24, 24)).toEqual({ minT: 0, maxT: 24 })
    expect(computeTimeWindow(3, 24)).toEqual({ minT: -21, maxT: 3 })
  })

  it('maps pitch to display coordinates', () => {
    expect(mapPitchYToDisplay(6)).toBeCloseTo(6)
    expect(mapPitchYToDisplay(7)).toBeCloseTo(7)
    expect(mapPitchYToDisplay(20)).toBeCloseTo(20)
  })

  it('toggles pause when chart is clicked', () => {
    const onTogglePause = vi.fn()
    const { getByLabelText } = render(
      <PitchChart points={[]} showCentCurve={false} nowSec={5} onTogglePause={onTogglePause} />
    )

    fireEvent.click(getByLabelText('Pitch chart'))
    expect(onTogglePause).toHaveBeenCalledTimes(1)
  })
})
