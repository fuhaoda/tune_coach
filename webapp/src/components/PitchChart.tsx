import { useEffect, useRef } from 'react'

type Point = {
  t: number
  y: number | null
  centY: number | null
}

type Props = {
  points: Point[]
  showCentCurve: boolean
  nowSec: number
  onTogglePause?: () => void
  windowSeconds?: number
}

const MAX_Y = 20
const DEGREES_PER_OCTAVE = 7
const OCTAVE_GAP = 0
const TOP_PADDING = 10
const BOTTOM_PADDING = 24
const VISIBLE_MARGIN = 0.08
const PLOT_LEFT_PADDING = 6
const RIGHT_AXIS_WIDTH = 60

export function mapPitchYToDisplay(rawY: number): number {
  const octaveIndex = Math.max(0, Math.floor(rawY / DEGREES_PER_OCTAVE))
  return rawY + octaveIndex * OCTAVE_GAP
}

export function computeTimeWindow(nowSec: number, windowSeconds: number): { minT: number; maxT: number } {
  return {
    minT: nowSec - windowSeconds,
    maxT: nowSec
  }
}

export default function PitchChart({
  points,
  showCentCurve,
  nowSec,
  onTogglePause,
  windowSeconds = 24
}: Props): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }
    const ctx = canvas.getContext('2d')
    if (!ctx) {
      return
    }

    const rect = canvas.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    const width = Math.max(320, Math.floor(rect.width))
    const height = Math.max(220, Math.floor(rect.height))
    canvas.width = Math.floor(width * dpr)
    canvas.height = Math.floor(height * dpr)
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    ctx.clearRect(0, 0, width, height)
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, width, height)

    const plotArea = drawGrid(ctx, width, height, windowSeconds)

    const { minT, maxT } = computeTimeWindow(nowSec, windowSeconds)
    drawLine(ctx, points, minT, maxT, height, plotArea, 'y', 'rgba(27, 94, 154, 0.62)', 3.35)
    if (showCentCurve) {
      drawLine(ctx, points, minT, maxT, height, plotArea, 'centY', 'rgba(219, 143, 45, 0.95)', 1.9)
    }
  }, [points, showCentCurve, nowSec, windowSeconds])

  return (
    <canvas
      ref={canvasRef}
      className="pitch-chart"
      aria-label="Pitch chart"
      onClick={() => {
        onTogglePause?.()
      }}
    />
  )
}

function drawGrid(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  windowSeconds: number
): { plotLeft: number; plotRight: number } {
  const plotLeft = PLOT_LEFT_PADDING
  const plotRight = Math.max(plotLeft + 80, width - RIGHT_AXIS_WIDTH)
  const plotWidth = plotRight - plotLeft
  const plotBottom = height - BOTTOM_PADDING - 6

  ctx.strokeStyle = 'rgba(0, 0, 0, 0.14)'
  ctx.lineWidth = 1
  for (let second = 0; second <= windowSeconds; second += 1) {
    const x = plotLeft + (second / windowSeconds) * plotWidth
    ctx.beginPath()
    ctx.moveTo(x, 0)
    ctx.lineTo(x, plotBottom)
    ctx.stroke()
  }

  ctx.strokeStyle = 'rgba(0, 0, 0, 0.1)'
  for (let y = 0; y <= MAX_Y; y += 1) {
    const py = yToPx(mapPitchYToDisplay(y), height)
    ctx.beginPath()
    ctx.moveTo(plotLeft, py)
    ctx.lineTo(plotRight, py)
    ctx.stroke()
  }

  const octaveSevens = [6, 13, 20]
  ctx.strokeStyle = 'rgba(205, 93, 58, 0.58)'
  ctx.lineWidth = 1.8
  for (const y of octaveSevens) {
    const py = yToPx(mapPitchYToDisplay(y), height)
    ctx.beginPath()
    ctx.moveTo(plotLeft, py)
    ctx.lineTo(plotRight, py)
    ctx.stroke()
  }

  ctx.strokeStyle = 'rgba(0, 0, 0, 0.18)'
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(plotRight, 0)
  ctx.lineTo(plotRight, plotBottom)
  ctx.stroke()

  drawTimeAxis(ctx, plotLeft, plotRight, plotBottom, windowSeconds)

  ctx.fillStyle = '#444'
  ctx.font = '12px "Avenir Next", "Segoe UI", sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  const slotHeight = Math.abs(
    yToPx(mapPitchYToDisplay(1), height) - yToPx(mapPitchYToDisplay(0), height)
  )
  const labelX = plotRight + (width - plotRight) / 2
  for (let octave = 0; octave < 3; octave += 1) {
    for (let degree = 1; degree <= 7; degree += 1) {
      const rawY = octave * DEGREES_PER_OCTAVE + (degree - 1)
      const py = yToPx(mapPitchYToDisplay(rawY), height)
      const label = String(degree)
      ctx.fillText(label, labelX, py)
      const metrics = ctx.measureText(label)
      const ascent = metrics.actualBoundingBoxAscent || 8
      const descent = metrics.actualBoundingBoxDescent || 3
      const lowDotOffset = Math.min(slotHeight * 0.34, descent + 1.85)
      const highDotOffset = Math.min(slotHeight * 0.48, ascent + 3)
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.2)'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(plotRight + 2, py)
      ctx.lineTo(plotRight + 9, py)
      ctx.stroke()
      if (octave === 0) {
        drawDot(ctx, labelX, py + lowDotOffset)
      } else if (octave === 2) {
        drawDot(ctx, labelX, py - highDotOffset)
      }
    }
  }

  return { plotLeft, plotRight }
}

function drawLine(
  ctx: CanvasRenderingContext2D,
  points: Array<{ t: number; y: number | null; centY: number | null }>,
  minT: number,
  maxT: number,
  height: number,
  plotArea: { plotLeft: number; plotRight: number },
  key: 'y' | 'centY',
  color: string,
  lineWidth: number
): void {
  const plotWidth = Math.max(1, plotArea.plotRight - plotArea.plotLeft)
  ctx.strokeStyle = color
  ctx.lineWidth = lineWidth

  let started = false
  ctx.beginPath()

  for (const p of points) {
    if (p.t < minT) {
      continue
    }
    const value = p[key]
    if (value === null || Number.isNaN(value)) {
      started = false
      continue
    }
    const x =
      plotArea.plotLeft + ((p.t - minT) / Math.max(0.0001, maxT - minT)) * plotWidth
    const y = yToPx(mapPitchYToDisplay(value), height)
    if (!started) {
      ctx.moveTo(x, y)
      started = true
    } else {
      ctx.lineTo(x, y)
    }
  }

  ctx.stroke()
}

function yToPx(displayY: number, height: number): number {
  const maxDisplayY = mapPitchYToDisplay(MAX_Y)
  const minVisibleY = -0.5 - VISIBLE_MARGIN
  const maxVisibleY = maxDisplayY + 0.5 + VISIBLE_MARGIN
  const range = maxVisibleY - minVisibleY
  return TOP_PADDING + ((maxVisibleY - displayY) / range) * (height - TOP_PADDING - BOTTOM_PADDING)
}

function drawTimeAxis(
  ctx: CanvasRenderingContext2D,
  plotLeft: number,
  plotRight: number,
  plotBottom: number,
  windowSeconds: number
): void {
  const axisY = plotBottom + 6
  const plotWidth = Math.max(1, plotRight - plotLeft)
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.25)'
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(plotLeft, axisY)
  ctx.lineTo(plotRight, axisY)
  ctx.stroke()

  ctx.fillStyle = '#666'
  ctx.font = '11px "Avenir Next", "Segoe UI", sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'top'

  const tickCount = 4
  const step = windowSeconds / tickCount
  for (let i = 0; i <= tickCount; i += 1) {
    const x = plotLeft + (i / tickCount) * plotWidth
    ctx.beginPath()
    ctx.moveTo(x, axisY)
    ctx.lineTo(x, axisY - 4)
    ctx.stroke()
    const label = Math.round(i * step - windowSeconds)
    const text = i === tickCount ? `${label}s` : String(label)
    ctx.fillText(text, x, axisY + 2)
  }
}

function drawDot(ctx: CanvasRenderingContext2D, x: number, y: number): void {
  ctx.save()
  ctx.fillStyle = '#444'
  ctx.beginPath()
  ctx.arc(x, y, 1.8, 0, 2 * Math.PI)
  ctx.fill()
  ctx.restore()
}
