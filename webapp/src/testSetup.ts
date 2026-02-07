import '@testing-library/jest-dom'

Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
  value: () => ({
    setTransform: () => undefined,
    clearRect: () => undefined,
    fillRect: () => undefined,
    beginPath: () => undefined,
    moveTo: () => undefined,
    lineTo: () => undefined,
    stroke: () => undefined,
    fillText: () => undefined,
    strokeText: () => undefined,
    measureText: () => ({
      actualBoundingBoxAscent: 8,
      actualBoundingBoxDescent: 3
    }),
    save: () => undefined,
    restore: () => undefined,
    arc: () => undefined,
    fill: () => undefined
  })
})
