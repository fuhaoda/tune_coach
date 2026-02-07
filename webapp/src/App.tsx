import { useCallback, useEffect, useRef, useState } from 'react'

import PitchChart from './components/PitchChart'
import TouchPad from './components/TouchPad'
import { clampBpm, degreeToFrequency, type Tuning } from './lib/audioMath'
import { isVoiceActivity, shouldAutoPause } from './lib/idle'
import { resolveOctave } from './lib/keypad'
import { encodeWav } from './lib/wav'

type PitchPoint = {
  t: number
  y: number | null
  centY: number | null
  hz: number | null
}

type ServerEvent = {
  type: string
  [key: string]: unknown
}

type Instrument = 'Piano' | 'Guitar'
type BrowserWindow = Window &
  typeof globalThis & {
    webkitAudioContext?: {
      new (): AudioContext
    }
  }

const TUNINGS: Tuning[] = ['Equal Temperament', 'Just Intonation']
const KEY_OPTIONS = [
  ['1=C', 0],
  ['1=C#/Db', 1],
  ['1=D', 2],
  ['1=D#/Eb', 3],
  ['1=E', 4],
  ['1=F', 5],
  ['1=F#/Gb', 6],
  ['1=G', 7],
  ['1=G#/Ab', 8],
  ['1=A', 9],
  ['1=A#/Bb', 10],
  ['1=B', 11]
] as const

const DEFAULT_DO = 130.8
const FEMALE_DO = 261.6
const WINDOW_SECONDS = 24
const IDLE_PAUSE_MS = 60_000
const IDLE_CHECK_MS = 1_000
const VOICE_ACTIVITY_RMS_THRESHOLD = 0.02
const CHART_FRAME_INTERVAL_MS = 33
const BRIDGE_GAP_SEC = 0.22

function bridgeShortGap(points: PitchPoint[], maxGapSec: number): PitchPoint[] {
  if (points.length < 3) {
    return points
  }
  const rightIdx = points.length - 1
  const right = points[rightIdx]
  if (right.y === null) {
    return points
  }

  let leftIdx = rightIdx - 1
  while (leftIdx >= 0 && points[leftIdx].y === null) {
    leftIdx -= 1
  }
  if (leftIdx < 0 || leftIdx === rightIdx - 1) {
    return points
  }
  const left = points[leftIdx]
  if (left.y === null || right.t - left.t > maxGapSec) {
    return points
  }

  for (let i = leftIdx + 1; i < rightIdx; i += 1) {
    if (points[i].y !== null) {
      return points
    }
  }

  const next = points.slice()
  for (let i = leftIdx + 1; i < rightIdx; i += 1) {
    next[i] = {
      ...next[i],
      y: left.y,
      centY: next[i].centY ?? left.centY
    }
  }
  return next
}

export default function App(): JSX.Element {
  const [status, setStatus] = useState('Ready to listen')
  const [connected, setConnected] = useState(false)
  const [listening, setListening] = useState(false)
  const [paused, setPaused] = useState(false)

  const [doHz, setDoHz] = useState(DEFAULT_DO)
  const [tuning, setTuning] = useState<Tuning>('Just Intonation')
  const [keySemitone, setKeySemitone] = useState(0)
  const [instrument, setInstrument] = useState<Instrument>('Guitar')

  const [centCurve, setCentCurve] = useState(false)
  const [metronomeEnabled, setMetronomeEnabled] = useState(false)
  const [bpm, setBpm] = useState(96)

  const [shiftSteps, setShiftSteps] = useState(0)
  const [isRecording, setIsRecording] = useState(false)
  const [hasRecording, setHasRecording] = useState(false)
  const [playing, setPlaying] = useState(false)

  const [shiftPressed, setShiftPressed] = useState(false)
  const [controlPressed, setControlPressed] = useState(false)

  const [points, setPoints] = useState<PitchPoint[]>([])
  const [chartNowSec, setChartNowSec] = useState(0)
  const [currentHz, setCurrentHz] = useState<number | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const workletNodeRef = useRef<AudioWorkletNode | null>(null)
  const muteGainRef = useRef<GainNode | null>(null)
  const workletReadyRef = useRef(false)

  const listeningRef = useRef(false)
  const recordingRef = useRef(false)
  const recordedChunksRef = useRef<Float32Array[]>([])
  const recordedClipRef = useRef<Float32Array | null>(null)
  const sampleRateRef = useRef(44_100)

  const metronomeTimerRef = useRef<number | null>(null)
  const chartRafRef = useRef<number | null>(null)
  const lastChartFrameAtMsRef = useRef(0)
  const chartNowSecRef = useRef(0)
  const sessionStartPerfRef = useRef(0)
  const serverTimeBaseRef = useRef<number | null>(null)
  const lastActivityAtMsRef = useRef(performance.now())

  const activeNotesRef = useRef<Map<string, { osc: OscillatorNode; gain: GainNode }>>(new Map())

  const sendJson = useCallback((payload: object) => {
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload))
    }
  }, [])

  const markUserActivity = useCallback(() => {
    lastActivityAtMsRef.current = performance.now()
  }, [])

  const ensureAudioContext = useCallback(async (): Promise<AudioContext> => {
    if (!audioContextRef.current) {
      const win = window as BrowserWindow
      const AudioContextCtor = win.AudioContext ?? win.webkitAudioContext
      if (!AudioContextCtor) {
        throw new Error('AudioContext is not supported')
      }
      audioContextRef.current = new AudioContextCtor()
      sampleRateRef.current = audioContextRef.current.sampleRate
    }
    if (audioContextRef.current.state !== 'running') {
      await audioContextRef.current.resume()
    }
    return audioContextRef.current
  }, [])

  const primeAudioContext = useCallback(() => {
    void ensureAudioContext().catch(() => {
      setStatus('Audio unavailable in browser')
    })
  }, [ensureAudioContext])

  const resetTrace = useCallback(() => {
    setPoints([])
    chartNowSecRef.current = 0
    setChartNowSec(0)
    serverTimeBaseRef.current = null
    if (listeningRef.current) {
      sessionStartPerfRef.current = performance.now()
    }
  }, [])

  const handleServerEvent = useCallback((event: ServerEvent) => {
    if (event.type === 'status' && typeof event.message === 'string') {
      setStatus(event.message)
      return
    }

    if (event.type === 'error' && typeof event.message === 'string') {
      setStatus(event.message)
      return
    }

    if (event.type === 'calibration_done' && typeof event.doHz === 'number') {
      setDoHz(event.doHz)
      setStatus(`Calibration done: Do=${event.doHz.toFixed(1)} Hz`)
      return
    }

    if (event.type === 'pitch_update' && typeof event.t === 'number') {
      const voiced = event.voiced === true
      const rms = typeof event.rms === 'number' ? event.rms : 0
      const active = isVoiceActivity(voiced, rms, VOICE_ACTIVITY_RMS_THRESHOLD)
      if (active) {
        markUserActivity()
      }
      if (serverTimeBaseRef.current === null) {
        const elapsedSec = Math.max(0, (performance.now() - sessionStartPerfRef.current) / 1_000)
        serverTimeBaseRef.current = event.t - elapsedSec
      }
      const displayT = Math.max(0, event.t - serverTimeBaseRef.current)
      let nextY = typeof event.y === 'number' ? event.y : null
      let nextCentY = typeof event.centY === 'number' ? event.centY : null
      if (nextY === null) {
        nextCentY = null
      }
      const point: PitchPoint = {
        t: displayT,
        y: nextY,
        centY: nextCentY,
        hz: typeof event.hz === 'number' ? event.hz : null
      }
      if (point.hz !== null) {
        setCurrentHz(point.hz)
      }
      chartNowSecRef.current = Math.max(chartNowSecRef.current, point.t)
      setChartNowSec(chartNowSecRef.current)
      setPoints((prev) => {
        const next = bridgeShortGap([...prev, point], BRIDGE_GAP_SEC)
        const minT = point.t - WINDOW_SECONDS
        let firstValid = 0
        while (firstValid < next.length && next[firstValid].t < minT) {
          firstValid += 1
        }
        return next.slice(firstValid)
      })
    }
  }, [markUserActivity])

  const connectSocket = useCallback(async (): Promise<WebSocket> => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      return wsRef.current
    }

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${protocol}://${window.location.host}/ws/realtime`)
    ws.binaryType = 'arraybuffer'

    ws.onopen = () => {
      setConnected(true)
      setStatus('Connected')
    }
    ws.onclose = () => {
      setConnected(false)
      if (listeningRef.current) {
        setStatus('Disconnected')
      }
    }
    ws.onerror = () => {
      setStatus('WebSocket error')
    }
    ws.onmessage = (message) => {
      if (typeof message.data !== 'string') {
        return
      }
      try {
        const payload = JSON.parse(message.data) as ServerEvent
        handleServerEvent(payload)
      } catch {
        setStatus('Invalid server event')
      }
    }

    wsRef.current = ws

    await new Promise<void>((resolve, reject) => {
      const onOpen = () => {
        cleanup()
        resolve()
      }
      const onError = () => {
        cleanup()
        reject(new Error('WebSocket failed to connect'))
      }
      const cleanup = () => {
        ws.removeEventListener('open', onOpen)
        ws.removeEventListener('error', onError)
      }

      if (ws.readyState === WebSocket.OPEN) {
        resolve()
        return
      }
      ws.addEventListener('open', onOpen)
      ws.addEventListener('error', onError)
    })

    return ws
  }, [handleServerEvent])

  const stopMetronome = useCallback(() => {
    if (metronomeTimerRef.current !== null) {
      window.clearInterval(metronomeTimerRef.current)
      metronomeTimerRef.current = null
    }
  }, [])

  const playMetronomeClick = useCallback(async () => {
    const ctx = await ensureAudioContext()
    const osc = ctx.createOscillator()
    const gain = ctx.createGain()
    osc.type = 'square'
    osc.frequency.value = 1600
    gain.gain.setValueAtTime(0.2, ctx.currentTime)
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.03)
    osc.connect(gain)
    gain.connect(ctx.destination)
    osc.start()
    osc.stop(ctx.currentTime + 0.035)
  }, [ensureAudioContext])

  const startMetronome = useCallback(() => {
    stopMetronome()
    const intervalMs = Math.floor(60_000 / clampBpm(bpm))
    metronomeTimerRef.current = window.setInterval(() => {
      void playMetronomeClick()
    }, intervalMs)
  }, [bpm, playMetronomeClick, stopMetronome])

  const stopAllNotes = useCallback(() => {
    const active = activeNotesRef.current
    for (const [key, note] of active.entries()) {
      note.gain.gain.cancelScheduledValues(note.gain.context.currentTime)
      note.gain.gain.setTargetAtTime(0.0001, note.gain.context.currentTime, 0.02)
      note.osc.stop(note.gain.context.currentTime + 0.08)
      active.delete(key)
    }
  }, [])

  const playNote = useCallback(
    async (noteId: string, degree: number, octave: number) => {
      const ctx = await ensureAudioContext()
      if (activeNotesRef.current.has(noteId)) {
        return
      }
      const freq = degreeToFrequency(degree, octave, doHz, tuning, keySemitone)
      if (freq <= 0) {
        return
      }
      const osc = ctx.createOscillator()
      const gain = ctx.createGain()
      osc.type = instrument === 'Guitar' ? 'triangle' : 'sine'
      osc.frequency.value = freq
      const peakGain = instrument === 'Guitar' ? 0.18 : 0.14
      const attack = instrument === 'Guitar' ? 0.02 : 0.015
      gain.gain.setValueAtTime(0.0001, ctx.currentTime)
      gain.gain.exponentialRampToValueAtTime(peakGain, ctx.currentTime + attack)
      osc.connect(gain)
      gain.connect(ctx.destination)
      osc.start()
      activeNotesRef.current.set(noteId, { osc, gain })
    },
    [doHz, ensureAudioContext, instrument, keySemitone, tuning]
  )

  const stopNote = useCallback((noteId: string) => {
    const note = activeNotesRef.current.get(noteId)
    if (!note) {
      return
    }
    const now = note.gain.context.currentTime
    note.gain.gain.cancelScheduledValues(now)
    note.gain.gain.setTargetAtTime(0.0001, now, 0.03)
    note.osc.stop(now + 0.1)
    activeNotesRef.current.delete(noteId)
  }, [])

  const stopCaptureGraph = useCallback(() => {
    workletNodeRef.current?.disconnect()
    sourceNodeRef.current?.disconnect()
    muteGainRef.current?.disconnect()
    workletNodeRef.current = null
    sourceNodeRef.current = null
    muteGainRef.current = null

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop())
      mediaStreamRef.current = null
    }

    recordingRef.current = false
    setIsRecording(false)
  }, [])

  const startListening = useCallback(async () => {
    const ctx = await ensureAudioContext()
    const ws = await connectSocket()

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false
      }
    })

    if (!workletReadyRef.current) {
      await ctx.audioWorklet.addModule('/audio-worklet-processor.js')
      workletReadyRef.current = true
    }

    const source = ctx.createMediaStreamSource(stream)
    const worklet = new AudioWorkletNode(ctx, 'mic-forwarder')
    const muteGain = ctx.createGain()
    muteGain.gain.value = 0

    source.connect(worklet)
    worklet.connect(muteGain)
    muteGain.connect(ctx.destination)

    worklet.port.onmessage = (event: MessageEvent<Float32Array>) => {
      const payload = event.data
      if (!(payload instanceof Float32Array)) {
        return
      }
      const frame = new Float32Array(payload)

      if (recordingRef.current) {
        recordedChunksRef.current.push(frame)
        const maxSamples = Math.floor(sampleRateRef.current * 10)
        let total = recordedChunksRef.current.reduce((acc, chunk) => acc + chunk.length, 0)
        while (total > maxSamples && recordedChunksRef.current.length > 0) {
          const first = recordedChunksRef.current[0]
          const overflow = total - maxSamples
          if (first.length <= overflow) {
            recordedChunksRef.current.shift()
            total -= first.length
          } else {
            recordedChunksRef.current[0] = first.slice(overflow)
            total = maxSamples
          }
        }
      }

      if (listeningRef.current && ws.readyState === WebSocket.OPEN) {
        ws.send(frame.buffer)
      }
    }

    mediaStreamRef.current = stream
    sourceNodeRef.current = source
    workletNodeRef.current = worklet
    muteGainRef.current = muteGain

    sampleRateRef.current = ctx.sampleRate
    const resumeAtSec = paused ? chartNowSecRef.current : 0
    sessionStartPerfRef.current = performance.now() - resumeAtSec * 1_000
    serverTimeBaseRef.current = null
    lastChartFrameAtMsRef.current = 0
    chartNowSecRef.current = resumeAtSec
    setChartNowSec(resumeAtSec)
    lastActivityAtMsRef.current = performance.now()
    listeningRef.current = true
    setListening(true)
    setPaused(false)

    sendJson({
      type: 'init',
      sampleRate: sampleRateRef.current,
      tuning,
      keySemitone,
      doHz
    })

    setStatus('Listening...')
    if (metronomeEnabled) {
      startMetronome()
    }
  }, [
    connectSocket,
    doHz,
    ensureAudioContext,
    keySemitone,
    metronomeEnabled,
    paused,
    sendJson,
    startMetronome,
    tuning
  ])

  const pauseListening = useCallback(
    (reason = 'Paused') => {
      stopCaptureGraph()
      stopMetronome()
      listeningRef.current = false
      setListening(false)
      setPaused(true)
      setStatus(reason)
      if (chartRafRef.current !== null) {
        window.cancelAnimationFrame(chartRafRef.current)
        chartRafRef.current = null
      }
    },
    [stopCaptureGraph, stopMetronome]
  )

  const stopListening = useCallback(() => {
    stopCaptureGraph()
    stopMetronome()
    if (chartRafRef.current !== null) {
      window.cancelAnimationFrame(chartRafRef.current)
      chartRafRef.current = null
    }
    listeningRef.current = false
    setListening(false)
    stopAllNotes()
    setPaused(false)
    setCurrentHz(null)
    serverTimeBaseRef.current = null
    chartNowSecRef.current = 0
    setChartNowSec(0)
    setPoints([])
    setStatus('Ready to listen')
    sendJson({ type: 'calibrate_cancel' })
  }, [sendJson, stopAllNotes, stopCaptureGraph, stopMetronome])

  const startRecording = useCallback(() => {
    if (!listeningRef.current) {
      setStatus('Press Start before recording')
      return
    }
    recordedChunksRef.current = []
    recordingRef.current = true
    setIsRecording(true)
    setStatus('Recording...')
  }, [])

  const stopRecording = useCallback(() => {
    if (!recordingRef.current) {
      return
    }
    recordingRef.current = false
    setIsRecording(false)
    const chunks = recordedChunksRef.current
    if (chunks.length === 0) {
      setStatus('No recording captured')
      return
    }
    const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0)
    const merged = new Float32Array(totalLength)
    let offset = 0
    for (const chunk of chunks) {
      merged.set(chunk, offset)
      offset += chunk.length
    }
    recordedClipRef.current = merged
    setHasRecording(true)
    setStatus('Recording saved')
  }, [])

  const playShifted = useCallback(async () => {
    if (playing) {
      return
    }
    if (!recordedClipRef.current) {
      setStatus('Record something first')
      return
    }

    setPlaying(true)
    setStatus('Processing...')
    try {
      const blob = encodeWav(recordedClipRef.current, sampleRateRef.current)
      const form = new FormData()
      form.append('audio', blob, 'recording.wav')
      form.append('steps', String(shiftSteps))
      form.append('do_hz', String(doHz))
      form.append('tuning', tuning)
      form.append('key_semitone', String(keySemitone))

      const response = await fetch('/api/voice/shift', {
        method: 'POST',
        body: form
      })
      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || 'Shift request failed')
      }
      const shiftedBlob = await response.blob()
      const arrayBuffer = await shiftedBlob.arrayBuffer()
      const ctx = await ensureAudioContext()
      const decoded = await ctx.decodeAudioData(arrayBuffer)
      const source = ctx.createBufferSource()
      source.buffer = decoded
      source.connect(ctx.destination)
      source.start()
      setStatus('Playing...')
      source.onended = () => {
        setStatus('Listening...')
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Playback failed'
      setStatus(message)
    } finally {
      setPlaying(false)
    }
  }, [doHz, ensureAudioContext, keySemitone, playing, shiftSteps, tuning])

  const calibrate = useCallback(async () => {
    try {
      if (!listeningRef.current) {
        await startListening()
      }
      sendJson({ type: 'calibrate_start', seconds: 4.0 })
      setStatus('Calibrating... sing Do now')
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Calibration failed'
      setStatus(message)
    }
  }, [sendJson, startListening])

  const handleTouchDigitDown = useCallback(
    (digit: number, pointerId: number) => {
      markUserActivity()
      const octave = resolveOctave(shiftPressed, controlPressed)
      const noteId = `touch-${pointerId}`
      void playNote(noteId, digit, octave)
    },
    [controlPressed, markUserActivity, playNote, shiftPressed]
  )

  const handleTouchDigitUp = useCallback((_digit: number, pointerId: number) => {
    markUserActivity()
    stopNote(`touch-${pointerId}`)
  }, [markUserActivity, stopNote])

  useEffect(() => {
    const unlock = () => {
      primeAudioContext()
    }
    window.addEventListener('pointerdown', unlock, { passive: true, once: true })
    window.addEventListener('keydown', unlock, { once: true })
    return () => {
      window.removeEventListener('pointerdown', unlock)
      window.removeEventListener('keydown', unlock)
    }
  }, [primeAudioContext])

  useEffect(() => {
    const codeDigits: Record<string, number> = {
      Digit1: 1,
      Digit2: 2,
      Digit3: 3,
      Digit4: 4,
      Digit5: 5,
      Digit6: 6,
      Digit7: 7,
      Numpad1: 1,
      Numpad2: 2,
      Numpad3: 3,
      Numpad4: 4,
      Numpad5: 5,
      Numpad6: 6,
      Numpad7: 7
    }
    const getNoteId = (event: KeyboardEvent): string => {
      const base = event.code || `${event.key}-${event.location}`
      return `kbd-${base}`
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.repeat) {
        return
      }

      const shiftedSymbols: Record<string, number> = {
        '!': 1,
        '@': 2,
        '#': 3,
        '$': 4,
        '%': 5,
        '^': 6,
        '&': 7
      }

      let degree: number | null = null
      if (/^[1-7]$/.test(event.key)) {
        degree = Number(event.key)
      } else if (event.key in shiftedSymbols) {
        degree = shiftedSymbols[event.key]
      } else if (event.code in codeDigits) {
        degree = codeDigits[event.code]
      } else {
        const legacy = (event as KeyboardEvent & { keyCode?: number }).keyCode
        if (typeof legacy === 'number') {
          if (legacy >= 49 && legacy <= 55) {
            degree = legacy - 48
          } else if (legacy >= 97 && legacy <= 103) {
            degree = legacy - 96
          }
        }
      }

      if (degree === null) {
        return
      }

      markUserActivity()
      primeAudioContext()
      const octave = event.shiftKey ? 1 : event.ctrlKey ? -1 : 0
      const noteId = getNoteId(event)
      if (activeNotesRef.current.has(noteId)) {
        return
      }
      void playNote(noteId, degree, octave)
      event.preventDefault()
    }

    const onKeyUp = (event: KeyboardEvent) => {
      markUserActivity()
      stopNote(getNoteId(event))
    }

    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup', onKeyUp)
    }
  }, [markUserActivity, playNote, primeAudioContext, stopNote])

  useEffect(() => {
    sendJson({
      type: 'set_config',
      tuning,
      keySemitone,
      doHz
    })
  }, [doHz, keySemitone, sendJson, tuning])

  useEffect(() => {
    resetTrace()
  }, [keySemitone, resetTrace])

  useEffect(() => {
    if (metronomeEnabled && listening) {
      startMetronome()
    } else {
      stopMetronome()
    }
  }, [bpm, listening, metronomeEnabled, startMetronome, stopMetronome])

  useEffect(() => {
    if (!listening || paused) {
      if (chartRafRef.current !== null) {
        window.cancelAnimationFrame(chartRafRef.current)
        chartRafRef.current = null
      }
      return
    }
    const tick = (nowMs: number) => {
      if (!listeningRef.current) {
        chartRafRef.current = null
        return
      }
      if (
        lastChartFrameAtMsRef.current === 0 ||
        nowMs - lastChartFrameAtMsRef.current >= CHART_FRAME_INTERVAL_MS
      ) {
        const elapsedSec = Math.max(0, (performance.now() - sessionStartPerfRef.current) / 1_000)
        chartNowSecRef.current = Math.max(chartNowSecRef.current, elapsedSec)
        setChartNowSec(chartNowSecRef.current)
        lastChartFrameAtMsRef.current = nowMs
      }
      chartRafRef.current = window.requestAnimationFrame(tick)
    }
    chartRafRef.current = window.requestAnimationFrame(tick)
    return () => {
      if (chartRafRef.current !== null) {
        window.cancelAnimationFrame(chartRafRef.current)
        chartRafRef.current = null
      }
    }
  }, [listening, paused])

  useEffect(() => {
    if (!listening || paused) {
      return
    }
    const timer = window.setInterval(() => {
      const now = performance.now()
      if (shouldAutoPause(lastActivityAtMsRef.current, now, IDLE_PAUSE_MS)) {
        pauseListening('Auto-paused after 60s inactivity')
      }
    }, IDLE_CHECK_MS)
    return () => window.clearInterval(timer)
  }, [listening, pauseListening, paused])

  useEffect(() => {
    const onUserActivity = () => {
      if (listeningRef.current) {
        markUserActivity()
      }
    }
    window.addEventListener('pointerdown', onUserActivity, { passive: true })
    window.addEventListener('keydown', onUserActivity)
    window.addEventListener('change', onUserActivity)
    return () => {
      window.removeEventListener('pointerdown', onUserActivity)
      window.removeEventListener('keydown', onUserActivity)
      window.removeEventListener('change', onUserActivity)
    }
  }, [markUserActivity])

  const togglePauseFromChart = useCallback(() => {
    markUserActivity()
    if (listening) {
      pauseListening()
    } else if (paused) {
      void startListening()
    }
  }, [listening, markUserActivity, pauseListening, paused, startListening])

  useEffect(() => {
    return () => {
      stopListening()
      stopMetronome()
      stopAllNotes()
      wsRef.current?.close()
      void audioContextRef.current?.close()
    }
  }, [stopAllNotes, stopListening, stopMetronome])

  return (
    <div className="app-shell">
      <header className="top-controls">
        <button type="button" className="btn" onClick={() => void calibrate()}>
          Calibrate (4s)
        </button>

        <div className="field-group radios">
          <button
            type="button"
            className={`btn ${Math.abs(doHz - DEFAULT_DO) <= 0.1 ? 'active' : ''}`}
            onClick={() => setDoHz(DEFAULT_DO)}
          >
            Male
          </button>
          <button
            type="button"
            className={`btn ${Math.abs(doHz - FEMALE_DO) <= 0.1 ? 'active' : ''}`}
            onClick={() => setDoHz(FEMALE_DO)}
          >
            Female
          </button>
        </div>

        <button type="button" className="btn start" disabled={listening} onClick={() => void startListening()}>
          Start
        </button>
        <button
          type="button"
          className="btn pause"
          disabled={!listening && !paused}
          onClick={() => {
            if (listening) {
              pauseListening()
            } else if (paused) {
              void startListening()
            }
          }}
        >
          {paused ? 'Resume' : 'Pause'}
        </button>
        <button type="button" className="btn stop" disabled={!listening && !paused} onClick={stopListening}>
          Stop
        </button>

        <label className="field-group field-tuning">
          <span>Tuning</span>
          <select value={tuning} onChange={(e) => setTuning(e.target.value as Tuning)}>
            {TUNINGS.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>

        <label className="field-group field-instrument">
          <span>Instrument</span>
          <select value={instrument} onChange={(e) => setInstrument(e.target.value as Instrument)}>
            <option value="Piano">Piano</option>
            <option value="Guitar">Guitar</option>
          </select>
        </label>

      </header>

      <section className="sub-controls">
        <label className="checkbox">
          <input type="checkbox" checked={centCurve} onChange={(e) => setCentCurve(e.target.checked)} />
          Cent Curve
        </label>

        <label className="checkbox">
          <input
            type="checkbox"
            checked={metronomeEnabled}
            onChange={(e) => setMetronomeEnabled(e.target.checked)}
          />
          Metronome
        </label>

        <label className="field-group compact">
          <span>BPM</span>
          <input
            type="number"
            min={50}
            max={160}
            value={bpm}
            onChange={(e) => setBpm(clampBpm(Number(e.target.value)))}
          />
        </label>

        <button
          type="button"
          className={`btn record ${isRecording ? 'active' : ''}`}
          disabled={!listening}
          onPointerDown={startRecording}
          onPointerUp={stopRecording}
          onPointerCancel={stopRecording}
          onPointerLeave={(e) => {
            if (e.buttons === 0) {
              stopRecording()
            }
          }}
        >
          Recording
        </button>

        <label className="field-group compact">
          <span>Shift</span>
          <input
            type="number"
            min={-5}
            max={5}
            value={shiftSteps}
            onChange={(e) => setShiftSteps(Math.max(-5, Math.min(5, Number(e.target.value))))}
          />
        </label>

        <button type="button" className="btn" disabled={!hasRecording || playing} onClick={() => void playShifted()}>
          Play
        </button>
      </section>

      <section className="status-row">
        <div className="status">{status}</div>
        <div className="status-meta">
          <span className={`meta-pill ${connected ? 'online' : 'offline'}`} title="WebSocket connection">
            {connected ? 'WS Connected' : 'WS Offline'}
          </span>
          <span className="meta-pill">{currentHz === null ? 'Hz --' : `Hz ${currentHz.toFixed(1)}`}</span>
          <label className="field-group compact field-key">
            <span>Key</span>
            <select
              value={String(keySemitone)}
              onChange={(e) => {
                setKeySemitone(Number(e.target.value))
                e.currentTarget.blur()
              }}
              aria-label="Key"
            >
              {KEY_OPTIONS.map(([label, semitone]) => (
                <option key={label} value={semitone}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <label className="field-group compact field-do">
            <span>Do</span>
            <input
              type="number"
              min={50}
              max={800}
              step={0.1}
              value={Number.isFinite(doHz) ? doHz : DEFAULT_DO}
              onChange={(e) => setDoHz(Number(e.target.value))}
            />
          </label>
        </div>
      </section>

      <section className="chart-section">
        <PitchChart
          points={points}
          showCentCurve={centCurve}
          nowSec={chartNowSec}
          onTogglePause={togglePauseFromChart}
        />
      </section>

      <section className="touchpad-section">
        <TouchPad
          shiftPressed={shiftPressed}
          controlPressed={controlPressed}
          onShiftDown={() => setShiftPressed(true)}
          onShiftUp={() => setShiftPressed(false)}
          onControlDown={() => setControlPressed(true)}
          onControlUp={() => setControlPressed(false)}
          onDigitDown={handleTouchDigitDown}
          onDigitUp={handleTouchDigitUp}
        />
      </section>
    </div>
  )
}
