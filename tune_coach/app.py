from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from collections import deque

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from tune_coach.audio import AudioInput
from tune_coach.calibration import CalibrationError, calibrate_do_from_stream
from tune_coach.jianpu import JianpuAxis, JianpuQuantizer, TuningSystem
from tune_coach.metronome import Metronome
from tune_coach.pitch import PitchTracker, PitchTrackerConfig
from tune_coach.synth import Instrument, NoteSynth


@dataclass(frozen=True)
class UiConfig:
    window_seconds: float = 24.0
    tick_seconds: float = 1.0
    left_margin_seconds: float = 0.4
    # Display delay to allow extra smoothing before rendering.
    delay_seconds: float = 0.25
    # Median window used for note stability detection.
    smooth_window: int = 11
    # Minimum count of the same note to accept it as stable.
    stable_min_count: int = 7
    # Extra confirmations required to switch to a new note.
    switch_confirm: int = 3
    # Short silence tolerance (seconds) before cutting the line.
    silence_timeout: float = 0.2


class MainWindow(QtWidgets.QMainWindow):
    _SHIFT_DIGIT_KEYS = {
        QtCore.Qt.Key.Key_Exclam: 1,
        QtCore.Qt.Key.Key_At: 2,
        QtCore.Qt.Key.Key_NumberSign: 3,
        QtCore.Qt.Key.Key_Dollar: 4,
        QtCore.Qt.Key.Key_Percent: 5,
        QtCore.Qt.Key.Key_AsciiCircum: 6,
        QtCore.Qt.Key.Key_Ampersand: 7,
    }
    _NSEVENT_CTRL_MASK = 1 << 18
    _NSEVENT_CMD_MASK = 1 << 20
    _DEFAULT_DO_HZ = 130.8

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Tune Coach")

        self._ui = UiConfig()
        self._audio = AudioInput()
        self._pitch = PitchTracker(PitchTrackerConfig(sample_rate=self._audio.sample_rate))
        self._quantizer: JianpuQuantizer | None = None
        self._metronome = Metronome(sample_rate=self._audio.sample_rate)
        self._synth = NoteSynth(sample_rate=self._audio.sample_rate)
        self._held_keys: dict[int, str] = {}
        self._shift_digit_keys = dict(self._SHIFT_DIGIT_KEYS)
        for name, degree in (("Key_Circumflex", 6), ("Key_Dead_Circumflex", 6)):
            key = getattr(QtCore.Qt.Key, name, None)
            if key is not None:
                self._shift_digit_keys[key] = degree

        self._start_time: float | None = None
        self._times: list[float] = []
        self._ys: list[float] = []
        # Buffer for delayed rendering (time, value).
        self._buffer = deque()
        # Short history for smoothing and stable note decision.
        self._y_smooth = deque(maxlen=self._ui.smooth_window)
        self._y_history = deque(maxlen=self._ui.smooth_window)
        # Current stable note and candidate note for switching.
        self._current_y: int | None = None
        self._candidate_y: int | None = None
        self._candidate_count = 0
        # Track last voiced time to detect silence gaps.
        self._last_voiced_time: float | None = None
        self._last_output_nan = False

        self._build_ui()

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)  # ~60 FPS UI update
        self._timer.timeout.connect(self._on_tick)
        self._reset_second_lines()
        QtWidgets.QApplication.instance().installEventFilter(self)
        self._apply_do_hz(self._DEFAULT_DO_HZ)
        self.plot.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.plot.setFocus()

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(root)

        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)

        self.btn_calibrate = QtWidgets.QPushButton("Calibrate (4s)")
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)

        controls.addWidget(self.btn_calibrate)
        controls.addSpacing(12)

        self.tuning_combo = QtWidgets.QComboBox()
        self.tuning_combo.addItems(
            [TuningSystem.EQUAL_TEMPERAMENT.value, TuningSystem.JUST_INTONATION.value]
        )
        self.tuning_combo.setCurrentText(TuningSystem.JUST_INTONATION.value)
        tuning_label = QtWidgets.QLabel("Tuning:")
        controls.addWidget(tuning_label)
        controls.addWidget(self.tuning_combo)

        self.instrument_combo = QtWidgets.QComboBox()
        self.instrument_combo.addItems([Instrument.PIANO.value, Instrument.GUITAR.value])
        self.instrument_combo.setCurrentText(Instrument.GUITAR.value)
        instrument_label = QtWidgets.QLabel("Instrument:")
        controls.addWidget(instrument_label)
        controls.addWidget(self.instrument_combo)

        controls.addStretch(1)
        controls.addWidget(self.btn_start)
        controls.addWidget(self.btn_stop)

        self.btn_calibrate.clicked.connect(self._on_calibrate)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.tuning_combo.currentTextChanged.connect(self._on_tuning_change)
        self.instrument_combo.currentTextChanged.connect(self._on_instrument_change)

        met_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(met_layout)
        self.chk_metronome = QtWidgets.QCheckBox("Metronome")
        self.chk_metronome.setChecked(False)
        self.spin_bpm = QtWidgets.QSpinBox()
        self.spin_bpm.setRange(50, 160)
        self.spin_bpm.setValue(96)
        self.spin_bpm.setSuffix(" BPM")
        self.spin_bpm.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        met_layout.addWidget(self.chk_metronome)
        met_layout.addWidget(self.spin_bpm)
        met_layout.addStretch(1)

        self.chk_metronome.toggled.connect(self._on_metronome_toggle)
        self.spin_bpm.valueChanged.connect(self._on_bpm_change)

        status_layout = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel("Calibrate or enter Do (Hz).")
        self.status.setStyleSheet("color: #f5f5f5; font-weight: bold;")
        self.do_label = QtWidgets.QLabel("Do:")
        self.do_input = QtWidgets.QDoubleSpinBox()
        self.do_input.setRange(0.0, 2000.0)
        self.do_input.setDecimals(1)
        self.do_input.setSingleStep(1.0)
        self.do_input.setSuffix(" Hz")
        self.do_input.setSpecialValueText("-- Hz")
        self.do_input.setValue(0.0)
        self.do_input.setKeyboardTracking(False)
        self.do_input.setFixedWidth(140)
        self.do_input.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        self.do_input.editingFinished.connect(self._on_do_input)
        status_layout.addWidget(self.status)
        status_layout.addStretch(1)
        status_layout.addWidget(self.do_label)
        status_layout.addWidget(self.do_input)
        layout.addLayout(status_layout)

        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        self.plot.showGrid(x=False, y=True, alpha=0.15)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setMenuEnabled(False)
        self.plot.setLabel("bottom", "Time", units="s")

        self._axis = JianpuAxis(octave_gap=0)
        plot_item = self.plot.getPlotItem()
        plot_item.hideAxis("left")
        plot_item.showAxis("right")
        right_axis = plot_item.getAxis("right")
        right_axis.setTicks([self._axis.ticks()])
        right_axis.setStyle(tickTextOffset=10)
        right_axis.setTickFont(QtGui.QFont("Helvetica", 12))
        right_axis.setWidth(60)
        self.plot.setYRange(-0.5, self._axis.max_y + 0.5)

        layout.addWidget(self.plot, 1)

        # Line breaks on silence via NaN (connect='finite'), points rendered separately.
        self._curve = self.plot.plot([], [], pen=pg.mkPen(color=(30, 90, 160), width=2), connect="finite")
        self._scatter = pg.ScatterPlotItem(
            size=6,
            brush=pg.mkBrush(30, 90, 160),
            pen=pg.mkPen(color=(30, 90, 160), width=1),
        )
        self.plot.addItem(self._scatter)

        self._second_lines: list[pg.InfiniteLine] = []
        self.setCentralWidget(root)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            self._timer.stop()
            self._audio.stop()
            self._metronome.stop()
            self._synth.stop()
        finally:
            super().closeEvent(event)

    @QtCore.Slot()
    def _on_calibrate(self) -> None:
        if self._timer.isActive():
            self._stop_listening()
        self._set_status("Calibrating... sing Do now.", "info")
        QtWidgets.QApplication.processEvents()
        try:
            do_hz = calibrate_do_from_stream(
                audio=self._audio,
                pitch=self._pitch,
                seconds=4.0,
            )
        except CalibrationError:
            self._set_status("Calibration failed", "error")
            self._quantizer = None
            self._set_do_input(0.0)
            return
        except Exception as exc:  # noqa: BLE001 - user-facing
            self._set_status(f"Calibration failed: {exc}", "error")
            self._quantizer = None
            self._set_do_input(0.0)
            return

        self._apply_do_hz(do_hz)

    @QtCore.Slot()
    def _on_start(self) -> None:
        if self._quantizer is None:
            self._set_status("Calibrate or enter Do (Hz).", "error")
            return

        self._reset_trace()
        self._start_time = time.monotonic()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_calibrate.setEnabled(False)
        self._set_status("Listening...", "info")

        self._audio.start()
        self._timer.start()

        self._reset_second_lines()

    @QtCore.Slot()
    def _on_stop(self) -> None:
        self._stop_listening()

    @QtCore.Slot(bool)
    def _on_metronome_toggle(self, enabled: bool) -> None:
        if enabled:
            self._metronome.set_bpm(int(self.spin_bpm.value()))
            self._metronome.start()
        else:
            self._metronome.stop()

    @QtCore.Slot(int)
    def _on_bpm_change(self, bpm: int) -> None:
        self._metronome.set_bpm(int(bpm))

    def _on_tuning_change(self, text: str) -> None:
        tuning = self._parse_tuning(text)
        if self._quantizer is not None:
            self._quantizer = JianpuQuantizer(
                do_hz=self._quantizer.do_hz,
                octave_gap=self._quantizer.octave_gap,
                tuning=tuning,
            )
            self._reset_trace()

    def _on_instrument_change(self, text: str) -> None:
        self._synth.set_instrument(text)

    def _on_do_input(self) -> None:
        do_hz = float(self.do_input.value())
        if do_hz <= 0:
            return
        self._apply_do_hz(do_hz)

    def _apply_do_hz(self, do_hz: float) -> None:
        if self._timer.isActive():
            self._stop_listening()
        tuning = self._parse_tuning(self.tuning_combo.currentText())
        self._quantizer = JianpuQuantizer(do_hz=do_hz, octave_gap=0, tuning=tuning)
        self._set_do_input(do_hz)
        self._reset_trace()
        self._set_status("Ready to listen", "ready")
        self.btn_start.setEnabled(True)

    def _set_do_input(self, do_hz: float) -> None:
        self.do_input.blockSignals(True)
        self.do_input.setValue(float(do_hz))
        self.do_input.blockSignals(False)

    def _parse_tuning(self, text: str) -> TuningSystem:
        try:
            return TuningSystem(text)
        except ValueError:
            return TuningSystem.EQUAL_TEMPERAMENT

    def _set_status(self, text: str, kind: str) -> None:
        if kind == "ready":
            self.status.setStyleSheet("color: #ff5c5c; font-weight: bold;")
        elif kind == "error":
            self.status.setStyleSheet("color: #ff5c5c; font-weight: bold;")
        elif kind == "info":
            self.status.setStyleSheet("color: #7fd1ff; font-weight: bold;")
        else:
            self.status.setStyleSheet("color: #f5f5f5; font-weight: bold;")
        self.status.setText(text)

    def _reset_trace(self) -> None:
        self._times.clear()
        self._ys.clear()
        self._buffer.clear()
        self._y_smooth.clear()
        self._y_history.clear()
        self._current_y = None
        self._candidate_y = None
        self._candidate_count = 0
        self._last_voiced_time = None
        self._last_output_nan = False

    def _stop_listening(self) -> None:
        self._timer.stop()
        self._audio.stop()
        self.btn_start.setEnabled(self._quantizer is not None)
        self.btn_stop.setEnabled(False)
        self.btn_calibrate.setEnabled(True)
        if self._quantizer is None:
            self._set_status("Calibrate or enter Do (Hz).", "error")
        else:
            self._set_status("Ready to listen", "ready")

    def _reset_second_lines(self) -> None:
        for line in self._second_lines:
            self.plot.removeItem(line)
        self._second_lines.clear()

        for i in range(int(self._ui.window_seconds) + 1):
            x = -self._ui.window_seconds + i * self._ui.tick_seconds
            line = pg.InfiniteLine(
                pos=x,
                angle=90,
                pen=pg.mkPen(color=(0, 0, 0, 60), width=1, style=QtCore.Qt.PenStyle.DashLine),
            )
            self.plot.addItem(line)
            self._second_lines.append(line)

        self.plot.setXRange(-self._ui.window_seconds - self._ui.left_margin_seconds, 0.0)

    def _on_tick(self) -> None:
        if self._start_time is None or self._quantizer is None:
            return

        now = time.monotonic()
        t = now - self._start_time

        frame = self._audio.read_latest()
        if frame is not None:
            pitch_hz = self._pitch.process(frame)
            if pitch_hz is not None:
                y = self._quantizer.quantize_to_y(pitch_hz)
                if y is not None:
                    self._y_smooth.append(y)
                    if len(self._y_smooth) >= 3:
                        y_sm = int(np.median(np.array(self._y_smooth, dtype=np.float32)))
                    else:
                        y_sm = y
                    self._y_history.append(y_sm)
                    self._last_voiced_time = t
                else:
                    self._y_smooth.clear()
            else:
                self._y_smooth.clear()

        # Decide output note based on stable trend within the recent window.
        output_y: float | None = None
        if self._last_voiced_time is not None and (t - self._last_voiced_time) <= self._ui.silence_timeout:
            if len(self._y_history) >= self._ui.stable_min_count:
                vals, counts = np.unique(np.array(self._y_history, dtype=np.int32), return_counts=True)
                stable_y = int(vals[np.argmax(counts)])
                stable_count = int(np.max(counts))
                if stable_count >= self._ui.stable_min_count:
                    if self._current_y is None:
                        self._current_y = stable_y
                    elif stable_y != self._current_y:
                        if self._candidate_y == stable_y:
                            self._candidate_count += 1
                        else:
                            self._candidate_y = stable_y
                            self._candidate_count = 1
                        if self._candidate_count >= self._ui.switch_confirm:
                            self._current_y = stable_y
                            self._candidate_y = None
                            self._candidate_count = 0
                    else:
                        self._candidate_y = None
                        self._candidate_count = 0
            output_y = float(self._current_y) if self._current_y is not None else None
        else:
            self._y_history.clear()
            self._current_y = None
            self._candidate_y = None
            self._candidate_count = 0
            output_y = None

        # Buffer output for delayed display and explicit silence gaps.
        if output_y is None:
            if not self._last_output_nan:
                self._buffer.append((t, float("nan")))
                self._last_output_nan = True
        else:
            self._buffer.append((t, output_y))
            self._last_output_nan = False

        # release buffered points with a small delay for extra smoothness
        release_before = t - self._ui.delay_seconds
        while self._buffer and self._buffer[0][0] <= release_before:
            bt, by = self._buffer.popleft()
            self._times.append(bt)
            self._ys.append(by)

        # keep only last window
        min_t = t - self._ui.window_seconds
        while self._times and self._times[0] < min_t:
            self._times.pop(0)
            self._ys.pop(0)

        if self._times:
            x = np.array(self._times, dtype=np.float32) - t  # shift so "now" is x=0
            y = np.array(self._ys, dtype=np.float32)
            self._curve.setData(x, y)
            mask = np.isfinite(y)
            if mask.any():
                self._scatter.setData(x[mask], y[mask])
            else:
                self._scatter.setData([], [])
        else:
            self._curve.setData([], [])
            self._scatter.setData([], [])

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        if event.type() == QtCore.QEvent.Type.ApplicationDeactivate:
            self._release_all_notes()
        elif event.type() == QtCore.QEvent.Type.WindowDeactivate:
            self._release_all_notes()
        if event.type() == QtCore.QEvent.Type.KeyPress:
            if self._handle_key_press(event):
                return True
        elif event.type() == QtCore.QEvent.Type.KeyRelease:
            if self._handle_key_release(event):
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if self._handle_key_press(event):
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self._handle_key_release(event):
            return
        super().keyReleaseEvent(event)

    def _handle_key_press(self, event) -> bool:
        if event.isAutoRepeat() or self._quantizer is None:
            return False
        focus = QtWidgets.QApplication.focusWidget()
        if isinstance(focus, (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox)):
            return False
        degree, octave = self._parse_key_event(event)
        if degree is None:
            return False
        key_id = self._key_id(event)
        if key_id in self._held_keys:
            return True
        freq = self._quantizer.degree_hz(degree, octave=octave)
        if freq is None:
            return False
        self._synth.start()
        note_key = f"{degree}:{octave}"
        self._held_keys[key_id] = note_key
        self._synth.note_on(note_key, freq)
        return True

    def _handle_key_release(self, event) -> bool:
        if event.isAutoRepeat():
            return False
        key_id = self._key_id(event)
        note_key = self._held_keys.pop(key_id, None)
        if note_key is None:
            alt_id = ("qt", int(event.key()))
            note_key = self._held_keys.pop(alt_id, None)
        if note_key is None:
            return False
        self._synth.note_off(note_key)
        return True

    def _parse_key_event(self, event) -> tuple[int | None, int]:
        text = event.text()
        symbol_map = {"!": 1, "@": 2, "#": 3, "$": 4, "%": 5, "^": 6, "&": 7}
        if text in symbol_map:
            degree = symbol_map[text]
        elif text and text.isdigit():
            degree = int(text)
            if degree < 1 or degree > 7:
                return None, 0
        else:
            key = event.key()
            if QtCore.Qt.Key.Key_1 <= key <= QtCore.Qt.Key.Key_7:
                degree = int(key - QtCore.Qt.Key.Key_0)
            elif key in self._shift_digit_keys:
                degree = int(self._shift_digit_keys[key])
            else:
                native = int(event.nativeVirtualKey())
                mac_map = {18: 1, 19: 2, 20: 3, 21: 4, 23: 5, 22: 6, 26: 7}
                if sys.platform == "darwin" and native in mac_map:
                    degree = mac_map[native]
                else:
                    return None, 0
        modifiers = event.modifiers()
        if modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
            octave = 1
        elif self._control_modifier_active(event):
            octave = -1
        else:
            octave = 0
        return degree, octave

    def _key_id(self, event) -> tuple[str, int]:
        native = int(event.nativeVirtualKey())
        if native:
            return ("native", native)
        return ("qt", int(event.key()))

    def _release_all_notes(self) -> None:
        if not self._held_keys:
            return
        for note_key in list(self._held_keys.values()):
            self._synth.note_off(note_key)
        self._held_keys.clear()


    def _control_modifier_active(self, event) -> bool:
        modifiers = event.modifiers()
        if (
            modifiers & QtCore.Qt.KeyboardModifier.ControlModifier
            and not (modifiers & QtCore.Qt.KeyboardModifier.MetaModifier)
        ):
            return True
        if sys.platform == "darwin":
            native = int(event.nativeModifiers())
            if (native & self._NSEVENT_CTRL_MASK) and not (native & self._NSEVENT_CMD_MASK):
                return True
        return False


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1800, 650)
    win.show()
    sys.exit(app.exec())
