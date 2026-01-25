from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from collections import deque

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from tune_coach.audio import AudioInput
from tune_coach.calibration import calibrate_do_from_stream
from tune_coach.jianpu import JianpuAxis, JianpuQuantizer
from tune_coach.metronome import Metronome
from tune_coach.pitch import PitchTracker, PitchTrackerConfig


@dataclass(frozen=True)
class UiConfig:
    window_seconds: float = 12.0
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
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Tune Coach")

        self._ui = UiConfig()
        self._audio = AudioInput()
        self._pitch = PitchTracker(PitchTrackerConfig(sample_rate=self._audio.sample_rate))
        self._quantizer: JianpuQuantizer | None = None
        self._metronome = Metronome(sample_rate=self._audio.sample_rate)

        self._start_time: float | None = None
        self._times: list[float] = []
        self._ys: list[float] = []
        # Buffer for delayed rendering (time, value).
        self._buffer = deque()
        self._last_voiced = False
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

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(root)

        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)

        self.btn_calibrate = QtWidgets.QPushButton("Calibrate (4s)")
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        controls.addWidget(self.btn_calibrate)
        controls.addStretch(1)
        controls.addWidget(self.btn_start)
        controls.addWidget(self.btn_stop)

        self.btn_calibrate.clicked.connect(self._on_calibrate)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)

        met_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(met_layout)
        self.chk_metronome = QtWidgets.QCheckBox("Metronome")
        self.chk_metronome.setChecked(False)
        self.spin_bpm = QtWidgets.QSpinBox()
        self.spin_bpm.setRange(50, 160)
        self.spin_bpm.setValue(96)
        self.spin_bpm.setSuffix(" BPM")
        met_layout.addWidget(self.chk_metronome)
        met_layout.addWidget(self.spin_bpm)
        met_layout.addStretch(1)

        self.chk_metronome.toggled.connect(self._on_metronome_toggle)
        self.spin_bpm.valueChanged.connect(self._on_bpm_change)

        status_layout = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel("Calibrate first: sing 1-2-3 within 4 seconds.")
        self.do_label = QtWidgets.QLabel("Do: -- Hz")
        status_layout.addWidget(self.status)
        status_layout.addStretch(1)
        status_layout.addWidget(self.do_label)
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
        finally:
            super().closeEvent(event)

    @QtCore.Slot()
    def _on_calibrate(self) -> None:
        self.status.setText("Calibrating... sing 1-2-3 now.")
        QtWidgets.QApplication.processEvents()
        try:
            do_hz = calibrate_do_from_stream(
                audio=self._audio,
                pitch=self._pitch,
                seconds=4.0,
            )
        except Exception as exc:  # noqa: BLE001 - user-facing
            self.status.setText(f"Calibration failed: {exc}")
            self._quantizer = None
            return

        self._quantizer = JianpuQuantizer(do_hz=do_hz, octave_gap=0)
        self.status.setText("Calibrated successfully. Press Start.")
        self.do_label.setText(f"Do: {do_hz:.1f} Hz")

    @QtCore.Slot()
    def _on_start(self) -> None:
        if self._quantizer is None:
            self.status.setText("Please calibrate first (Calibrate).")
            return

        self._times.clear()
        self._ys.clear()
        self._buffer.clear()
        self._last_voiced = False
        self._y_smooth.clear()
        self._y_history.clear()
        self._current_y = None
        self._candidate_y = None
        self._candidate_count = 0
        self._last_voiced_time = None
        self._last_output_nan = False
        self._start_time = time.monotonic()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_calibrate.setEnabled(False)
        self.status.setText("Listening...")

        self._audio.start()
        self._timer.start()

        self._reset_second_lines()

    @QtCore.Slot()
    def _on_stop(self) -> None:
        self._timer.stop()
        self._audio.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_calibrate.setEnabled(True)
        self.status.setText("Stopped.")

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
                    self._last_voiced = True
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
            self._last_voiced = False
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


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(900, 650)
    win.show()
    sys.exit(app.exec())
