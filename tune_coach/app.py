from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from collections import deque

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import sounddevice as sd

from tune_coach.audio import AudioInput
from tune_coach.calibration import CalibrationError, calibrate_do_from_stream
from tune_coach.jianpu import JianpuAxis, JianpuQuantizer, TuningSystem
from tune_coach.metronome import Metronome
from tune_coach.pitch import PitchTracker, PitchTrackerConfig
from tune_coach.synth import Instrument, NoteSynth
from tune_coach.voice_shift import pitch_shift_formant


@dataclass(frozen=True)
class UiConfig:
    window_seconds: float = 24.0
    tick_seconds: float = 1.0
    left_margin_seconds: float = 0.4
    # Display delay to allow extra smoothing before rendering.
    delay_seconds: float = 0.25
    # Median window used for note stability detection.
    smooth_window: int = 11
    cent_smooth_window: int = 3
    # Minimum count of the same note to accept it as stable.
    stable_min_count: int = 7
    # Extra confirmations required to switch to a new note.
    switch_confirm: int = 3
    # Short silence tolerance (seconds) before cutting the line.
    silence_timeout: float = 0.2


class MainWindow(QtWidgets.QMainWindow):
    status_signal = QtCore.Signal(str, str)
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
    _FEMALE_DO_HZ = 261.6
    _KEY_OPTIONS = [
        ("1=C", 0),
        ("1=C#/Db", 1),
        ("1=D", 2),
        ("1=D#/Eb", 3),
        ("1=E", 4),
        ("1=F", 5),
        ("1=F#/Gb", 6),
        ("1=G", 7),
        ("1=G#/Ab", 8),
        ("1=A", 9),
        ("1=A#/Bb", 10),
        ("1=B", 11),
    ]

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
        self._recording = False
        self._recording_frames: list[np.ndarray] = []
        self._recording_samples = 0
        self._recording_limit_samples = int(10 * self._audio.sample_rate)
        self._recording_limit_hit = False
        self._recording_lock = threading.Lock()
        self._recorded_audio: np.ndarray | None = None
        self._play_thread: threading.Thread | None = None
        self._play_lock = threading.Lock()
        self._paused = False
        self._pause_time: float | None = None
        self._shift_digit_keys = dict(self._SHIFT_DIGIT_KEYS)
        for name, degree in (("Key_Circumflex", 6), ("Key_Dead_Circumflex", 6)):
            key = getattr(QtCore.Qt.Key, name, None)
            if key is not None:
                self._shift_digit_keys[key] = degree

        self._start_time: float | None = None
        self._times: list[float] = []
        self._ys: list[float] = []
        self._cent_times: list[float] = []
        self._cent_ys: list[float] = []
        # Buffer for delayed rendering (time, value).
        self._buffer = deque()
        self._cent_buffer = deque()
        # Short history for smoothing and stable note decision.
        self._y_smooth = deque(maxlen=self._ui.smooth_window)
        self._y_history = deque(maxlen=self._ui.smooth_window)
        self._cent_smooth = deque(maxlen=self._ui.cent_smooth_window)
        # Current stable note and candidate note for switching.
        self._current_y: int | None = None
        self._candidate_y: int | None = None
        self._candidate_count = 0
        # Track last voiced time to detect silence gaps.
        self._last_voiced_time: float | None = None
        self._last_output_nan = False
        self._cent_last_output_nan = False
        self._last_cent_value: float | None = None
        self._last_cent_time: float | None = None
        self._transition_start: float | None = None
        self._transition_target: int | None = None

        self._build_ui()
        self.status_signal.connect(self._set_status)
        self._apply_button_icons()

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)  # ~60 FPS UI update
        self._timer.timeout.connect(self._on_tick)
        self._reset_second_lines()
        QtWidgets.QApplication.instance().installEventFilter(self)
        self._synth.set_instrument(self.instrument_combo.currentText())
        self._apply_do_hz(self._DEFAULT_DO_HZ)
        self.plot.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.plot.setFocus()

    def _apply_button_icons(self) -> None:
        style = self.style()
        self.btn_start.setIcon(style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay))
        self.btn_stop.setIcon(style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaStop))
        self.btn_play.setIcon(style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay))
        record_icon = (
            QtWidgets.QStyle.StandardPixmap.SP_MediaRecord
            if hasattr(QtWidgets.QStyle.StandardPixmap, "SP_MediaRecord")
            else QtWidgets.QStyle.StandardPixmap.SP_DialogYesButton
        )
        self.btn_record.setIcon(style.standardIcon(record_icon))
        self._set_pause_icon(paused=False)
        self._apply_button_styles()
        self._set_pause_button_width()

    def _set_pause_icon(self, *, paused: bool) -> None:
        style = self.style()
        if paused:
            self.btn_pause.setIcon(style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay))
            self.btn_pause.setText("Resume")
        else:
            self.btn_pause.setIcon(style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPause))
            self.btn_pause.setText("Pause")
        self._set_pause_button_width()

    def _set_pause_button_width(self) -> None:
        metrics = self.btn_pause.fontMetrics()
        text_width = max(
            metrics.horizontalAdvance("Pause"),
            metrics.horizontalAdvance("Resume"),
        )
        icon_w = max(0, self.btn_pause.iconSize().width())
        padding = 24
        spacing = 8 if icon_w > 0 else 0
        self.btn_pause.setMinimumWidth(text_width + icon_w + padding + spacing)

    def _apply_button_styles(self) -> None:
        self._set_button_style(
            self.btn_start,
            base="#3fae5f",
            hover="#4fcf73",
            border="#2d8a49",
        )
        self._set_button_style(
            self.btn_pause,
            base="#e0a020",
            hover="#f0b232",
            border="#b27a18",
        )
        self._set_button_style(
            self.btn_stop,
            base="#d9534f",
            hover="#e46a66",
            border="#b84643",
        )
        self._set_button_style(
            self.btn_record,
            base="#c74b4b",
            hover="#de5a5a",
            border="#9f3a3a",
        )
        self._set_button_style(
            self.btn_play,
            base="#3f7acb",
            hover="#4a8fe6",
            border="#2f5fa0",
        )

    def _set_button_style(
        self, button: QtWidgets.QPushButton, *, base: str, hover: str, border: str
    ) -> None:
        button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {base};
                color: #ffffff;
                border: 1px solid {border};
                border-radius: 6px;
                padding: 4px 10px;
            }}
            QPushButton:hover {{
                background-color: {hover};
            }}
            QPushButton:disabled {{
                background-color: #3b3b3b;
                color: #8a8a8a;
                border: 1px solid #444444;
            }}
            """
        )

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(root)

        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)

        self.btn_calibrate = QtWidgets.QPushButton("Calibrate (4s)")
        self.radio_male = QtWidgets.QRadioButton("Male")
        self.radio_female = QtWidgets.QRadioButton("Female")
        self._preset_group = QtWidgets.QButtonGroup(self)
        self._preset_group.setExclusive(True)
        self._preset_group.addButton(self.radio_male)
        self._preset_group.addButton(self.radio_female)
        self.radio_male.setChecked(True)
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_start.setObjectName("btnStart")
        self.btn_pause.setObjectName("btnPause")
        self.btn_stop.setObjectName("btnStop")
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

        controls.addWidget(self.btn_calibrate)
        controls.addSpacing(8)
        controls.addWidget(self.radio_male)
        controls.addWidget(self.radio_female)
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
        controls.addWidget(self.btn_pause)
        controls.addWidget(self.btn_stop)

        self.btn_calibrate.clicked.connect(self._on_calibrate)
        self.radio_male.toggled.connect(self._on_preset_toggle)
        self.radio_female.toggled.connect(self._on_preset_toggle)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_pause.clicked.connect(self._on_pause_clicked)
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
        self._fix_spinbox_style(self.spin_bpm)
        met_layout.addWidget(self.chk_metronome)
        met_layout.addWidget(self.spin_bpm)
        self.chk_cent_curve = QtWidgets.QCheckBox("Cent Curve")
        self.chk_cent_curve.setChecked(False)
        met_layout.addSpacing(12)
        met_layout.addWidget(self.chk_cent_curve)
        met_layout.addStretch(1)

        self.chk_metronome.toggled.connect(self._on_metronome_toggle)
        self.spin_bpm.valueChanged.connect(self._on_bpm_change)
        self.chk_cent_curve.toggled.connect(self._on_cent_curve_toggle)

        self.btn_record = QtWidgets.QPushButton("Recording")
        self.btn_record.setEnabled(False)
        self.btn_record.setCheckable(False)
        self.spin_shift = QtWidgets.QSpinBox()
        self.spin_shift.setRange(-5, 5)
        self.spin_shift.setValue(0)
        self.spin_shift.setPrefix("Shift ")
        self.spin_shift.setSuffix(" steps")
        self.spin_shift.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        self._fix_spinbox_style(self.spin_shift)
        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_play.setEnabled(False)

        met_layout.addStretch(1)
        met_layout.addWidget(self.btn_record)
        met_layout.addSpacing(12)
        met_layout.addWidget(self.spin_shift)
        met_layout.addWidget(self.btn_play)

        self.btn_record.pressed.connect(self._on_record_start)
        self.btn_record.released.connect(self._on_record_stop)
        self.btn_play.clicked.connect(self._on_play)

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
        self._fix_spinbox_style(self.do_input)
        self.do_input.editingFinished.connect(self._on_do_input)
        self.key_combo = QtWidgets.QComboBox()
        for label, _ in self._KEY_OPTIONS:
            self.key_combo.addItem(label)
        self.key_combo.currentTextChanged.connect(self._on_key_change)
        self.key_combo.setFixedWidth(120)
        self.key_label = QtWidgets.QLabel("Key:")
        status_layout.addWidget(self.status)
        status_layout.addStretch(1)
        status_layout.addWidget(self.do_label)
        status_layout.addWidget(self.do_input)
        status_layout.addSpacing(8)
        status_layout.addWidget(self.key_label)
        status_layout.addWidget(self.key_combo)
        layout.addLayout(status_layout)

        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        self.plot.showGrid(x=False, y=True, alpha=0.15)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setMenuEnabled(False)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.scene().sigMouseClicked.connect(self._on_plot_click)

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
        self._cent_curve = self.plot.plot(
            [], [], pen=pg.mkPen(color=(255, 170, 60), width=2), connect="finite"
        )
        self._cent_curve.setVisible(False)

        self._second_lines: list[pg.InfiniteLine] = []
        self.setCentralWidget(root)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            self._timer.stop()
            self._audio.stop()
            self._metronome.stop()
            self._synth.stop()
            self._stop_playback()
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
        self._update_preset_radios(do_hz)

    @QtCore.Slot()
    def _on_start(self) -> None:
        if self._quantizer is None:
            self._set_status("Calibrate or enter Do (Hz).", "error")
            return

        self._reset_trace()
        self._start_time = time.monotonic()
        self._paused = False
        self._pause_time = None
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self._set_pause_icon(paused=False)
        self.btn_stop.setEnabled(True)
        self.btn_calibrate.setEnabled(False)
        self.btn_record.setEnabled(True)
        self.btn_play.setEnabled(self._recorded_audio is not None)
        self._set_status("Listening...", "info")

        self._audio.start()
        self._timer.start()

        self._reset_second_lines()

    @QtCore.Slot()
    def _on_stop(self) -> None:
        self._stop_listening()

    @QtCore.Slot()
    def _on_pause_clicked(self) -> None:
        if not self._timer.isActive() and not self._paused:
            return
        if self._paused:
            self._resume_listening()
        else:
            self._pause_listening()

    def _pause_listening(self) -> None:
        if self._paused or not self._timer.isActive():
            return
        self._paused = True
        self._pause_time = time.monotonic()
        self._timer.stop()
        self._audio.stop()
        self._audio.set_tap(None)
        self._stop_recording(cancel=True)
        self._stop_playback()
        self._metronome.stop()
        self.btn_record.setEnabled(False)
        self.btn_play.setEnabled(False)
        self._set_pause_icon(paused=True)
        self._set_status("Paused", "info")
        self._y_smooth.clear()
        self._y_history.clear()
        self._cent_smooth.clear()
        self._current_y = None
        self._candidate_y = None
        self._candidate_count = 0
        self._last_voiced_time = None
        self._last_output_nan = False
        self._cent_last_output_nan = False

    def _resume_listening(self) -> None:
        if not self._paused:
            return
        self._paused = False
        now = time.monotonic()
        if self._pause_time is not None and self._start_time is not None:
            self._start_time += now - self._pause_time
        self._pause_time = None
        self._audio.start()
        if self.chk_metronome.isChecked():
            self._metronome.set_bpm(int(self.spin_bpm.value()))
            self._metronome.start()
        self._timer.start()
        self.btn_record.setEnabled(True)
        self.btn_play.setEnabled(self._recorded_audio is not None)
        self._set_pause_icon(paused=False)
        self._set_status("Listening...", "info")

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

    @QtCore.Slot(bool)
    def _on_cent_curve_toggle(self, enabled: bool) -> None:
        self._cent_curve.setVisible(bool(enabled))
        self._cent_times.clear()
        self._cent_ys.clear()
        self._cent_buffer.clear()
        self._cent_smooth.clear()
        self._cent_last_output_nan = False
        self._last_cent_value = None
        self._last_cent_time = None
        self._transition_start = None
        self._transition_target = None
        if not enabled:
            self._cent_curve.setData([], [])

    def _on_record_start(self) -> None:
        if not self._timer.isActive():
            self._set_status("Press Start before recording.", "error")
            return
        if self._recording:
            return
        with self._recording_lock:
            self._recording_frames.clear()
            self._recording_samples = 0
        self._recording_limit_hit = False
        self._recording = True
        self._audio.set_tap(self._record_tap)
        self._set_status("Recording...", "info")

    def _on_record_stop(self) -> None:
        if self._recording:
            self._stop_recording()

    def _stop_recording(self, *, cancel: bool = False) -> None:
        self._audio.set_tap(None)
        if not self._recording and not cancel:
            return
        self._recording = False
        if cancel:
            with self._recording_lock:
                self._recording_frames.clear()
                self._recording_samples = 0
            self._recorded_audio = None
            return
        with self._recording_lock:
            frames = list(self._recording_frames)
            self._recording_frames.clear()
            self._recording_samples = 0
        if not frames:
            self._recorded_audio = None
            self._set_status("Listening...", "info")
            self.btn_play.setEnabled(False)
            return
        self._recorded_audio = np.concatenate(frames).astype(np.float32, copy=False)
        self.btn_play.setEnabled(True)
        self._set_status("Listening...", "info")

    def _record_tap(self, frame: np.ndarray) -> None:
        if not self._recording:
            return
        with self._recording_lock:
            self._recording_frames.append(frame.copy())
            self._recording_samples += frame.size
            if self._recording_samples >= self._recording_limit_samples:
                self._recording_limit_hit = True

    def _on_play(self) -> None:
        if not self._timer.isActive():
            self._set_status("Press Start before playback.", "error")
            return
        if self._recorded_audio is None:
            self._set_status("Record something first.", "error")
            return
        with self._play_lock:
            if self._play_thread is not None and self._play_thread.is_alive():
                return
            self._set_status("Processing...", "info")
            audio = self._recorded_audio.copy()
            steps = int(self.spin_shift.value())
            self._play_thread = threading.Thread(
                target=self._render_and_play, args=(audio, steps), daemon=True
            )
            self._play_thread.start()

    def _render_and_play(self, audio: np.ndarray, steps: int) -> None:
        try:
            start = time.monotonic()
            ratio = self._degree_shift_ratio(audio, steps)
            shifted = pitch_shift_formant(audio, self._audio.sample_rate, ratio)
            if shifted.size == 0:
                self._set_status_async("Playback failed (empty audio).", "error")
                return
            shifted = self._match_rms(shifted, audio, max_gain=6.0)
            shifted = self._apply_fade(shifted, ms=8.0)
            elapsed = time.monotonic() - start
            if elapsed > 8.0:
                self._set_status_async(f"Processing slow ({elapsed:.1f}s).", "error")
            self._set_status_async("Playing...", "info")
            sd.play(shifted, samplerate=self._audio.sample_rate, blocking=True)
            self._set_status_async("Listening...", "info")
        except Exception as exc:  # noqa: BLE001 - user-facing
            self._set_status_async(f"Playback failed: {exc}", "error")

    def _match_rms(self, audio: np.ndarray, ref: np.ndarray, *, max_gain: float) -> np.ndarray:
        if audio.size == 0 or ref.size == 0:
            return audio
        rms_audio = self._percentile_rms(audio, 90.0)
        rms_ref = self._percentile_rms(ref, 90.0)
        if rms_audio < 1e-6 or rms_ref < 1e-6:
            return audio
        gain = rms_ref / rms_audio
        gain = float(max(0.3, min(max_gain, gain)))
        out = audio * gain
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 0.99:
            out = out / peak * 0.99
        return out

    def _percentile_rms(self, audio: np.ndarray, percentile: float) -> float:
        if audio.size == 0:
            return 0.0
        hop = max(256, self._audio.block_size)
        window = max(hop, 1024)
        x = audio.astype(np.float32, copy=False)
        rms_list: list[float] = []
        for i in range(0, len(x) - window + 1, hop):
            frame = x[i : i + window]
            rms = float(np.sqrt(np.mean(np.square(frame))))
            rms_list.append(rms)
        if not rms_list:
            return float(np.sqrt(np.mean(np.square(x))))
        return float(np.percentile(np.array(rms_list, dtype=np.float32), percentile))

    def _apply_fade(self, audio: np.ndarray, *, ms: float) -> np.ndarray:
        if audio.size == 0:
            return audio
        n = int((ms / 1000.0) * self._audio.sample_rate)
        n = max(1, min(n, audio.size // 2))
        out = audio.copy()
        fade_in = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, n, endpoint=False, dtype=np.float32)
        out[:n] *= fade_in
        out[-n:] *= fade_out
        return out

    def _set_status_async(self, text: str, kind: str) -> None:
        self.status_signal.emit(text, kind)

    def _estimate_recording_pitch(self, audio: np.ndarray) -> float | None:
        tracker = PitchTracker(PitchTrackerConfig(sample_rate=self._audio.sample_rate))
        hop = self._audio.block_size
        hz_list: list[float] = []
        for i in range(0, len(audio), hop):
            frame = audio[i : i + hop]
            if frame.size == 0:
                continue
            hz = tracker.process(frame)
            if hz is not None:
                hz_list.append(hz)
        if not hz_list:
            return None
        return float(np.median(np.array(hz_list, dtype=np.float32)))

    def _degree_shift_ratio(self, audio: np.ndarray, steps: int) -> float:
        if steps == 0:
            return 1.0
        quantizer = self._quantizer
        if quantizer is None:
            quantizer = JianpuQuantizer(do_hz=self._DEFAULT_DO_HZ, tuning=TuningSystem.JUST_INTONATION)
        base_hz = self._estimate_recording_pitch(audio)
        if base_hz is None or base_hz <= 0:
            return float(2.0 ** ((2 * steps) / 12.0))
        nearest = quantizer.nearest_degree(base_hz)
        if nearest is None:
            return float(2.0 ** ((2 * steps) / 12.0))
        degree, octave = nearest
        new_degree, octave_shift = self._shift_degree(degree, steps)
        target = quantizer.degree_hz(new_degree, octave=octave + octave_shift)
        if target is None:
            return float(2.0 ** ((2 * steps) / 12.0))
        return float(target / base_hz)

    def _shift_degree(self, degree: int, steps: int) -> tuple[int, int]:
        idx = (degree - 1) + steps
        octave_shift = idx // 7
        new_idx = idx % 7
        return int(new_idx + 1), int(octave_shift)

    def _stop_playback(self) -> None:
        sd.stop()

    def _on_tuning_change(self, text: str) -> None:
        tuning = self._parse_tuning(text)
        if self._quantizer is not None:
            self._quantizer = JianpuQuantizer(
                do_hz=self._quantizer.do_hz,
                octave_gap=self._quantizer.octave_gap,
                tuning=tuning,
                key_semitone=self._quantizer.key_semitone,
            )
            self._reset_trace()

    def _on_key_change(self, text: str) -> None:
        key_semitone = self._parse_key_semitone(text)
        if self._quantizer is not None:
            self._quantizer = JianpuQuantizer(
                do_hz=self._quantizer.do_hz,
                octave_gap=self._quantizer.octave_gap,
                tuning=self._quantizer.tuning,
                key_semitone=key_semitone,
            )
            self._reset_trace()

    def _on_instrument_change(self, text: str) -> None:
        self._synth.set_instrument(text)

    def _on_preset_toggle(self) -> None:
        if self.radio_male.isChecked():
            self._apply_do_hz(self._DEFAULT_DO_HZ)
        elif self.radio_female.isChecked():
            self._apply_do_hz(self._FEMALE_DO_HZ)

    def _on_do_input(self) -> None:
        do_hz = float(self.do_input.value())
        if do_hz <= 0:
            return
        self._apply_do_hz(do_hz)
        self._update_preset_radios(do_hz)

    def _apply_do_hz(self, do_hz: float) -> None:
        if self._timer.isActive():
            self._stop_listening()
        tuning = self._parse_tuning(self.tuning_combo.currentText())
        key_semitone = self._parse_key_semitone(self.key_combo.currentText())
        self._quantizer = JianpuQuantizer(
            do_hz=do_hz,
            octave_gap=0,
            tuning=tuning,
            key_semitone=key_semitone,
        )
        self._set_do_input(do_hz)
        self._reset_trace()
        self._set_status("Ready to listen", "ready")
        self.btn_start.setEnabled(True)

    def _parse_key_semitone(self, text: str) -> int:
        for label, semitone in self._KEY_OPTIONS:
            if label == text:
                return int(semitone)
        return 0

    def _update_preset_radios(self, do_hz: float) -> None:
        is_male = abs(do_hz - self._DEFAULT_DO_HZ) <= 0.1
        is_female = abs(do_hz - self._FEMALE_DO_HZ) <= 0.1
        self._preset_group.setExclusive(False)
        self.radio_male.setChecked(is_male)
        self.radio_female.setChecked(is_female)
        self._preset_group.setExclusive(True)

    def _set_do_input(self, do_hz: float) -> None:
        self.do_input.blockSignals(True)
        self.do_input.setValue(float(do_hz))
        self.do_input.blockSignals(False)

    def _on_plot_click(self, event) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if not self._timer.isActive() and not self._paused:
            return
        if self._paused:
            self._resume_listening()
        else:
            self._pause_listening()

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

    def _fix_spinbox_style(self, widget: QtWidgets.QAbstractSpinBox) -> None:
        if sys.platform != "darwin":
            return
        style = QtWidgets.QStyleFactory.create("Fusion")
        if style is not None:
            widget.setStyle(style)

    def _reset_trace(self) -> None:
        self._times.clear()
        self._ys.clear()
        self._cent_times.clear()
        self._cent_ys.clear()
        self._buffer.clear()
        self._cent_buffer.clear()
        self._y_smooth.clear()
        self._y_history.clear()
        self._cent_smooth.clear()
        self._current_y = None
        self._candidate_y = None
        self._candidate_count = 0
        self._last_voiced_time = None
        self._last_output_nan = False
        self._cent_last_output_nan = False
        self._last_cent_value = None
        self._last_cent_time = None
        self._transition_start = None
        self._transition_target = None

    def _stop_listening(self) -> None:
        self._timer.stop()
        self._audio.stop()
        self._audio.set_tap(None)
        self._stop_recording(cancel=True)
        self._stop_playback()
        self._paused = False
        self._pause_time = None
        self.btn_start.setEnabled(self._quantizer is not None)
        self.btn_pause.setEnabled(False)
        self._set_pause_icon(paused=False)
        self.btn_stop.setEnabled(False)
        self.btn_calibrate.setEnabled(True)
        self.btn_record.setEnabled(False)
        self.btn_play.setEnabled(False)
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

    def _backfill_buffer(self, start_time: float, y_value: int) -> None:
        if not self._buffer:
            return
        updated = deque()
        for bt, by in self._buffer:
            if bt >= start_time:
                updated.append((bt, float(y_value)))
            else:
                updated.append((bt, by))
        self._buffer = updated

    def _on_tick(self) -> None:
        if self._start_time is None or self._quantizer is None:
            return

        now = time.monotonic()
        t = now - self._start_time

        if self._recording_limit_hit:
            self._recording_limit_hit = False
            self._stop_recording()

        pitch_hz: float | None = None
        observed_y: int | None = None
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
                    observed_y = y_sm
                    self._y_history.append(y_sm)
                    self._last_voiced_time = t
                else:
                    self._y_smooth.clear()
            else:
                self._y_smooth.clear()
                if self.chk_cent_curve.isChecked():
                    self._cent_smooth.clear()

        if observed_y is not None:
            if self._current_y is None:
                if self._transition_target != observed_y:
                    self._transition_start = t
                    self._transition_target = observed_y
            elif observed_y != self._current_y:
                if self._transition_target != observed_y:
                    self._transition_start = t
                    self._transition_target = observed_y
            else:
                self._transition_start = None
                self._transition_target = None

        cent_y: float | None = None
        if self.chk_cent_curve.isChecked() and pitch_hz is not None:
            cent_y = self._quantizer.continuous_y(pitch_hz)
            if cent_y is not None:
                self._cent_smooth.append(cent_y)
                if len(self._cent_smooth) >= 3:
                    cent_y = float(np.median(np.array(self._cent_smooth, dtype=np.float32)))
                self._last_cent_value = cent_y
                self._last_cent_time = t
            else:
                self._cent_smooth.clear()

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
                        start_time = (
                            self._transition_start
                            if self._transition_target == stable_y
                            else None
                        )
                        if start_time is not None:
                            self._backfill_buffer(start_time, stable_y)
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
                            start_time = (
                                self._transition_start
                                if self._transition_target == stable_y
                                else None
                            )
                            if start_time is not None:
                                self._backfill_buffer(start_time, stable_y)
                            self._transition_start = None
                            self._transition_target = None
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

        if self.chk_cent_curve.isChecked():
            cent_output: float | None = cent_y
            if cent_output is None and self._last_cent_time is not None:
                if (t - self._last_cent_time) <= self._ui.silence_timeout:
                    cent_output = self._last_cent_value
            if cent_output is None:
                if not self._cent_last_output_nan:
                    self._cent_buffer.append((t, float("nan")))
                    self._cent_last_output_nan = True
            else:
                self._cent_buffer.append((t, float(cent_output)))
                self._cent_last_output_nan = False

        # release buffered points with a small delay for extra smoothness
        release_before = t - self._ui.delay_seconds
        while self._buffer and self._buffer[0][0] <= release_before:
            bt, by = self._buffer.popleft()
            self._times.append(bt)
            self._ys.append(by)
        while self._cent_buffer and self._cent_buffer[0][0] <= release_before:
            ct, cy = self._cent_buffer.popleft()
            self._cent_times.append(ct)
            self._cent_ys.append(cy)

        # keep only last window
        min_t = t - self._ui.window_seconds
        while self._times and self._times[0] < min_t:
            self._times.pop(0)
            self._ys.pop(0)
        while self._cent_times and self._cent_times[0] < min_t:
            self._cent_times.pop(0)
            self._cent_ys.pop(0)

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
        if self._cent_curve.isVisible() and self._cent_times:
            cx = np.array(self._cent_times, dtype=np.float32) - t
            cy = np.array(self._cent_ys, dtype=np.float32)
            self._cent_curve.setData(cx, cy)
        elif self._cent_curve.isVisible():
            self._cent_curve.setData([], [])

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
