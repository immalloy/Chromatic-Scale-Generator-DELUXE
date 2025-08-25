# Chromatic Scale Generator Deluxe (PySide6 / Qt)
# GPL-3.0-or-later
#
# Requirements:
#   pip install praat-parselmouth PySide6 numpy

import os
import glob
import random
import wave
import struct
from dataclasses import dataclass
from os.path import join, exists, basename

import numpy as np
import parselmouth

from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication, QWidget, QFileDialog, QMessageBox,
    QGridLayout, QLabel, QLineEdit, QPushButton, QComboBox,
    QCheckBox, QHBoxLayout, QVBoxLayout, QGroupBox, QProgressBar
)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
OCTAVES = [str(n) for n in range(0, 9)]  # 0..8


@dataclass
class Settings:
    folder: str
    start_note_index: int
    start_octave: int
    range_semitones: int
    gap_seconds: float
    pitched: bool
    dump_samples: bool
    order_mode: str  # 'sequential' | 'shuffle' | 'random'
    normalize: bool
    fade_ms: int
    out_samplerate: int
    fixed_note_len: float  # seconds, 0 to disable


# --------------------------- Audio helpers ---------------------------

def hz_from_semitone(semitone_index: int) -> float:
    return 32.703 * (2 ** (semitone_index / 12.0))  # C1 base


def quantize_seconds_to_frames(seconds: float, samplerate: int) -> int:
    if seconds <= 0:
        return 0
    return max(0, int(round(seconds * samplerate)))


def seconds_from_frames(frames: int, samplerate: int) -> float:
    return 0.0 if frames <= 0 else frames / float(samplerate)


def make_silence(seconds: float, samplerate: int) -> parselmouth.Sound:
    frames = quantize_seconds_to_frames(seconds, samplerate)
    dur = seconds_from_frames(frames, samplerate)
    return parselmouth.praat.call(
        "Create Sound from formula", "silence", 1, 0, dur, samplerate, "0"
    )


def to_mono_48k(snd: parselmouth.Sound) -> parselmouth.Sound:
    s = parselmouth.praat.call(snd, "Resample", 48000, 1)
    s = parselmouth.praat.call(s, "Convert to mono")
    return s


def list_source_files(folder: str):
    numbered = []
    i = 1
    while exists(join(folder, f"{i}.wav")):
        numbered.append(join(folder, f"{i}.wav"))
        i += 1
    if numbered:
        return numbered
    files = sorted(glob.glob(join(folder, "*.wav")))
    # avoid consuming our own output
    return [f for f in files if basename(f).lower() != "chromatic.wav"]


def load_source(path: str) -> parselmouth.Sound:
    return to_mono_48k(parselmouth.Sound(path))


def retune_to_freq(snd: parselmouth.Sound, freq_hz: float) -> parselmouth.Sound:
    # Lower pitch floor helps for C0/C1; faster time step improves tracking
    manipulation = parselmouth.praat.call(snd, "To Manipulation", 0.03, 20, 1200)
    pitch_tier = parselmouth.praat.call(manipulation, "Extract pitch tier")
    parselmouth.praat.call(pitch_tier, "Formula", f"{freq_hz}")
    parselmouth.praat.call([pitch_tier, manipulation], "Replace pitch tier")
    return parselmouth.praat.call(manipulation, "Get resynthesis (overlap-add)")


def apply_fade(snd: parselmouth.Sound, fade_ms: int) -> parselmouth.Sound:
    if fade_ms <= 0:
        return snd
    dur = snd.get_total_duration()
    fade_s = min(fade_ms / 1000.0, dur / 2) if dur > 0 else 0
    if fade_s <= 0:
        return snd
    snd2 = parselmouth.Sound(snd.values.copy(), snd.sampling_frequency)
    parselmouth.praat.call(snd2, "Fade in", 0.0, fade_s)
    parselmouth.praat.call(snd2, "Fade out", dur - fade_s, dur)
    return snd2


def peak_normalize(snd: parselmouth.Sound, target_peak: float = 0.98) -> parselmouth.Sound:
    mx = float(max(abs(snd.values.min()), abs(snd.values.max())))
    if mx <= 1e-9:
        return snd
    gain = target_peak / mx
    return parselmouth.Sound(snd.values * gain, snd.sampling_frequency)


def pad_or_trim(snd: parselmouth.Sound, length_s: float, samplerate: int) -> parselmouth.Sound:
    if length_s <= 0:
        return snd
    target_frames = quantize_seconds_to_frames(length_s, samplerate)
    if target_frames <= 0:
        return make_silence(0.0, samplerate)
    cur_frames = quantize_seconds_to_frames(snd.get_total_duration(), int(snd.sampling_frequency))
    if cur_frames > target_frames:
        exact_end = seconds_from_frames(target_frames, samplerate)
        return parselmouth.praat.call(snd, "Extract part", 0, exact_end, "rectangular", 1, "yes")
    elif cur_frames < target_frames:
        pad_frames = target_frames - cur_frames
        silence = make_silence(seconds_from_frames(pad_frames, samplerate), samplerate)
        return parselmouth.Sound.concatenate([snd, silence])
    else:
        return snd


def resample_if_needed(snd: parselmouth.Sound, samplerate: int) -> parselmouth.Sound:
    if int(snd.sampling_frequency) == samplerate:
        return snd
    return parselmouth.praat.call(snd, "Resample", samplerate, 1)


def sound_to_int16_frames(snd: parselmouth.Sound) -> bytes:
    """Convert mono Sound (-1..1 floats) to PCM16 bytes."""
    arr = np.asarray(snd.values, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype(np.int16).tobytes()


# --------- RIFF cue + adtl label chunks (Slicex-friendly) + size fix ---------

def _riff_align2(n: int) -> int:
    return n + (n & 1)

def _append_markers_and_fix_riff(path: str, marker_frames: list[int]):
    """
    Append RIFF 'cue ' and 'LIST/adtl' ('labl') so FL Studio Slicex shows slice markers.
    - marker_frames are sample-frame indices from the start of the 'data' chunk (mono PCM16).
    """
    if not marker_frames:
        return

    with open(path, 'ab') as f:
        # ---------------- cue --------------
        num = len(marker_frames)
        cue_chunk_size = 4 + num * 24
        f.write(b'cue ')
        f.write(struct.pack('<I', cue_chunk_size))
        f.write(struct.pack('<I', num))
        for i, frames in enumerate(marker_frames, start=1):
            # dwIdentifier, dwPosition(frames), fccChunk('data'), dwChunkStart, dwBlockStart, dwSampleOffset(frames)
            f.write(struct.pack('<I', i))
            f.write(struct.pack('<I', frames))
            f.write(b'data')
            f.write(struct.pack('<I', 0))
            f.write(struct.pack('<I', 0))
            f.write(struct.pack('<I', frames))

        # --------------- LIST/adtl with labl per marker ---------------
        label_chunks = []
        for i in range(1, num + 1):
            txt = f"Marker {i}".encode('utf-8') + b'\x00'
            labl_size = 4 + len(txt)               # cue ID + text
            payload = struct.pack('<I', i) + txt
            pad = (labl_size & 1)
            label_chunks.append(b'labl' + struct.pack('<I', labl_size) + payload + (b'\x00' if pad else b''))

        adtl_payload = b''.join(label_chunks)
        list_size = 4 + len(adtl_payload)          # 'adtl' tag + children
        f.write(b'LIST')
        f.write(struct.pack('<I', list_size))
        f.write(b'adtl')
        f.write(adtl_payload)

        file_size = f.tell()

    # Fix RIFF size at offset 4 = total size - 8
    with open(path, 'r+b') as f:
        riff_size = file_size - 8
        f.seek(4)
        f.write(struct.pack('<I', riff_size))


# --------------------------- Core logic (streaming, callback-able) ---------------------------

def generate_stream(settings: Settings, on_progress=None, is_canceled=None) -> str:
    files = list_source_files(settings.folder)
    if not files:
        raise RuntimeError("No WAV files found in the folder.")

    order = list(range(len(files)))
    if settings.order_mode == "shuffle":
        random.shuffle(order)
    elif settings.order_mode == "random":
        order = [random.randrange(0, len(files)) for _ in range(settings.range_semitones)]

    def pick_idx(i):
        if settings.order_mode == "random":
            return order[i]
        return order[i % len(order)]

    gap_48k = make_silence(settings.gap_seconds, 48000)
    starting_key = settings.start_note_index + 12 * (settings.start_octave - 1)

    out_path = join(settings.folder, "chromatic.wav")
    marker_frames: list[int] = []
    current_frames = 0  # frames written to 'data' so far (mono int16 => 1 frame = 2 bytes)

    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(settings.out_samplerate)

        out_dir = None
        if settings.dump_samples:
            sub = "pitched_samples" if settings.pitched else "samples"
            out_dir = join(settings.folder, sub)
            os.makedirs(out_dir, exist_ok=True)

        total = settings.range_semitones
        for i in range(total):
            if is_canceled and is_canceled():
                raise KeyboardInterrupt

            # record marker at the start of the note
            marker_frames.append(current_frames)

            src = load_source(files[pick_idx(i)])
            if settings.pitched:
                semitone_index = i + starting_key
                freq = hz_from_semitone(semitone_index)
                snd = retune_to_freq(src, freq)
            else:
                snd = src

            if settings.fade_ms > 0:
                snd = apply_fade(snd, settings.fade_ms)
            if settings.fixed_note_len > 0:
                snd = pad_or_trim(snd, settings.fixed_note_len, 48000)
            if settings.normalize:
                snd = peak_normalize(snd, 0.98)

            snd_out = resample_if_needed(snd, settings.out_samplerate)
            pcm = sound_to_int16_frames(snd_out)
            w.writeframes(pcm)
            current_frames += len(pcm) // 2  # bytes -> frames for mono int16

            if out_dir is not None:
                snd_out.save(join(out_dir, f"note_{i+1}.wav"), "WAV")

            if i < total - 1 and settings.gap_seconds > 0:
                gap_out = resample_if_needed(gap_48k, settings.out_samplerate)
                gpcm = sound_to_int16_frames(gap_out)
                w.writeframes(gpcm)
                current_frames += len(gpcm) // 2

            if on_progress:
                total_semitone = settings.start_note_index + i
                label = f"{NOTE_NAMES[total_semitone % 12]}{settings.start_octave + (total_semitone // 12)}"
                on_progress(i + 1, total, label)

    # append marker chunks after writing audio
    _append_markers_and_fix_riff(out_path, marker_frames)
    return out_path


# --------------------------- Worker + UI ---------------------------

class GenWorker(QObject):
    progressed = Signal(int, int, str)
    finished = Signal(str)
    failed = Signal(str)
    canceled = Signal()

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _is_canceled(self):
        return self._cancel

    def run(self):
        try:
            def _progress(d, t, label):
                self.progressed.emit(d, t, label)
            out = generate_stream(self.settings, on_progress=_progress, is_canceled=self._is_canceled)
            if self._cancel:
                self.canceled.emit()
            else:
                self.finished.emit(out)
        except KeyboardInterrupt:
            self.canceled.emit()
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chromatic Scale Generator Deluxe")
        # Change this path to your icon, or use a portable resource helper.
        self.setWindowIcon(QIcon(r"C:\Users\Henry J\Downloads\src\icon.ico"))
        self.setMinimumWidth(720)
        self._build_ui()
        self.worker_thread = None
        self.worker = None

    def _group(self, title: str, layout: QGridLayout | QVBoxLayout) -> QGroupBox:
        box = QGroupBox(title)
        box.setLayout(layout)
        return box

    def _build_ui(self):
        root = QVBoxLayout(self)

        # Source
        grid_src = QGridLayout(); r = 0
        grid_src.addWidget(QLabel("Sample folder:"), r, 0)
        self.folder_edit = QLineEdit(); grid_src.addWidget(self.folder_edit, r, 1)
        btn_browse = QPushButton("Browse"); btn_browse.clicked.connect(self._browse); grid_src.addWidget(btn_browse, r, 2); r += 1
        root.addWidget(self._group("Source", grid_src))

        # Pitch & Range
        grid_pitch = QGridLayout(); r = 0
        grid_pitch.addWidget(QLabel("Start note:"), r, 0)
        self.note_combo = QComboBox(); self.note_combo.addItems(NOTE_NAMES); grid_pitch.addWidget(self.note_combo, r, 1); r += 1
        grid_pitch.addWidget(QLabel("Start octave:"), r, 0)
        self.oct_combo = QComboBox(); self.oct_combo.addItems(OCTAVES); self.oct_combo.setCurrentText("2"); grid_pitch.addWidget(self.oct_combo, r, 1); r += 1
        grid_pitch.addWidget(QLabel("Range (semitones):"), r, 0)
        self.range_edit = QLineEdit("36"); grid_pitch.addWidget(self.range_edit, r, 1); r += 1
        grid_pitch.addWidget(QLabel("Sample gap (s):"), r, 0)
        self.gap_edit = QLineEdit("0.3"); grid_pitch.addWidget(self.gap_edit, r, 1); r += 1
        grid_pitch.addWidget(QLabel("Order:"), r, 0)
        self.order_combo = QComboBox(); self.order_combo.addItems(["sequential", "shuffle", "random"]); grid_pitch.addWidget(self.order_combo, r, 1); r += 1
        root.addWidget(self._group("Pitch & Range", grid_pitch))

        # Options
        grid_opt = QGridLayout(); r = 0
        self.pitched_check = QCheckBox("Pitched (retune)"); self.pitched_check.setChecked(True)
        grid_opt.addWidget(self.pitched_check, r, 0)
        self.dump_check = QCheckBox("Dump individual samples")
        grid_opt.addWidget(self.dump_check, r, 1); r += 1
        self.norm_check = QCheckBox("Peak normalize")
        grid_opt.addWidget(self.norm_check, r, 0)
        grid_opt.addWidget(QLabel("Fade (ms):"), r, 1); self.fade_edit = QLineEdit("0"); grid_opt.addWidget(self.fade_edit, r, 2); r += 1
        grid_opt.addWidget(QLabel("Fixed note length (s, 0 = original):"), r, 0); self.len_edit = QLineEdit("0"); grid_opt.addWidget(self.len_edit, r, 1); r += 1
        grid_opt.addWidget(QLabel("Output sample rate:"), r, 0); self.sr_combo = QComboBox(); self.sr_combo.addItems(["48000", "44100"]); self.sr_combo.setCurrentText("48000"); grid_opt.addWidget(self.sr_combo, r, 1); r += 1
        root.addWidget(self._group("Options", grid_opt))

        # Actions & progress
        row = QHBoxLayout()
        self.btn_go = QPushButton("Generate")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_go.clicked.connect(self._on_generate)
        self.btn_cancel.clicked.connect(self._on_cancel)
        row.addWidget(self.btn_go); row.addWidget(self.btn_cancel)
        root.addLayout(row)

        self.bar = QProgressBar(); self.bar.setRange(0, 1); self.bar.setValue(0)
        self.status = QLabel("Idle")
        root.addWidget(self.bar)
        root.addWidget(self.status)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Choose sample folder", "")
        if d:
            self.folder_edit.setText(d)

    def _set_busy(self, busy: bool, total: int = 0):
        for w in [self.folder_edit, self.note_combo, self.oct_combo, self.range_edit, self.gap_edit,
                  self.order_combo, self.pitched_check, self.dump_check, self.norm_check,
                  self.fade_edit, self.len_edit, self.sr_combo, self.btn_go]:
            w.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        if busy:
            self.bar.setRange(0, total); self.bar.setValue(0)
        else:
            self.bar.setRange(0, 1); self.bar.setValue(0)

    def _on_generate(self):
        try:
            folder = self.folder_edit.text().strip()
            if not folder or not os.path.isdir(folder):
                raise ValueError("Choose a valid folder.")
            rng = int(self.range_edit.text())
            gap = float(self.gap_edit.text())
            if rng <= 0: raise ValueError("Range must be > 0.")
            if gap < 0:  raise ValueError("Gap cannot be negative.")
            fade_ms = int(float(self.fade_edit.text()))
            fixed_len = float(self.len_edit.text())
            if fixed_len < 0: raise ValueError("Fixed length cannot be negative.")
            settings = Settings(
                folder=folder,
                start_note_index=self.note_combo.currentIndex(),
                start_octave=int(self.oct_combo.currentText()),
                range_semitones=rng,
                gap_seconds=gap,
                pitched=self.pitched_check.isChecked(),
                dump_samples=self.dump_check.isChecked(),
                order_mode=self.order_combo.currentText(),
                normalize=self.norm_check.isChecked(),
                fade_ms=fade_ms,
                out_samplerate=int(self.sr_combo.currentText()),
                fixed_note_len=fixed_len,
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e)); return

        self._set_busy(True, settings.range_semitones)
        self.status.setText("Rendering…")

        self.worker = GenWorker(settings)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progressed.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.canceled.connect(self._on_canceled)
        self.worker_thread.start()

    def _on_cancel(self):
        if self.worker:
            self.worker.cancel()
            self.status.setText("Canceling…")

    def _on_progress(self, done: int, total: int, label: str):
        self.bar.setMaximum(total)
        self.bar.setValue(done)
        self.status.setText(f"Note {done}/{total} · {label}")

    def _on_finished(self, out_path: str):
        self._teardown_worker()
        self._set_busy(False)
        self.status.setText("Done")
        QMessageBox.information(self, "Success", f"Saved\n{out_path}")

    def _on_failed(self, msg: str):
        self._teardown_worker()
        self._set_busy(False)
        self.status.setText("Error")
        QMessageBox.critical(self, "Error", msg)

    def _on_canceled(self):
        self._teardown_worker()
        self._set_busy(False)
        self.status.setText("Canceled")
        QMessageBox.information(self, "Canceled", "Generation canceled.")

    def _teardown_worker(self):
        if self.worker_thread:
            self.worker_thread.quit(); self.worker_thread.wait()
        self.worker_thread = None
        self.worker = None


if __name__ == "__main__":
    app = QApplication([])
    w = MainWindow()
    w.show()
    app.exec()
