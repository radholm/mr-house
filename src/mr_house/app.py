"""The orchestrator: the Mr. House state machine.

Runs in a background thread (the display owns the main thread) and drives the
full loop:

    IDLE ──wake──▶ LISTENING ──speech──▶ THINKING ──tokens──▶ SPEAKING ──▶ IDLE

Latency tricks live here:

* As soon as a question is sent we arm a *filler timer*; if the first token is
  slow we speak an in-character "thinking" line so there's no dead air.
* The brain yields **sentences**; each is synthesized + effected and queued for
  playback immediately, so speech starts on sentence 1 while sentence 2 is still
  being written.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .config import Config, REPO_ROOT
from .audio.capture import MicrophoneStream
from .audio.wake_word import WakeWordDetector
from .audio.vad import UtteranceRecorder
from .audio.stt import SpeechToText
from .audio.tts import TextToSpeech
from .audio.effects import VoiceEffects
from .audio.player import AudioPlayer
from .brain.personality import build_system_prompt
from .brain.memory import ConversationMemory
from .brain.fillers import random_filler, random_tool_filler
from .brain.mcp_tools import MCPToolManager
from .brain.llm import Brain

log = logging.getLogger(__name__)


class MrHouse:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._stop = threading.Event()
        self._stopped_done = False
        self._worker: Optional[threading.Thread] = None

        # --- audio ---
        self.mic = MicrophoneStream(
            sample_rate=cfg.audio.sample_rate,
            block_size=cfg.audio.block_size,
            device=cfg.audio.input_device,
        )
        self.wake = WakeWordDetector(
            models=cfg.wake_word.models,
            threshold=cfg.wake_word.threshold,
            repo_root=REPO_ROOT,
        ) if cfg.wake_word.enabled else None
        self.recorder = UtteranceRecorder(
            sample_rate=cfg.audio.sample_rate,
            aggressiveness=cfg.vad.aggressiveness,
            silence_ms=cfg.vad.silence_ms,
            max_utterance_ms=cfg.vad.max_utterance_ms,
            start_timeout_ms=cfg.vad.start_timeout_ms,
            start_speech_ms=cfg.vad.start_speech_ms,
            min_voiced_ms=cfg.vad.min_voiced_ms,
        )
        self.stt = SpeechToText(
            model=cfg.stt.model,
            device=cfg.stt.device,
            compute_type=cfg.stt.compute_type,
            language=cfg.stt.language,
            beam_size=cfg.stt.beam_size,
            cpu_threads=cfg.stt.cpu_threads,
            vad_filter=cfg.stt.vad_filter,
        ) if cfg.stt.enabled else None
        self.tts = TextToSpeech(
            voice_path=str(cfg.resolve_path(cfg.tts.voice)),
            length_scale=cfg.tts.length_scale,
            noise_scale=cfg.tts.noise_scale,
            noise_w=cfg.tts.noise_w,
            expressiveness=cfg.tts.expressiveness,
            sentence_silence=cfg.tts.sentence_silence,
        ) if cfg.tts.enabled else None
        out_sr = self.tts.sample_rate if (self.tts and self.tts.available) else cfg.audio.output_sample_rate
        self.fx = VoiceEffects(cfg.voice_fx, sample_rate=out_sr)
        self.player = AudioPlayer(sample_rate=out_sr, device=cfg.audio.output_device)

        # --- brain ---
        self.tools = MCPToolManager(cfg.mcp.servers) if cfg.mcp.enabled else None
        persist_path = (
            str(cfg.resolve_path(cfg.brain.memory_file))
            if cfg.brain.persist_memory else None
        )
        self.memory = ConversationMemory(
            system_prompt=build_system_prompt(),
            max_turns=cfg.brain.max_history_turns,
            persist_path=persist_path,
        )
        self.brain = Brain(cfg.brain, self.memory, self.tools)

        # filler control
        self._filler_timer: Optional[threading.Timer] = None
        self._first_token_seen = threading.Event()
        # Serialises TTS synthesis + queueing so the filler (spoken from a timer
        # thread) and the response (spoken from the worker thread) never race —
        # the filler is always fully queued before the response, and the
        # sequential player then guarantees the response waits for it to finish.
        self._tts_lock = threading.Lock()

    # ------------------------------------------------------------------ api
    @property
    def is_speaking(self) -> bool:
        return self.player.is_speaking

    def start(self) -> None:
        """Start all subsystems and the worker thread."""
        self.player.start()
        if self.tools is not None:
            self.tools.start()
        try:
            self.mic.start()
        except Exception as exc:
            log.error("Could not start microphone: %s", exc)
        self._worker = threading.Thread(target=self._run, name="MrHouseWorker", daemon=True)
        self._worker.start()

    def request_stop(self) -> None:
        """Lightweight: signal all loops to stop (safe from a signal handler)."""
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def stop(self) -> None:
        # Idempotent: cleanup runs once even if called from several places.
        if self._stopped_done:
            return
        self._stopped_done = True
        self._stop.set()
        self._cancel_filler()
        try:
            self.player.stop()
        except Exception:
            pass
        try:
            self.mic.stop()
        except Exception:
            pass
        if self.tools is not None:
            try:
                self.tools.stop()
            except Exception:
                pass
        if self._worker is not None:
            self._worker.join(timeout=2)
        log.info("Mr. House is offline.")

    # --------------------------------------------------------------- loop
    def _run(self) -> None:
        log.info("Mr. House is awake. Say the wake word to begin.")
        self._greet()
        while not self._stop.is_set():
            try:
                if not self._wait_for_wake():
                    continue
                self._conversation()
            except Exception as exc:
                log.exception("Interaction error: %s", exc)
                time.sleep(0.5)

    def _greet(self) -> None:
        line = "Systems online. I am at your disposal."
        log.info("House: %s", line)
        self._speak(line)

    def _wait_for_wake(self) -> bool:
        """Block until the wake word is heard (or no wake configured)."""
        if self.wake is None or not self.wake.available:
            # No wake word: act on any speech. Wait a beat then proceed.
            time.sleep(0.2)
            return True
        self.mic.flush()
        last_status = time.time()
        peak_score = 0.0
        peak_rms = 0.0
        silent_for = 0.0
        warned_silence = False
        block_dur = self.cfg.audio.block_size / self.cfg.audio.sample_rate

        for block in self.mic.blocks():
            if self._stop.is_set():
                return False

            # Half-duplex: never run wake detection on our own voice. While the
            # player is speaking we drain mic blocks without scoring them.
            if self.player.is_speaking:
                self.wake.reset()
                continue

            rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)) + 1e-9)
            peak_rms = max(peak_rms, rms)
            silent_for = silent_for + block_dur if rms < 8.0 else 0.0

            name = self.wake.process(block)
            scores = self.wake.last_scores
            if scores:
                peak_score = max(peak_score, max(scores.values()))
            if name:
                self._on_wake()
                return True

            # Periodic status so you can see the mic is alive and how close the
            # wake score is getting. Also warns about a silent (permission?) mic.
            now = time.time()
            if now - last_status >= 3.0:
                log.info(
                    "Listening... mic level=%.0f  best wake score=%.2f (need >= %.2f)",
                    peak_rms, peak_score, self.cfg.wake_word.threshold,
                )
                if silent_for >= 3.0 and not warned_silence:
                    log.warning(
                        "Microphone appears SILENT. On macOS, grant microphone "
                        "access to your terminal/IDE: System Settings > Privacy & "
                        "Security > Microphone. Then restart. (Try `python run.py "
                        "--mic-test` to verify.)"
                    )
                    warned_silence = True
                last_status, peak_score, peak_rms = now, 0.0, 0.0
        return False

    def _on_wake(self) -> None:
        log.info("== WAKE ==")
        # Speak the acknowledgement to completion BEFORE we start recording, then
        # flush the mic, so we don't record (and transcribe) our own "Yes?".
        self._speak("Yes?", blocking=True)
        self.mic.flush()

    def _handle_interaction(self, start_timeout_ms: Optional[int] = None,
                            prompt_on_unclear: bool = True) -> bool:
        """Record -> transcribe -> respond. Returns True if a question was answered."""
        # 1) record the question
        log.info("Listening for your question...")
        audio = self.recorder.record(self.mic, start_timeout_ms=start_timeout_ms)
        if audio is None or len(audio) == 0:
            return False

        # 2) transcribe
        text = self.stt.transcribe(audio, self.cfg.audio.sample_rate) if (self.stt and self.stt.available) else ""
        if not text.strip():
            # Empty = silence / noise / a filtered hallucination. Don't respond.
            if prompt_on_unclear:
                self._say("I didn't quite catch that.")
            return False
        log.info("You: %s", text)

        # 3) think + speak (streaming)
        self._respond_streaming(text)
        return True

    def _conversation(self) -> None:
        """Handle the first question, then keep listening for follow-ups without
        requiring the wake word again, until you go quiet."""
        if not self._handle_interaction(prompt_on_unclear=True):
            log.info("No question captured; back to standby.")
            return

        conv = self.cfg.conversation
        if not conv.followup_enabled:
            return

        followups = 0
        while not self._stop.is_set():
            if conv.max_followups and followups >= conv.max_followups:
                break
            log.info("Listening for a follow-up (no wake word needed)...")
            self.mic.flush()
            if not self._handle_interaction(
                start_timeout_ms=conv.followup_window_ms,
                prompt_on_unclear=False,
            ):
                log.info("No follow-up; returning to standby. Say the wake word to resume.")
                break
            followups += 1

    # ------------------------------------------------------------- speaking
    def _respond_streaming(self, text: str) -> None:
        self._first_token_seen.clear()
        self._arm_filler()

        def on_first_token() -> None:
            self._first_token_seen.set()
            self._cancel_filler()

        def on_tool_call(_name: str) -> None:
            # A tool call means a longer wait — cover it with a tool filler.
            self._cancel_filler()
            if not self.player.is_speaking:
                self._speak(random_tool_filler(), blocking=False)

        spoken_any = False
        for sentence in self.brain.respond(text, on_first_token=on_first_token, on_tool_call=on_tool_call):
            log.info("House: %s", sentence)
            self._speak(sentence, blocking=False)
            spoken_any = True

        self._cancel_filler()
        if not spoken_any:
            self._speak("I have nothing to add.", blocking=False)
        self.player.mark_end()
        # Half-duplex: wait until we've finished talking, then throw away any
        # audio we captured of our OWN voice before we start listening again.
        self.player.wait_until_done()
        self.mic.flush()

    def _say(self, text: str) -> None:
        """Speak to completion and then discard self-captured audio."""
        self._speak(text, blocking=True)
        self.mic.flush()

    def _speak(self, text: str, blocking: bool = True) -> None:
        """Synthesize -> FX -> queue for playback."""
        if not text.strip():
            return
        if self.tts is None or not self.tts.available:
            log.info("[TTS unavailable] %s", text)
            return
        # Hold the lock across synth + enqueue so concurrent callers (filler
        # timer vs. worker) can't interleave or reorder their audio. Whoever
        # gets here first is fully queued before the next caller proceeds.
        with self._tts_lock:
            audio = self.tts.synthesize(text)
            if audio is None:
                return
            audio = self.fx.apply(audio)
            self.player.play(audio)
        if blocking:
            self.player.wait_until_done()

    # -------------------------------------------------------------- fillers
    def _arm_filler(self) -> None:
        self._cancel_filler()
        delay = max(0.05, self.cfg.brain.filler_after_ms / 1000.0)

        def fire() -> None:
            if not self._first_token_seen.is_set() and not self._stop.is_set():
                line = random_filler()
                log.info("House (filler): %s", line)
                self._speak(line, blocking=False)

        self._filler_timer = threading.Timer(delay, fire)
        self._filler_timer.daemon = True
        self._filler_timer.start()

    def _cancel_filler(self) -> None:
        if self._filler_timer is not None:
            self._filler_timer.cancel()
            self._filler_timer = None

