"""
Mid-sentence stream dispatcher and token deduplication engine for EchoFlux.
Consumes rolling partial STT hypotheses, slices unconsumed phrases, and triggers Laya AI decisions mid-sentence.
"""

import sys
import time
import logging
from typing import Callable, Optional, Dict, Any, List

from src.engine.laya_router import LayaRouter, LayaDecision
from src.actions.system_actions import SystemActions

logger = logging.getLogger("EchoFlux.Dispatcher")

# Optional audio chime
def play_chime():
    """Produce subtle audio feedback on action trigger."""
    try:
        if sys.platform == "win32":
            import winsound
            winsound.Beep(880, 45)  # 880 Hz for 45 ms (A5 note)
    except Exception as e:
        logger.debug("Chime playback error: %s", e)


class StreamDispatcher:
    """
    Sliding-window stream processor for real-time mid-sentence command execution.
    """

    def __init__(
        self,
        router: Optional[LayaRouter] = None,
        action_callback: Optional[Callable[[str, int, str], None]] = None,
        transcript_callback: Optional[Callable[[str, str], None]] = None,
        level_callback: Optional[Callable[[float], None]] = None,
        action_threshold: float = 0.82,
        debounce_seconds: float = 0.6,
    ):
        self.router = router or LayaRouter()
        self.action_callback = action_callback
        self.transcript_callback = transcript_callback
        self.level_callback = level_callback
        self.action_threshold = action_threshold
        self.debounce_seconds = debounce_seconds

        self.buffer_text: str = ""
        self.consumed_index: int = 0
        self.last_action_time: float = 0.0
        self.last_action_intent: str = ""
        self.action_history: List[Dict[str, Any]] = []

    def reset(self):
        """Reset the consumer buffer and index (e.g. on prolonged silence)."""
        self.buffer_text = ""
        self.consumed_index = 0

    def process_partial(self, transcript: str, rms_level: float = 0.0):
        """
        Main entry point called on every partial hypothesis from STT engine (~80-120ms).
        """
        # Pass audio energy level to visualizer
        if self.level_callback:
            self.level_callback(rms_level)

        transcript = transcript.strip()
        if not transcript:
            if self.transcript_callback:
                self.transcript_callback("", "")
            return

        # If transcript shrunk or reset significantly, reset consumed index
        if len(transcript) < self.consumed_index:
            self.consumed_index = 0

        self.buffer_text = transcript

        # Slice unconsumed segment
        active_phrase = transcript[self.consumed_index:].strip()

        if self.transcript_callback:
            self.transcript_callback(transcript, active_phrase)

        if not active_phrase or len(active_phrase) < 3:
            return

        # Pass active phrase to Laya AI Decision Engine
        decision: LayaDecision = self.router.predict(active_phrase)

        # Check actionability threshold and non-noop intent
        if decision.is_actionable >= self.action_threshold and decision.intent != "noop":
            now = time.time()
            # Debounce check to avoid duplicate execution within window
            if (
                decision.intent == self.last_action_intent
                and (now - self.last_action_time) < self.debounce_seconds
            ):
                logger.debug("Debounced duplicate intent: %s", decision.intent)
                return

            self.last_action_time = now
            self.last_action_intent = decision.intent

            # 1. Dispatch system action immediately
            success, msg = SystemActions.execute(decision.intent, decision.slots)
            logger.info("⚡ Mid-sentence Trigger: '%s' -> %s (Success: %s, Message: %s)", active_phrase, decision.intent, success, msg)

            # 2. Advance consumed index to current transcript length to prevent re-triggering
            self.consumed_index = len(transcript)

            # 3. Play subtle audio chime
            play_chime()

            # 4. Notify UI overlay with action badge
            badge_text = msg if success else f"⚠️ {decision.intent.replace('_', ' ').title()}"
            if self.action_callback:
                self.action_callback(badge_text, decision.urgency, decision.intent)

            self.action_history.append({
                "time": now,
                "phrase": active_phrase,
                "intent": decision.intent,
                "badge": badge_text,
                "success": success
            })
