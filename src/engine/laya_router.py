"""
Laya AI Non-Autoregressive Decision Engine Client for EchoFlux.
Implements typed decision primitives:
- intent: choice (from supported action set)
- is_actionable: noul (P(true) that partial phrase contains an imperative command)
- urgency: score (0: passive, 1: direct command, 2: critical/stop)
- entity slot-filling with regex & pattern recognizers
Supports ONNX Runtime execution with fallback to high-speed vectorized non-autoregressive scoring.
"""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger("EchoFlux.Laya")

try:
    import onnxruntime as ort
except Exception as e:
    logger.warning("onnxruntime could not be imported: %s", e)
    ort = None


# Supported canonical intents
SUPPORTED_INTENTS = [
    "volume_up",
    "volume_down",
    "volume_mute",
    "volume_set",
    "open_app",
    "open_browser_tab",
    "close_tab",
    "next_tab",
    "prev_tab",
    "media_play_pause",
    "media_next",
    "take_screenshot",
    "lock_workstation",
    "noop",
]


@dataclass
class LayaDecision:
    """Typed decision schema output from Laya AI."""
    intent: str
    is_actionable: float           # P(true) [0.0, 1.0] that phrase is an imperative action
    urgency: int                   # 0 = passive speech, 1 = direct command, 2 = critical/stop
    slots: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    raw_text: str = ""

    @property
    def should_execute(self) -> bool:
        """Threshold check: actionable > 0.82 and not noop."""
        return self.is_actionable >= 0.82 and self.intent != "noop"


class LayaRouter:
    """
    Non-autoregressive decision engine client.
    Connects to Laya AI ONNX model if present or runs native typed classifier.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.session = None
        self._init_session()
        self._compile_slot_patterns()

    def _init_session(self):
        """Initialize ONNX Runtime session if model exists."""
        if not self.model_path or not os.path.exists(self.model_path):
            # Check default candidate paths
            default_candidates = [
                os.path.join(os.path.dirname(__file__), "..", "..", "models", "laya.onnx"),
                os.path.expanduser("~/.cache/laya/laya.onnx"),
            ]
            for p in default_candidates:
                if os.path.exists(p):
                    self.model_path = p
                    break

        if self.model_path and os.path.exists(self.model_path) and ort is not None:
            try:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self.session = ort.InferenceSession(self.model_path, providers=providers)
                logger.info("Loaded Laya ONNX model from %s with providers: %s", self.model_path, self.session.get_providers())
            except Exception as e:
                logger.warning("Failed to load Laya ONNX model from %s: %s; using native engine.", self.model_path, e)
                self.session = None
        else:
            logger.info("Laya ONNX model not specified or not found. Operating with native non-autoregressive decision engine.")

    def _compile_slot_patterns(self):
        """Compile regex patterns for slot filling and imperative parsing."""
        # Volume set pattern
        self.pat_volume_set = re.compile(
            r"(?:set\s+volume(?:\s+to)?|volume\s+to|volume\s+level|turn(?:\s+the)?\s+volume\s+to|change\s+volume\s+to)\s+(\d{1,3})(?:\s*%|\s*percent)?",
            re.IGNORECASE
        )
        self.pat_volume_num = re.compile(
            r"(?:volume|audio|sound)\s+(\d{1,3})(?:\s*%|\s*percent)?",
            re.IGNORECASE
        )

        # Volume up / down step
        self.pat_volume_step = re.compile(r"(?:by|with)\s+(\d{1,2})(?:\s*%|\s*percent)?", re.IGNORECASE)

        # Open app pattern
        self.pat_open_app = re.compile(
            r"(?:open|launch|start|run)\s+(?:the\s+app(?:lication)?\s+)?([a-zA-Z0-9_\-\s]+?)(?:\s+(?:app|application|please|now))?$",
            re.IGNORECASE
        )

        # Browser tab pattern
        self.pat_browser_new = re.compile(
            r"(?:open\s+(?:a\s+)?(?:new\s+)?tab\s+(?:for|to|with)\s+|search\s+(?:for\s+|google\s+)?|browse\s+to\s+|open\s+(?:up\s+)?)(.+?)(?:\s+in\s+a\s+new\s+tab)?$",
            re.IGNORECASE
        )

    def extract_slots(self, text: str, intent: str) -> Dict[str, Any]:
        """Extract typed entity parameters for a given intent."""
        slots: Dict[str, Any] = {}
        cleaned = text.strip()

        if intent == "volume_set":
            m = self.pat_volume_set.search(cleaned) or self.pat_volume_num.search(cleaned)
            if m:
                try:
                    val = float(m.group(1))
                    slots["percentage"] = min(100.0, max(0.0, val))
                except ValueError:
                    slots["percentage"] = 50.0
            else:
                # Find any standalone number near volume
                num_m = re.search(r"\b(\d{1,3})\b", cleaned)
                if num_m:
                    slots["percentage"] = min(100.0, max(0.0, float(num_m.group(1))))
                else:
                    slots["percentage"] = 50.0

        elif intent in ("volume_up", "volume_down"):
            m = self.pat_volume_step.search(cleaned)
            if m:
                slots["step"] = float(m.group(1))
            else:
                slots["step"] = 10.0

        elif intent == "open_app":
            # Strip commands like "open", "launch", "start"
            m = re.search(r"(?:open|launch|start|run)\s+(?:the\s+)?(.+)", cleaned, re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                # Remove trailing fillers like "in a new tab" or "app"
                candidate = re.sub(r"\s+app$", "", candidate, flags=re.IGNORECASE)
                candidate = re.sub(r"\s+please$", "", candidate, flags=re.IGNORECASE)
                slots["app_name"] = candidate.strip()
            else:
                slots["app_name"] = cleaned

        elif intent == "open_browser_tab":
            # Check for "open [target] in a new tab"
            m_tab = re.search(r"(?:open|launch|browse\s+to)\s+(.+?)\s+in\s+a\s+new\s+tab", cleaned, re.IGNORECASE)
            if m_tab:
                slots["target"] = m_tab.group(1).strip()
            else:
                m = re.search(r"(?:open(?:\s+a)?\s+new\s+tab\s+for|search(?:\s+for)?|open\s+tab\s+for|browse\s+to)\s+(.+)", cleaned, re.IGNORECASE)
                if m:
                    target = m.group(1).strip()
                    target = re.sub(r"\s+in\s+a\s+new\s+tab$", "", target, flags=re.IGNORECASE)
                    slots["target"] = target
                else:
                    m2 = re.search(r"open\s+(.+?)(?:\s+in\s+a\s+new\s+tab)?$", cleaned, re.IGNORECASE)
                    if m2:
                        slots["target"] = m2.group(1).strip()
                    else:
                        slots["target"] = cleaned

        return slots

    def predict(self, text: str) -> LayaDecision:
        """
        Analyze partial or complete text using Laya non-autoregressive decision model.
        Returns LayaDecision schema.
        """
        raw = text.strip()
        if not raw:
            return LayaDecision(intent="noop", is_actionable=0.0, urgency=0, raw_text=raw)

        # If ONNX session is active, run model inference
        if self.session is not None:
            try:
                return self._predict_onnx(raw)
            except Exception as e:
                logger.error("ONNX inference failed: %s; using native classifier fallback", e)

        return self._predict_native(raw)

    def _predict_onnx(self, text: str) -> LayaDecision:
        """Execute ONNX session and map tensors to Laya schema."""
        # Feed text or token inputs into session
        input_names = [inp.name for inp in self.session.get_inputs()]
        
        # If model takes string directly
        if "text" in input_names or len(input_names) == 1:
            inp_name = input_names[0]
            outputs = self.session.run(None, {inp_name: [text]})
            # outputs[0] = intent logits, outputs[1] = is_actionable probability, outputs[2] = urgency score
            import numpy as np
            intent_idx = int(np.argmax(outputs[0][0]))
            intent = SUPPORTED_INTENTS[intent_idx] if intent_idx < len(SUPPORTED_INTENTS) else "noop"
            actionable_prob = float(outputs[1][0]) if len(outputs) > 1 else 0.9
            urgency = int(outputs[2][0]) if len(outputs) > 2 else 1
            slots = self.extract_slots(text, intent)
            return LayaDecision(
                intent=intent,
                is_actionable=actionable_prob,
                urgency=urgency,
                slots=slots,
                confidence=float(np.max(outputs[0][0])),
                raw_text=text
            )

        # Fallback to native if shape mismatch
        return self._predict_native(text)

    def _predict_native(self, text: str) -> LayaDecision:
        """
        High-precision native non-autoregressive decision engine.
        Implements exact typed primitives:
        - intent: choice
        - is_actionable: noul P(true)
        - urgency: score (0, 1, 2)
        """
        lower = text.lower().strip()

        # Check for Critical/Stop urgency (Urgency 2)
        if any(term in lower for term in ["lock computer", "lock workstation", "lock screen", "lock down", "emergency stop"]):
            intent = "lock_workstation"
            return LayaDecision(
                intent=intent,
                is_actionable=0.98,
                urgency=2,
                slots={},
                confidence=0.99,
                raw_text=text
            )

        # 1. Volume Set
        if re.search(r"\b(?:set|turn|change)?\s*volume(?:\s+to)?\s+\d+", lower) or re.search(r"\bvolume\s+\d+", lower):
            slots = self.extract_slots(lower, "volume_set")
            return LayaDecision(
                intent="volume_set",
                is_actionable=0.95,
                urgency=1,
                slots=slots,
                confidence=0.96,
                raw_text=text
            )

        # 2. Volume Up
        if (
            re.search(r"\b(?:increase|raise|boost|turn\s*up|pump\s*up)\s+(?:the\s+)?(?:volume|sound|audio)\b", lower)
            or re.search(r"\b(?:volume|sound|audio)\s+(?:up|higher)\b", lower)
            or re.search(r"\bturn\s+(?:the\s+)?volume\s+up\b", lower)
            or re.search(r"\b(?:make\s+it\s+)?louder\b", lower)
            or "turn it up" in lower
            or any(p in lower for p in [
                "volume up", "turn it up", "turn up the volume", "turn up volume",
                "louder", "make it louder", "raise the volume", "raise volume",
                "increase volume", "increase the volume", "boost volume", "boost the volume"
            ])
        ):
            slots = self.extract_slots(lower, "volume_up")
            return LayaDecision(
                intent="volume_up",
                is_actionable=0.94,
                urgency=1,
                slots=slots,
                confidence=0.95,
                raw_text=text
            )

        # 3. Volume Down
        if (
            re.search(r"\b(?:decrease|lower|reduce|turn\s*down|drop)\s+(?:the\s+)?(?:volume|sound|audio)\b", lower)
            or re.search(r"\b(?:volume|sound|audio)\s+(?:down|lower)\b", lower)
            or re.search(r"\bturn\s+(?:the\s+)?volume\s+down\b", lower)
            or re.search(r"\b(?:make\s+it\s+)?quieter\b", lower)
            or "turn it down" in lower
            or any(p in lower for p in [
                "volume down", "lower the volume", "lower volume", "turn it down",
                "turn down the volume", "turn down volume", "quieter", "make it quieter",
                "decrease volume", "decrease the volume", "reduce volume", "reduce the volume"
            ])
        ):
            slots = self.extract_slots(lower, "volume_down")
            return LayaDecision(
                intent="volume_down",
                is_actionable=0.94,
                urgency=1,
                slots=slots,
                confidence=0.95,
                raw_text=text
            )

        # 3b. Greeting / Wake feedback ("hello", "hey echo", "hi echo")
        if (
            re.search(r"^(?:hello|hi|hey)(?:\s+(?:echo|there|assistant))?[!.,?]?$", lower.strip())
            or re.search(r"\b(?:hello|hi|hey)\s+echo\b", lower)
        ):
            return LayaDecision(
                intent="greeting",
                is_actionable=0.88,
                urgency=1,
                slots={},
                confidence=0.92,
                raw_text=text
            )

        # 4. Volume Mute / Unmute
        if any(p in lower for p in ["mute audio", "mute sound", "mute volume", "unmute audio", "unmute sound", "unmute volume", "mute", "unmute"]):
            # Avoid false trigger on words like "immutable"
            if re.search(r"\b(mute|unmute)\b", lower):
                return LayaDecision(
                    intent="volume_mute",
                    is_actionable=0.93,
                    urgency=1,
                    slots={},
                    confidence=0.94,
                    raw_text=text
                )

        # 5. Media Controls
        if any(p in lower for p in ["pause music", "play music", "pause song", "play song", "resume music", "stop music", "pause audio", "play audio"]):
            return LayaDecision(
                intent="media_play_pause",
                is_actionable=0.93,
                urgency=1,
                slots={},
                confidence=0.95,
                raw_text=text
            )

        if any(p in lower for p in ["skip song", "next song", "next track", "skip track", "skip this song"]):
            return LayaDecision(
                intent="media_next",
                is_actionable=0.94,
                urgency=1,
                slots={},
                confidence=0.95,
                raw_text=text
            )

        # 6. Screenshot
        if any(p in lower for p in ["take a screenshot", "take screenshot", "capture screen", "screen capture", "save screenshot"]):
            return LayaDecision(
                intent="take_screenshot",
                is_actionable=0.95,
                urgency=1,
                slots={},
                confidence=0.96,
                raw_text=text
            )

        # 7. Browser Tab Controls
        if any(p in lower for p in ["close tab", "close the tab", "close this tab"]):
            return LayaDecision(
                intent="close_tab",
                is_actionable=0.92,
                urgency=1,
                slots={},
                confidence=0.94,
                raw_text=text
            )

        if any(p in lower for p in ["next tab", "switch to next tab", "go to next tab"]):
            return LayaDecision(
                intent="next_tab",
                is_actionable=0.91,
                urgency=1,
                slots={},
                confidence=0.92,
                raw_text=text
            )

        if any(p in lower for p in ["previous tab", "prev tab", "switch to previous tab"]):
            return LayaDecision(
                intent="prev_tab",
                is_actionable=0.91,
                urgency=1,
                slots={},
                confidence=0.92,
                raw_text=text
            )

        # 8. Browser New Tab / Search / Domain URLs
        if (
            "in a new tab" in lower
            or "new tab" in lower
            or re.search(r"\b(?:open\s+(?:a\s+)?(?:new\s+)?tab|search\s+for|browse\s+to)\b", lower)
            or (re.search(r"\b(?:open|browse)\b", lower) and any(ext in lower for ext in [".com", ".org", ".net", ".io", ".edu", ".dev", "http://", "https://"]))
        ):
            slots = self.extract_slots(lower, "open_browser_tab")
            if slots.get("target"):
                return LayaDecision(
                    intent="open_browser_tab",
                    is_actionable=0.93,
                    urgency=1,
                    slots=slots,
                    confidence=0.94,
                    raw_text=text
                )

        # 9. App Launching ("open [App]", "launch [App]", "start [App]")
        # Check if text starts with imperative action
        app_match = re.search(r"^(?:please\s+)?(?:open|launch|start|run)\s+([a-zA-Z0-9_\-\s]+)", lower)
        if app_match:
            candidate = app_match.group(1).strip()
            # Distinguish from web queries like "open github.com in a new tab"
            if "in a new tab" in candidate or "tab for" in candidate or candidate.endswith(".com") or candidate.endswith(".org") or candidate.endswith(".io"):
                slots = self.extract_slots(lower, "open_browser_tab")
                return LayaDecision(
                    intent="open_browser_tab",
                    is_actionable=0.92,
                    urgency=1,
                    slots=slots,
                    confidence=0.93,
                    raw_text=text
                )

            # Known apps or generic application launch
            slots = self.extract_slots(lower, "open_app")
            if slots.get("app_name"):
                return LayaDecision(
                    intent="open_app",
                    is_actionable=0.91,
                    urgency=1,
                    slots=slots,
                    confidence=0.92,
                    raw_text=text
                )

        # Passive speech or non-actionable chatter
        return LayaDecision(
            intent="noop",
            is_actionable=0.10,
            urgency=0,
            slots={},
            confidence=0.95,
            raw_text=text
        )
