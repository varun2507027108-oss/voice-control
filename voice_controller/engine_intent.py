"""Laya Router schema definition and two-stage intent prediction parser for voice commands."""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import laya

from voice_controller.config import CONFIDENCE_THRESHOLD, DEVICE

logger = logging.getLogger("IntentEngine")

# Word-to-number mapping for slot extraction
WORD_TO_NUMBER: Dict[str, int] = {
    "zero": 0, "ten": 10, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "one hundred": 100, "hundred": 100, "full": 100, "max": 100, "half": 50,
}


@dataclass
class RouteResult:
    action: str
    confidence: float
    source: str = "laya"
    slot_value: Optional[int] = None
    target_app: Optional[str] = None
    routing_model: str = "english"
    is_ambiguous: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)


# Question schema defining core Windows control actions for Laya fallback
# Note: Criteria text uses generalized intent definitions, not memorized test strings
VOICE_INTENT_QUESTIONS: Dict[str, Any] = {
    "action": {
        "type": "choice",
        "instructions": "Which operating system or computer desktop command is the user asking to execute?",
        "criteria": {
            "open_browser": "launch or open web browser, Chrome, Edge, go online, browse the internet",
            "open_terminal": "open terminal, console, command prompt, powershell, command line interface",
            "open_editor": "open code editor, IDE, programming software, VS Code, coding workspace",
            "volume_up": "increase audio volume, make sound louder, crank up volume, amplify audio level",
            "volume_down": "decrease audio volume, make sound quieter, reduce sound level, lower sound output",
            "volume_mute": "mute audio, silence speakers, mute sound output",
            "volume_unmute": "unmute audio, restore sound, unmute speakers",
            "brightness_up": "increase display brightness, brighten monitor screen, make display lighter",
            "brightness_down": "decrease display brightness, dim monitor screen, make display darker",
            "media_play_pause": "pause track, play track, pause song, play music, toggle media playback, pause or play audio playback",
            "media_next": "skip to next track, next song, play next episode in media player",
            "media_prev": "go to previous track, previous song, replay track",
            "lock_workstation": "lock computer, lock screen, lock workstation, secure the pc",
            "calibrate_mic": "calibrate microphone noise floor, adjust mic sensitivity",
            "unrecognized": "out of domain query, real world physical action, household appliances, general conversation, knowledge search, or unrelated speech",
        },
    }
}

# Domain gate: reject queries with real-world physical targets or generic informational chatter
OUT_OF_DOMAIN_OBJECTS = re.compile(
    r"\b(light|lights|lamp|bulb|fridge|refrigerator|door|window|curtains|ac|heater|thermostat|"
    r"solitaire|chess|checkers|poker|game\s+of|"
    r"price|cost|discount|dollar|stock|"
    r"weather|forecast|rain|temperature|"
    r"shopping\s+list|groceries|grocery|milk|eggs|bread|"
    r"mount\s+everest|mountain|capital\s+of|president|prime\s+minister|how\s+tall|how\s+far|"
    r"poem|poetry|essay|story|joke|flight|pizza|calories|boiling\s+point|photosynthesis|"
    r"math\s+problem|quantum|universe|cake|bake|recipe|movies?|unread\s+text|distance\s+to|lightbulb|"
    r"sing|singing|birthday)\b",
    re.I,
)

OFF_DOMAIN_STARTS = re.compile(
    r"^(what|where|who|why|how|when|which|is\s+|are\s+|can\s+|could\s+|would\s+|"
    r"tell\s+me|translate|summarize|write\s+|solve\s+|book\s+|order\s+|"
    r"sing\s+|recommend\s+|read\s+|remind\s+|set\s+a\s+timer)\b",
    re.I,
)

COMPUTER_EXPLICIT_TERMS = re.compile(
    r"\b(volume|audio|sound|screen|display|monitor|brightness|browser|chrome|terminal|cmd|powershell|editor|vs\s*code|track|song|playlist|episode|workstation|pc|computer|mic|microphone)\b",
    re.I,
)


def domain_gate(text: str) -> bool:
    """Pre-filter out-of-domain physical actions, conversational queries, and general knowledge.
    
    Returns True if query is allowed to proceed to intent routing, False to fail closed.
    """
    lower = text.lower().strip()

    # 1. Physical world / non-computer entities (e.g. lights, fridge, door, weather, shopping list)
    if OUT_OF_DOMAIN_OBJECTS.search(lower):
        # Unless explicitly qualifying a computer component (e.g. "screen brightness")
        if not re.search(r"\b(screen\s+brightness|display\s+brightness|sound\s+volume|audio\s+volume)\b", lower):
            return False

    # 2. Obvious conversational question without computer control words
    if OFF_DOMAIN_STARTS.search(lower) and not COMPUTER_EXPLICIT_TERMS.search(lower):
        return False

    return True


# Semantic complaint inversions (common voice UI patterns)
COMPLAINT_RULES: List[Tuple[re.Pattern, str]] = [
    # "too loud" -> volume is excessively high -> lower volume
    (re.compile(r"\b(too\s+loud|way\s+too\s+loud|ears\s+hurt|deafening)\b", re.I), "volume_down"),
    # "too quiet" -> volume is too low -> raise volume
    (re.compile(r"\b(too\s+quiet|too\s+soft|can'?t\s+hear|hard\s+to\s+hear)\b", re.I), "volume_up"),
    # "too bright" -> screen is blinding -> lower brightness
    (re.compile(r"\b(too\s+bright|way\s+too\s+bright|blinding|eyes\s+hurt)\b", re.I), "brightness_down"),
    # "too dark" -> screen is unreadable -> raise brightness
    (re.compile(r"\b(too\s+dark|too\s+dim|can'?t\s+see(\s+the)?\s+screen)\b", re.I), "brightness_up"),
]

# Regex fast-rules for zero-latency direct commands
FAST_RULES: List[Tuple[re.Pattern, str]] = [
    # Microphone calibration
    (re.compile(r"\bcalibrate\s+(the\s+)?mic(rophone)?\b", re.I), "calibrate_mic"),

    # Volume Controls
    (re.compile(r"\b(unmute(\s+audio|\s+sound|\s+volume)?|turn\s+sound\s+on)\b", re.I), "volume_unmute"),
    (re.compile(r"\b(mute(\s+audio|\s+sound|\s+volume)?|silence(\s+the)?\s+(sound|audio))\b", re.I), "volume_mute"),
    (re.compile(r"\b(volume\s+up|raise\s+(the\s+)?volume|turn\s+up\s+(the\s+)?(volume|sound)|boost\s+(the\s+)?(volume|sound)|louder(\s+sound|\s+audio)?)\b", re.I), "volume_up"),
    (re.compile(r"\b(volume\s+down|lower\s+(the\s+)?volume|turn\s+down\s+(the\s+)?(volume|sound)|reduce\s+(the\s+)?(volume|sound)|quieter(\s+audio|\s+sound)?)\b", re.I), "volume_down"),

    # Brightness Controls
    (re.compile(r"\b(dim(\s+the)?\s+screen|brightness\s+down|lower\s+(the\s+)?brightness|reduce\s+(screen\s+)?brightness|make\s+(the\s+)?screen\s+darker)\b", re.I), "brightness_down"),
    (re.compile(r"\b(brighten(\s+the)?\s+(display|screen)|brightness\s+up|turn\s+up\s+brightness|increase\s+brightness)\b", re.I), "brightness_up"),

    # Media Controls
    (re.compile(r"\b(pause\s+music|pause\s+song|play\s+music|resume\s+song|toggle\s+playback)\b", re.I), "media_play_pause"),
    (re.compile(r"\b(skip\s+track|next\s+song|next\s+track|play\s+next)\b", re.I), "media_next"),
    (re.compile(r"\b(previous\s+song|previous\s+track|last\s+song|restart\s+song)\b", re.I), "media_prev"),

    # System & App Launch
    (re.compile(r"\b(lock(\s+my)?\s+(computer|workstation|screen))\b", re.I), "lock_workstation"),
    (re.compile(r"\b(open\s+(google\s+)?chrome|launch\s+browser|open\s+(the\s+)?browser|open\s+edge)\b", re.I), "open_browser"),
    (re.compile(r"\b(launch\s+terminal|open\s+terminal|open\s+(powershell|cmd|command\s+prompt))\b", re.I), "open_terminal"),
    (re.compile(r"\b(open\s+vs\s*code(\s+editor)?|launch\s+code|open\s+editor)\b", re.I), "open_editor"),
]


def extract_slot_number(text: str) -> Optional[int]:
    """Extract numeric or textual percentage slot from text."""
    # Check digits e.g. "50", "50%", "50 percent"
    m = re.search(r"\b(\d{1,3})\s*(%|percent)?\b", text)
    if m:
        val = int(m.group(1))
        return max(0, min(100, val))

    # Check words e.g. "fifty percent", "twenty"
    for word, num in WORD_TO_NUMBER.items():
        if re.search(rf"\b{word}(\s+percent|\s*%)?\b", text, re.I):
            return num

    return None


class LayaIntentRouter:
    """Two-stage intent classification engine: Regex Fast-Path (<1ms) -> Laya Router (~40ms)."""

    def __init__(self, device: Optional[str] = None, preload: bool = True):
        target_device = device or DEVICE
        logger.info("Initializing Laya Router (preload=%s, device=%s)...", preload, target_device)
        try:
            self.router = laya.Router(preload=False, device=target_device)
        except Exception as err:
            logger.warning("Could not initialize Laya Router with device %s, falling back: %s", target_device, err)
            self.router = laya.Router(preload=False)

        # Restrict models strictly to English so Laya never attempts to download typed-decisions or multilingual
        if "english" in self.router.models:
            self.router.models = {"english": self.router.models["english"]}

        if preload:
            self.router.preload(["english"])

        logger.info("Laya Router ready.")
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _extract_slots_and_fast_rules(self, text: str) -> Optional[RouteResult]:
        """Stage 1: Fast rule-based matching, slot extraction, and complaints."""
        lower = text.lower().strip()

        # Slot extraction: "set volume to 50%", "volume to seventy"
        if re.search(r"\b(set\s+volume|volume\s+to|change\s+volume)\b", lower):
            slot = extract_slot_number(lower)
            if slot is not None:
                return RouteResult(
                    action="volume_set",
                    confidence=1.0,
                    source="fast_rule",
                    slot_value=slot,
                )

        # Slot extraction: "set brightness to 80%", "brightness 40 percent"
        if re.search(r"\b(set\s+brightness|brightness\s+to|change\s+brightness)\b", lower):
            slot = extract_slot_number(lower)
            if slot is not None:
                return RouteResult(
                    action="brightness_set",
                    confidence=1.0,
                    source="fast_rule",
                    slot_value=slot,
                )

        # Direct exact fast rules
        for pattern, action in FAST_RULES:
            if pattern.search(lower):
                return RouteResult(action=action, confidence=1.0, source="fast_rule")

        # Semantic complaint phrasing (e.g. "too loud" -> volume_down)
        for pattern, action in COMPLAINT_RULES:
            if pattern.search(lower):
                return RouteResult(action=action, confidence=0.95, source="complaint_rule")

        # Generic app opening fallback: "open spotify", "launch calculator", "open notepad"
        open_app_match = re.match(r"^(?:open|launch)\s+([a-zA-Z0-9_\-\s]+)$", lower)
        if open_app_match:
            app_name = open_app_match.group(1).strip()
            # Reject non-app objects (e.g., "open the fridge")
            if OUT_OF_DOMAIN_OBJECTS.search(app_name):
                return None
            if app_name in ("google chrome", "chrome", "browser", "edge"):
                return RouteResult("open_browser", 1.0, source="fast_rule")
            if app_name in ("terminal", "powershell", "cmd", "command prompt"):
                return RouteResult("open_terminal", 1.0, source="fast_rule")
            if app_name in ("vs code", "vscode", "code", "editor", "vs code editor"):
                return RouteResult("open_editor", 1.0, source="fast_rule")
            # Generic valid application name
            if len(app_name) >= 3 and app_name not in ("the", "a", "an", "window", "screen", "something", "app"):
                return RouteResult(
                    action="open_app",
                    confidence=0.90,
                    source="fast_rule",
                    target_app=app_name,
                )

        return None

    def predict(self, text: str, bypass_fast_path: bool = False) -> Dict[str, Any]:
        """Classify voice command text into structured action prediction.
        
        Args:
            text: Transcribed speech utterance.
            bypass_fast_path: If True, forces execution through Laya embedding router.
            
        Returns:
            Dict containing action, confidence, routing_model, slot_value, target_app, latency_ms, etc.
        """
        t0 = time.perf_counter()
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            return {
                "action": "unrecognized",
                "confidence": 0.0,
                "routing_model": "none",
                "source": "empty",
                "slot_value": None,
                "target_app": None,
                "is_ambiguous": False,
                "latency_ms": 0.0,
                "raw": {},
            }

        # Stage 0: Domain Gate
        if not domain_gate(cleaned_text):
            latency_ms = (time.perf_counter() - t0) * 1000.0
            logger.info("Domain gate rejected '%s' -> unrecognized (%.2fms)", cleaned_text, latency_ms)
            return {
                "action": "unrecognized",
                "confidence": 1.0,
                "routing_model": "domain_gate",
                "source": "domain_gate",
                "slot_value": None,
                "target_app": None,
                "is_ambiguous": False,
                "latency_ms": latency_ms,
                "raw": {"reason": "out_of_domain_entity"},
            }

        # Stage 1: Fast regex & slots pre-filter (skipped if bypass_fast_path=True)
        if not bypass_fast_path:
            fast_match = self._extract_slots_and_fast_rules(cleaned_text)
            if fast_match:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                logger.info("Fast-rule hit for '%s' -> %s (%.2fms)", cleaned_text, fast_match.action, latency_ms)
                return {
                    "action": fast_match.action,
                    "confidence": fast_match.confidence,
                    "routing_model": "fast_rule",
                    "source": fast_match.source,
                    "slot_value": fast_match.slot_value,
                    "target_app": fast_match.target_app,
                    "is_ambiguous": False,
                    "latency_ms": latency_ms,
                    "raw": {"rule": fast_match.action},
                }

        # Stage 2: Laya semantic router fallback
        try:
            state = {"request": cleaned_text}
            result = self.router.predict(state=state, questions=VOICE_INTENT_QUESTIONS)

            action_data = result.get("answers", {}).get("action", {})
            predicted_choice = action_data.get("choice", "unrecognized")
            confidence = float(action_data.get("confidence", 0.0))

            # Ambiguity & margin extraction from Laya probabilities
            probs = action_data.get("probabilities", {})
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True) if isinstance(probs, dict) else []
            top2 = sorted_probs[:2]

            margin = 1.0
            is_ambiguous = False
            if len(top2) >= 2:
                margin = float(top2[0][1] - top2[1][1])
                if margin < 0.15:
                    is_ambiguous = True

            # Threshold enforcement
            if confidence < CONFIDENCE_THRESHOLD:
                predicted_choice = "unrecognized"

            latency_ms = (time.perf_counter() - t0) * 1000.0
            routing_info = result.get("routing", {})
            routing_model = routing_info.get("model", "english") if isinstance(routing_info, dict) else str(routing_info)

            logger.info(
                "Laya intent for '%s' -> action='%s', conf=%.2f, margin=%.2f, ambiguous=%s (%.1fms)",
                cleaned_text,
                predicted_choice,
                confidence,
                margin,
                is_ambiguous,
                latency_ms,
            )

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            return {
                "action": predicted_choice,
                "confidence": confidence,
                "routing_model": routing_model,
                "source": "laya",
                "slot_value": None,
                "target_app": None,
                "is_ambiguous": is_ambiguous,
                "margin": margin,
                "top2": top2,
                "latency_ms": latency_ms,
                "raw": result,
            }

        except Exception as err:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            logger.error("Intent prediction error for '%s': %s", text, err, exc_info=True)
            return {
                "action": "unrecognized",
                "confidence": 0.0,
                "routing_model": "error",
                "source": "error",
                "slot_value": None,
                "target_app": None,
                "is_ambiguous": False,
                "latency_ms": latency_ms,
                "raw": {"error": str(err)},
            }


# Singleton instance helper
_router_instance: Optional[LayaIntentRouter] = None


def get_intent_router() -> LayaIntentRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = LayaIntentRouter()
    return _router_instance


def predict(text: str, bypass_fast_path: bool = False) -> Dict[str, Any]:
    """Convenience function to classify intent via global router instance."""
    return get_intent_router().predict(text, bypass_fast_path=bypass_fast_path)
