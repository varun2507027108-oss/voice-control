"""
Cross-Platform OS Automation Controllers for EchoFlux.
Supports Windows (pycaw, ctypes, startfile), macOS (osascript), and Linux (pactl, xdg-open).
"""

import sys
import os
import re
import platform
import logging
import subprocess
import webbrowser
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger("EchoFlux.Actions")

# Try to import pyautogui, fallback safely if headless/missing
try:
    import pyautogui
    pyautogui.FAILSAFE = False
except Exception as e:
    logger.warning("pyautogui not available: %s", e)
    pyautogui = None

# Windows Audio Controller using pycaw
_pycaw_volume = None
if sys.platform == "win32":
    try:
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        spk = AudioUtilities.GetSpeakers()
        if hasattr(spk, "EndpointVolume"):
            _pycaw_volume = spk.EndpointVolume
        elif hasattr(spk, "Activate"):
            from comtypes import CLSCTX_ALL
            interface = spk.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            _pycaw_volume = interface.QueryInterface(IAudioEndpointVolume)
    except Exception as e:
        logger.warning("pycaw initialization failed: %s; falling back to keyboard media keys", e)


class SystemActions:
    """Unified system execution manager for desktop voice actions."""

    # Common Windows application aliases mapping to executable names or URI schemes
    APP_ALIASES: Dict[str, str] = {
        "vs code": "code",
        "vscode": "code",
        "visual studio code": "code",
        "spotify": "spotify",
        "discord": "discord",
        "terminal": "wt.exe",
        "windows terminal": "wt.exe",
        "command prompt": "cmd.exe",
        "powershell": "powershell.exe",
        "slack": "slack",
        "chrome": "chrome",
        "google chrome": "chrome",
        "firefox": "firefox",
        "edge": "msedge",
        "microsoft edge": "msedge",
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "calc": "calc.exe",
        "explorer": "explorer.exe",
        "file explorer": "explorer.exe",
        "task manager": "taskmgr.exe",
        "settings": "ms-settings:",
    }

    # -------------------------------------------------------------
    # Volume Control
    # -------------------------------------------------------------
    @classmethod
    def get_volume(cls) -> float:
        """Get current master volume as percentage (0-100)."""
        if sys.platform == "win32" and _pycaw_volume:
            try:
                scalar = _pycaw_volume.GetMasterVolumeLevelScalar()
                return round(scalar * 100.0, 1)
            except Exception as e:
                logger.error("Error getting volume via pycaw: %s", e)
        return 50.0

    @classmethod
    def volume_set(cls, level_percent: float) -> Tuple[bool, str]:
        """Set volume to specific percentage (0 to 100)."""
        level = max(0.0, min(100.0, float(level_percent)))
        
        if sys.platform == "win32" and _pycaw_volume:
            try:
                _pycaw_volume.SetMasterVolumeLevelScalar(level / 100.0, None)
                return True, f"🔊 Volume set to {int(level)}%"
            except Exception as e:
                logger.error("Failed to set volume via pycaw: %s", e)

        elif sys.platform == "darwin":
            try:
                subprocess.run(["osascript", "-e", f"set volume output volume {int(level)}"], check=True)
                return True, f"🔊 Volume set to {int(level)}%"
            except Exception as e:
                logger.error("macOS volume set error: %s", e)

        elif sys.platform.startswith("linux"):
            try:
                subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{int(level)}%"], check=True)
                return True, f"🔊 Volume set to {int(level)}%"
            except Exception as e:
                logger.error("Linux volume set error: %s", e)

        # Fallback simulation
        return False, f"Volume set to {int(level)}% (hardware controller unavailable)"

    @classmethod
    def volume_up(cls, step: float = 10.0) -> Tuple[bool, str]:
        """Increase system volume by step percentage."""
        current = cls.get_volume()
        new_vol = min(100.0, current + step)
        if _pycaw_volume or sys.platform in ("darwin", "linux"):
            return cls.volume_set(new_vol)
        elif pyautogui:
            pyautogui.press("volumeup")
            return True, f"🔊 Volume +{int(step)}%"
        return False, "Failed to raise volume"

    @classmethod
    def volume_down(cls, step: float = 10.0) -> Tuple[bool, str]:
        """Decrease system volume by step percentage."""
        current = cls.get_volume()
        new_vol = max(0.0, current - step)
        if _pycaw_volume or sys.platform in ("darwin", "linux"):
            return cls.volume_set(new_vol)
        elif pyautogui:
            pyautogui.press("volumedown")
            return True, f"🔉 Volume -{int(step)}%"
        return False, "Failed to lower volume"

    @classmethod
    def volume_mute(cls, toggle: bool = True) -> Tuple[bool, str]:
        """Mute / Unmute audio."""
        if sys.platform == "win32" and _pycaw_volume:
            try:
                is_muted = _pycaw_volume.GetMute()
                new_state = not is_muted if toggle else 1
                _pycaw_volume.SetMute(new_state, None)
                msg = "🔇 Audio Muted" if new_state else "🔊 Audio Unmuted"
                return True, msg
            except Exception as e:
                logger.error("pycaw mute error: %s", e)

        elif pyautogui:
            pyautogui.press("volumemute")
            return True, "🔇 Audio Mute Toggled"
        return False, "Could not toggle mute"

    # -------------------------------------------------------------
    # Application & Tab Management
    # -------------------------------------------------------------
    @classmethod
    def open_app(cls, app_name: str) -> Tuple[bool, str]:
        """Launch or focus a desktop application by name."""
        clean_name = app_name.strip().lower()
        if not clean_name:
            return False, "Empty app name"

        target = cls.APP_ALIASES.get(clean_name, clean_name)

        try:
            if sys.platform == "win32":
                # Special cases for URI schemes or direct executables
                if target.endswith(":") or target.endswith(".exe"):
                    os.startfile(target)
                else:
                    # Try direct spawn or explorer start
                    try:
                        subprocess.Popen([target], shell=True)
                    except Exception:
                        os.startfile(target)
                return True, f"⚡ Opened {app_name.title()}"

            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-a", app_name])
                return True, f"⚡ Opened {app_name.title()}"

            elif sys.platform.startswith("linux"):
                subprocess.Popen([target], shell=True)
                return True, f"⚡ Opened {app_name.title()}"

        except Exception as e:
            logger.error("Failed to launch app '%s': %s", app_name, e)
            return False, f"Failed to open {app_name}"

        return False, f"Cannot open {app_name}"

    @classmethod
    def open_browser_tab(cls, query_or_url: str) -> Tuple[bool, str]:
        """Open a website or run a search query in a new browser tab."""
        target = query_or_url.strip()
        if not target:
            return False, "Empty URL/Query"

        # Check if it looks like a URL or domain
        is_url = bool(re.match(r"^(https?://|www\.)", target, re.IGNORECASE)) or (
            "." in target and " " not in target
        )

        if is_url:
            if not target.startswith("http://") and not target.startswith("https://"):
                url = f"https://{target}"
            else:
                url = target
            display_msg = f"🌐 Opened {target}"
        else:
            # Search query
            import urllib.parse
            url = f"https://www.google.com/search?q={urllib.parse.quote_plus(target)}"
            display_msg = f"🔍 Searched: {target[:25]}"

        try:
            webbrowser.open_new_tab(url)
            return True, display_msg
        except Exception as e:
            logger.error("Failed to open browser tab: %s", e)
            return False, "Failed to open browser tab"

    @classmethod
    def close_tab(cls) -> Tuple[bool, str]:
        """Close current browser/editor tab via hotkey."""
        if not pyautogui:
            return False, "pyautogui required for tab control"
        key = "command" if sys.platform == "darwin" else "ctrl"
        pyautogui.hotkey(key, "w")
        return True, "❌ Closed Tab"

    @classmethod
    def next_tab(cls) -> Tuple[bool, str]:
        """Switch to next tab."""
        if not pyautogui:
            return False, "pyautogui required"
        key = "command" if sys.platform == "darwin" else "ctrl"
        pyautogui.hotkey(key, "tab")
        return True, "➡️ Next Tab"

    @classmethod
    def prev_tab(cls) -> Tuple[bool, str]:
        """Switch to previous tab."""
        if not pyautogui:
            return False, "pyautogui required"
        key = "command" if sys.platform == "darwin" else "ctrl"
        pyautogui.hotkey(key, "shift", "tab")
        return True, "⬅️ Prev Tab"

    # -------------------------------------------------------------
    # Media & Utility Shortcuts
    # -------------------------------------------------------------
    @classmethod
    def media_play_pause(cls) -> Tuple[bool, str]:
        """Play/Pause media playback."""
        if pyautogui:
            pyautogui.press("playpause")
            return True, "⏯️ Media Play/Pause"
        return False, "Keyboard simulation unavailable"

    @classmethod
    def media_next(cls) -> Tuple[bool, str]:
        """Skip to next media track."""
        if pyautogui:
            pyautogui.press("nexttrack")
            return True, "⏭️ Next Track"
        return False, "Keyboard simulation unavailable"

    @classmethod
    def take_screenshot(cls) -> Tuple[bool, str]:
        """Capture entire screen and save to user's pictures/screenshots folder."""
        import datetime
        try:
            home = Path.home()
            save_dir = home / "Pictures" / "Screenshots"
            if not save_dir.exists():
                save_dir = home / "Pictures"
                if not save_dir.exists():
                    save_dir = Path("./screenshots")
                    save_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = save_dir / f"EchoFlux_{timestamp}.png"

            if pyautogui:
                img = pyautogui.screenshot()
                img.save(str(filename))
                return True, f"📸 Saved {filename.name}"
            else:
                return False, "Screenshot tool unavailable"
        except Exception as e:
            logger.error("Screenshot capture error: %s", e)
            return False, "Failed to capture screenshot"

    @classmethod
    def lock_workstation(cls) -> Tuple[bool, str]:
        """Lock desktop workstation."""
        try:
            if sys.platform == "win32":
                import ctypes
                ctypes.windll.user32.LockWorkStation()
                return True, "🔒 Locked Workstation"
            elif sys.platform == "darwin":
                subprocess.run(
                    ["/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession", "-suspend"],
                    check=True,
                )
                return True, "🔒 Locked Workstation"
            elif sys.platform.startswith("linux"):
                subprocess.run(["xdg-screensaver", "lock"], check=True)
                return True, "🔒 Locked Workstation"
        except Exception as e:
            logger.error("Lock workstation error: %s", e)
            return False, "Failed to lock workstation"

        return False, "Unsupported platform for locking"

    # -------------------------------------------------------------
    # Action Dispatcher
    # -------------------------------------------------------------
    @classmethod
    def execute(cls, intent: str, slots: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """Dispatch action based on intent and extracted slots."""
        slots = slots or {}
        intent = intent.lower()

        if intent == "volume_up":
            step = float(slots.get("step", 10.0))
            return cls.volume_up(step)

        elif intent == "volume_down":
            step = float(slots.get("step", 10.0))
            return cls.volume_down(step)

        elif intent == "volume_mute":
            return cls.volume_mute(toggle=True)

        elif intent == "volume_set":
            level = float(slots.get("percentage", 50.0))
            return cls.volume_set(level)

        elif intent == "open_app":
            app_name = slots.get("app_name", "")
            return cls.open_app(app_name)

        elif intent == "open_browser_tab":
            target = slots.get("target", slots.get("query", ""))
            return cls.open_browser_tab(target)

        elif intent == "close_tab":
            return cls.close_tab()

        elif intent == "next_tab":
            return cls.next_tab()

        elif intent == "prev_tab":
            return cls.prev_tab()

        elif intent == "media_play_pause":
            return cls.media_play_pause()

        elif intent == "media_next":
            return cls.media_next()

        elif intent == "take_screenshot":
            return cls.take_screenshot()

        elif intent == "lock_workstation":
            return cls.lock_workstation()

        elif intent == "greeting":
            return True, "👋 Hello! Echo is listening..."

        elif intent == "noop":
            return False, "No-op"

        return False, f"Unknown intent '{intent}'"
