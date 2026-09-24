"""Concrete Windows action executors for Voice Controller with Win32 API injection and thread-safe COM."""

from dataclasses import dataclass, field
import ctypes
import logging
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import webbrowser

logger = logging.getLogger("Actions")

# Win32 Virtual-Key codes for media keys (instant injection, 0ms pause)
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_STOP = 0xB2
VK_MEDIA_PLAY_PAUSE = 0xB3
VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
KEYEVENTF_KEYUP = 0x0002


def _send_vk(vk_code: int):
    """Direct Win32 key event injection to bypass pyautogui's default 100ms pause."""
    try:
        ctypes.windll.user32.keybd_event(vk_code, 0, 0, 0)
        ctypes.windll.user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)
    except Exception as err:
        logger.warning("Virtual key injection failed for 0x%X: %s", vk_code, err)


@dataclass
class ActionResult:
    """Structured result returned by every action execution."""
    success: bool
    message: str
    data: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


class VolumeController:
    """Controls Windows Master Volume using pycaw with thread-safe COM initialization."""

    def __init__(self):
        self._volume_endpoint = None
        self._init_com()

    def _init_com(self):
        """Initialize COM library on current thread."""
        try:
            import comtypes
            comtypes.CoInitialize()
        except Exception:
            pass

    def _get_endpoint(self, force_refresh: bool = False):
        if self._volume_endpoint is not None and not force_refresh:
            return self._volume_endpoint

        self._init_com()
        try:
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

            devices = AudioUtilities.GetSpeakers()
            if devices is None:
                logger.warning("No default audio speaker device found.")
                return None

            if hasattr(devices, "EndpointVolume") and devices.EndpointVolume is not None:
                self._volume_endpoint = devices.EndpointVolume
                return self._volume_endpoint

            from comtypes import CLSCTX_ALL
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self._volume_endpoint = interface.QueryInterface(IAudioEndpointVolume)
            return self._volume_endpoint
        except Exception as err:
            logger.warning("Could not initialize pycaw volume endpoint: %s", err)
            return None

    def get_volume(self) -> Optional[float]:
        endpoint = self._get_endpoint()
        if endpoint is None:
            return None
        try:
            return float(endpoint.GetMasterVolumeLevelScalar())
        except Exception:
            endpoint = self._get_endpoint(force_refresh=True)
            if endpoint:
                try:
                    return float(endpoint.GetMasterVolumeLevelScalar())
                except Exception:
                    pass
            return None

    def set_volume_percent(self, pct: int) -> ActionResult:
        """Set volume to an explicit percentage [0 - 100]."""
        target = max(0.0, min(1.0, pct / 100.0))
        endpoint = self._get_endpoint()
        if endpoint is None:
            return ActionResult(False, "Audio endpoint unavailable", {"volume": None})

        try:
            endpoint.SetMasterVolumeLevelScalar(target, None)
            return ActionResult(True, f"Volume set to {pct}%", {"volume": pct})
        except Exception:
            endpoint = self._get_endpoint(force_refresh=True)
            if endpoint:
                try:
                    endpoint.SetMasterVolumeLevelScalar(target, None)
                    return ActionResult(True, f"Volume set to {pct}%", {"volume": pct})
                except Exception as err:
                    return ActionResult(False, f"Volume set failed: {err}", {})
            return ActionResult(False, "Audio endpoint connection lost", {})

    set_volume = set_volume_percent

    def adjust_volume(self, delta: float) -> ActionResult:
        current = self.get_volume()
        if current is None:
            # Low-latency Win32 fallback
            vk = VK_VOLUME_UP if delta > 0 else VK_VOLUME_DOWN
            for _ in range(3):
                _send_vk(vk)
            direction = "raised" if delta > 0 else "lowered"
            return ActionResult(True, f"Volume {direction}", {})

        new_val = max(0.0, min(1.0, current + delta))
        pct = int(round(new_val * 100))
        return self.set_volume_percent(pct)

    def set_mute(self, target: Optional[bool] = None) -> ActionResult:
        """Set mute explicitly or toggle if target is None."""
        endpoint = self._get_endpoint()
        if endpoint is None:
            _send_vk(VK_VOLUME_MUTE)
            return ActionResult(True, "Mute toggled", {})

        try:
            current_mute = bool(endpoint.GetMute())
            new_mute = (not current_mute) if target is None else target

            if new_mute == current_mute and target is not None:
                # Already in requested state
                state_str = "muted" if new_mute else "unmuted"
                return ActionResult(True, f"Volume already {state_str}", {"muted": new_mute})

            endpoint.SetMute(new_mute, None)
            msg = "Volume muted" if new_mute else "Volume unmuted"
            return ActionResult(True, msg, {"muted": new_mute})
        except Exception:
            endpoint = self._get_endpoint(force_refresh=True)
            if endpoint:
                try:
                    new_mute = not bool(endpoint.GetMute()) if target is None else target
                    endpoint.SetMute(new_mute, None)
                    msg = "Volume muted" if new_mute else "Volume unmuted"
                    return ActionResult(True, msg, {"muted": new_mute})
                except Exception:
                    pass
            _send_vk(VK_VOLUME_MUTE)
            return ActionResult(True, "Mute toggled (key fallback)", {})


class BrightnessController:
    """Controls display screen brightness with screen_brightness_control and error handling."""

    def set_percent(self, target_pct: int) -> ActionResult:
        try:
            import screen_brightness_control as sbc
            target = max(0, min(100, target_pct))
            sbc.set_brightness(target)
            return ActionResult(True, f"Brightness set to {target}%", {"brightness": target})
        except Exception as err:
            logger.warning("Brightness control error: %s", err)
            return ActionResult(False, f"Brightness error (DDC/CI not supported)", {})

    def adjust(self, delta_percent: int) -> ActionResult:
        try:
            import screen_brightness_control as sbc

            current = sbc.get_brightness()
            if isinstance(current, list):
                curr_val = current[0] if current else 50
            else:
                curr_val = current or 50

            target = max(0, min(100, curr_val + delta_percent))
            sbc.set_brightness(target)
            return ActionResult(True, f"Brightness set to {target}%", {"brightness": target})
        except Exception as err:
            logger.warning("Brightness control error: %s", err)
            return ActionResult(False, "Brightness unavailable on display", {})


# Singletons
_volume_ctrl = VolumeController()
_brightness_ctrl = BrightnessController()


def _spawn_detached(exe_path: str, args: Optional[List[str]] = None) -> bool:
    """Launch a process detached from the current console/process tree."""
    cmd = [exe_path] + (args or [])
    try:
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008  # DETACHED_PROCESS = 0x00000008
        subprocess.Popen(
            cmd,
            creationflags=flags,
            close_fds=True,
            shell=False,
        )
        return True
    except Exception as err:
        logger.error("Failed to spawn detached process %s: %s", exe_path, err)
        return False


def action_volume_up(step: float = 0.10) -> ActionResult:
    return _volume_ctrl.adjust_volume(step)


def action_volume_down(step: float = 0.10) -> ActionResult:
    return _volume_ctrl.adjust_volume(-step)


def action_volume_set(pct: int) -> ActionResult:
    return _volume_ctrl.set_volume_percent(pct)


def action_volume_mute() -> ActionResult:
    return _volume_ctrl.set_mute(target=True)


def action_volume_unmute() -> ActionResult:
    return _volume_ctrl.set_mute(target=False)


def action_brightness_up(step: int = 15) -> ActionResult:
    return _brightness_ctrl.adjust(step)


def action_brightness_down(step: int = 15) -> ActionResult:
    return _brightness_ctrl.adjust(-step)


def action_brightness_set(pct: int) -> ActionResult:
    return _brightness_ctrl.set_percent(pct)


def action_open_browser() -> ActionResult:
    chrome_path = shutil.which("chrome")
    if chrome_path and _spawn_detached(chrome_path):
        return ActionResult(True, "Chrome browser launched", {"app": "chrome"})

    edge_path = shutil.which("msedge")
    if edge_path and _spawn_detached(edge_path):
        return ActionResult(True, "Edge browser launched", {"app": "msedge"})

    webbrowser.open("https://www.google.com")
    return ActionResult(True, "Browser opened", {"app": "default_browser"})


def action_open_terminal() -> ActionResult:
    wt_path = shutil.which("wt")
    if wt_path and _spawn_detached(wt_path):
        return ActionResult(True, "Windows Terminal launched", {"app": "wt"})

    pwsh_path = shutil.which("powershell.exe")
    if pwsh_path and _spawn_detached(pwsh_path):
        return ActionResult(True, "PowerShell launched", {"app": "powershell"})

    return ActionResult(False, "Could not locate Terminal or PowerShell", {})


def action_open_editor() -> ActionResult:
    # Look for code or code.cmd in PATH
    code_path = shutil.which("code") or shutil.which("code.cmd")
    if code_path and _spawn_detached(code_path):
        return ActionResult(True, "VS Code launched", {"app": "vscode"})

    notepad_path = shutil.which("notepad.exe")
    if notepad_path and _spawn_detached(notepad_path):
        return ActionResult(True, "Notepad launched", {"app": "notepad"})

    return ActionResult(False, "Could not locate code editor", {})


def action_open_generic_app(app_name: str) -> ActionResult:
    """Attempt to launch an application by name."""
    clean_name = app_name.strip().lower()

    # Map common aliases
    alias_map = {
        "calculator": "calc.exe",
        "calc": "calc.exe",
        "spotify": "spotify.exe",
        "task manager": "taskmgr.exe",
        "settings": "ms-settings:",
        "file explorer": "explorer.exe",
        "explorer": "explorer.exe",
    }

    target = alias_map.get(clean_name, clean_name)

    if target.endswith(":"):  # Windows URI protocol scheme
        try:
            os.startfile(target)
            return ActionResult(True, f"Opened {app_name}", {"app": app_name})
        except Exception as err:
            return ActionResult(False, f"Failed to open {app_name}: {err}", {})

    exe_path = shutil.which(target) or shutil.which(f"{target}.exe")
    if exe_path and _spawn_detached(exe_path):
        return ActionResult(True, f"Launched {app_name}", {"app": app_name})

    # Try os.startfile as fallback for installed Windows apps
    try:
        os.startfile(target)
        return ActionResult(True, f"Launched {app_name}", {"app": app_name})
    except Exception:
        return ActionResult(False, f"Application '{app_name}' not found", {})


def action_media_play_pause() -> ActionResult:
    _send_vk(VK_MEDIA_PLAY_PAUSE)
    return ActionResult(True, "Media play/pause toggled", {})


def action_media_next() -> ActionResult:
    _send_vk(VK_MEDIA_NEXT_TRACK)
    return ActionResult(True, "Next track skipped", {})


def action_media_prev() -> ActionResult:
    _send_vk(VK_MEDIA_PREV_TRACK)
    return ActionResult(True, "Previous track skipped", {})


def action_lock_workstation() -> ActionResult:
    try:
        ctypes.windll.user32.LockWorkStation()
        return ActionResult(True, "Workstation locked", {})
    except Exception as err:
        return ActionResult(False, f"Lock failed: {err}", {})


def action_calibrate_mic() -> ActionResult:
    """Action feedback for microphone calibration."""
    return ActionResult(True, "Microphone noise floor calibrated", {})


def get_microphone_status() -> Dict[str, Any]:
    """Report the Windows default capture endpoint mute/level state for diagnostics.

    A muted or zero-gain microphone is a common "the app can't hear me" cause that is
    completely invisible to PortAudio (the stream opens fine and captures silence).

    Returns ``{"available", "muted", "level_pct", "device_id", "error"}``; never raises.
    """
    status_info: Dict[str, Any] = {
        "available": False,
        "muted": False,
        "level_pct": 100.0,
        "device_id": "",
        "error": None,
    }
    try:
        import comtypes
        comtypes.CoInitialize()
    except Exception:
        pass

    try:
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        device = AudioUtilities.GetMicrophone()
        if device is None:
            status_info["error"] = "no default capture endpoint"
            return status_info

        try:
            status_info["device_id"] = str(device.GetId())
        except Exception:
            pass

        interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        endpoint = interface.QueryInterface(IAudioEndpointVolume)
        status_info["muted"] = bool(endpoint.GetMute())
        status_info["level_pct"] = round(float(endpoint.GetMasterVolumeLevelScalar()) * 100.0, 1)
        status_info["available"] = True
    except Exception as err:
        status_info["error"] = str(err)

    return status_info


def set_microphone_level(percent: float) -> ActionResult:
    """Set the default capture endpoint volume (the Windows 'Microphone volume' slider).

    The target machine measured 27.5%, which throws away roughly 11 dB of input level and is
    the single largest remaining accuracy factor. This mirrors Settings > Sound > Input >
    Volume and is only invoked when the user opts in via ``VOICE_CONTROL_MIC_LEVEL``.
    """
    try:
        import comtypes
        comtypes.CoInitialize()
    except Exception:
        pass

    target = max(0.0, min(1.0, float(percent) / 100.0))
    try:
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        device = AudioUtilities.GetMicrophone()
        if device is None:
            return ActionResult(False, "No default capture endpoint found", {})

        interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        endpoint = interface.QueryInterface(IAudioEndpointVolume)
        previous_pct = round(float(endpoint.GetMasterVolumeLevelScalar()) * 100.0, 1)
        endpoint.SetMasterVolumeLevelScalar(target, None)
        new_pct = round(float(endpoint.GetMasterVolumeLevelScalar()) * 100.0, 1)
        return ActionResult(
            True,
            f"Microphone level set to {new_pct:.0f}% (was {previous_pct:.0f}%)",
            {"previous_pct": previous_pct, "level_pct": new_pct},
        )
    except Exception as err:
        logger.warning("Could not set microphone input level: %s", err)
        return ActionResult(False, f"Could not set microphone input level: {err}", {})


def get_microphone_permissions() -> Dict[str, Any]:
    """Read the Windows microphone privacy consent state (registry) for this app and machine.

    When microphone access is denied, Windows keeps handing out a perfectly healthy
    shared-mode capture stream that only ever contains *digital silence*. Reading the
    consent store therefore turns an otherwise invisible "the app can't hear me" cause
    into an explicit, actionable log line.

    Returns ``{"available", "global_access", "desktop_apps", "this_app", "app_key",
    "machine_policy", "error"}``; never raises.
    """
    info: Dict[str, Any] = {
        "available": False,
        "global_access": "unknown",
        "desktop_apps": "unknown",
        "this_app": "unknown",
        "app_key": None,
        "machine_policy": "unknown",
        "error": None,
    }

    try:
        import winreg
    except Exception as err:  # pragma: no cover - Windows only
        info["error"] = f"winreg unavailable: {err}"
        return info

    consent_root = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone"

    def _read_value(root, path: str) -> Optional[str]:
        """Return the 'Value' (Allow/Deny) of a consent-store key, or None when absent."""
        try:
            with winreg.OpenKey(root, path) as key:
                return str(winreg.QueryValueEx(key, "Value")[0])
        except Exception:
            return None

    try:
        info["global_access"] = _read_value(winreg.HKEY_CURRENT_USER, consent_root) or "unknown"
        info["desktop_apps"] = _read_value(winreg.HKEY_CURRENT_USER, consent_root + r"\NonPackaged") or "unknown"
        info["machine_policy"] = _read_value(winreg.HKEY_LOCAL_MACHINE, consent_root) or "unknown"

        # Per-app consent entry is keyed by the executable path with '\' replaced by '#'
        wanted_exe = os.path.basename(getattr(sys, "executable", "") or "").lower()
        if wanted_exe:
            non_packaged = consent_root + r"\NonPackaged"
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, non_packaged) as parent:
                    subkey_count = winreg.QueryInfoKey(parent)[0]
                    for i in range(subkey_count):
                        sub_name = winreg.EnumKey(parent, i)
                        if sub_name.lower().rsplit("#", 1)[-1] == wanted_exe:
                            info["app_key"] = sub_name
                            info["this_app"] = _read_value(
                                winreg.HKEY_CURRENT_USER, non_packaged + "\\" + sub_name
                            ) or "unknown"
                            break
            except Exception:
                pass

        info["available"] = True
    except Exception as err:
        info["error"] = str(err)

    return info


# Action mapping dictionary
ACTION_REGISTRY = {
    "volume_up": action_volume_up,
    "volume_down": action_volume_down,
    "volume_mute": action_volume_mute,
    "volume_unmute": action_volume_unmute,
    "volume_set": action_volume_set,
    "brightness_up": action_brightness_up,
    "brightness_down": action_brightness_down,
    "brightness_set": action_brightness_set,
    "open_browser": action_open_browser,
    "open_terminal": action_open_terminal,
    "open_editor": action_open_editor,
    "open_app": action_open_generic_app,
    "media_play_pause": action_media_play_pause,
    "media_next": action_media_next,
    "media_prev": action_media_prev,
    "lock_workstation": action_lock_workstation,
    "calibrate_mic": action_calibrate_mic,
}


# Action circuit breaker state: action_name -> list of execution timestamps
_action_history: Dict[str, List[float]] = {}
BREAKER_MAX_REPEATS: int = 3
BREAKER_WINDOW_S: float = 30.0


def check_circuit_breaker(action_name: str, now: Optional[float] = None) -> Tuple[bool, str]:
    """Check if action has exceeded rate limits (>3 times in 30s).
    
    Returns (tripped, message).
    """
    if now is None:
        now = time.time()

    # Prune timestamps older than window
    history = [t for t in _action_history.get(action_name, []) if (now - t) < BREAKER_WINDOW_S]
    _action_history[action_name] = history

    if len(history) >= BREAKER_MAX_REPEATS:
        friendly = action_name.replace("_", " ").capitalize()
        return True, f"{friendly} repeated too fast — pausing"

    return False, ""


def record_action_execution(action_name: str, now: Optional[float] = None):
    """Record an action execution timestamp."""
    if now is None:
        now = time.time()
    if action_name not in _action_history:
        _action_history[action_name] = []
    _action_history[action_name].append(now)


def reset_circuit_breaker():
    """Reset the circuit breaker history (useful for test isolation)."""
    _action_history.clear()


def execute(action_name: str, slot_value: Optional[int] = None, target_app: Optional[str] = None) -> ActionResult:
    """Execute a registered action safely with circuit-breaker protection."""
    handler = ACTION_REGISTRY.get(action_name)
    if not handler:
        return ActionResult(False, f"No action executor for '{action_name}'", {})

    # Rate-limiting circuit breaker check
    tripped, reason = check_circuit_breaker(action_name)
    if tripped:
        logger.warning("Circuit breaker tripped for '%s': %s", action_name, reason)
        return ActionResult(False, reason, {"breaker_tripped": True})

    try:
        if action_name in ("volume_set", "brightness_set") and slot_value is not None:
            res = handler(slot_value)
        elif action_name == "open_app" and target_app:
            res = handler(target_app)
        else:
            res = handler()

        if res.success:
            record_action_execution(action_name)
        return res
    except Exception as err:
        logger.error("Execution failed for '%s': %s", action_name, err, exc_info=True)
        return ActionResult(False, f"Action '{action_name}' failed: {err}", {})
