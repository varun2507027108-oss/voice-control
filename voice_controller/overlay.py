"""Modern Non-AI Generic Floating Voice HUD Overlay.

Sleek, dark obsidian command bar inspired by Raycast, Linear, and precision audio
tooling. Features real-time live transcription streaming, multi-bar audio VU
visualizer, discrete execution badges, system tray integration, and High-DPI support.
"""

import ctypes
import datetime
import logging
import queue
import threading
import time
import tkinter as tk
from typing import Any, Dict, Optional
from PIL import Image, ImageDraw
import pystray

from voice_controller.config import (
    COLOR_BG,
    COLOR_PANEL,
    COLOR_CARD,
    COLOR_BORDER,
    COLOR_BORDER_SUBTLE,
    COLOR_TEXT_MAIN,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_SUBTLE,
    COLOR_TEXT_LIVE,
    COLOR_ACCENT,
    COLOR_ACCENT_AMBER,
    COLOR_STATUS_LISTENING,
    COLOR_STATUS_HEARING,
    COLOR_STATUS_PROCESSING,
    COLOR_STATUS_EXECUTED,
    COLOR_STATUS_IGNORED,
    COLOR_STATUS_OFFLINE,
    COLOR_BTN_HOVER,
    COLOR_BTN_CLOSE_HOVER,
    COLOR_TL_RED,
    COLOR_TL_YELLOW,
    COLOR_TL_GREEN,
    COLOR_TL_BORDER_RED,
    COLOR_TL_BORDER_YELLOW,
    COLOR_TL_BORDER_GREEN,
    HUD_WIDTH,
    HUD_HEIGHT,
    HUD_ALPHA,
    HUD_PADDING_X,
    HUD_PADDING_Y,
)

logger = logging.getLogger("VoiceOverlayHUD")

# Precision Modern Typography stack
FONT_BRAND = ("Segoe UI", 8, "bold")
FONT_STATUS = ("Segoe UI", 7, "bold")
FONT_TRANSCRIPTION = ("Segoe UI", 12, "bold")
FONT_FEEDBACK = ("Segoe UI", 8, "bold")
FONT_TELEMETRY = ("Segoe UI", 7, "bold")
FONT_LOAD_TITLE = ("Segoe UI", 11, "bold")
FONT_LOAD_SUB = ("Segoe UI", 8)


def _enable_high_dpi():
    """Enable Per-Monitor High-DPI awareness on Windows."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _generate_tray_icon() -> Image.Image:
    """Generate crisp 64x64 RGBA system tray icon for dark / light taskbars."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Dark obsidian capsule background
    draw.rounded_rectangle((4, 4, 60, 60), radius=16, fill="#0B0D11", outline="#38BDF8", width=3)
    # Microphone glyph
    draw.rounded_rectangle((26, 16, 38, 36), radius=6, fill="#38BDF8")
    draw.arc((20, 24, 44, 44), start=0, end=180, fill="#F8FAFC", width=3)
    draw.line((32, 44, 32, 50), fill="#F8FAFC", width=3)
    draw.line((24, 50, 40, 50), fill="#F8FAFC", width=3)
    return img


class HUDOverlay:
    """Precision floating voice command HUD with live transcription and audio level VU meter."""

    def __init__(self, on_close_callback=None):
        self.on_close_callback = on_close_callback
        self.msg_queue: queue.Queue = queue.Queue()
        self.root: Optional[tk.Tk] = None
        self._is_running = False
        self._tray_icon: Optional[pystray.Icon] = None

        # State tracking
        self.current_status = "INITIALIZING"
        self.is_loading_mode = True
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._last_committed_text = ""
        self._reset_timer = None

        # Audio VU Visualizer smoothing state
        self._vu_bars = [2.0] * 7
        self._vu_multipliers = [0.45, 0.70, 1.00, 0.95, 0.75, 0.55, 0.35]
        self._last_vu_time = 0.0
        self._current_audio_level = 0.0

    def start(self):
        """Initialize Tkinter root with High-DPI awareness and enter event loop."""
        _enable_high_dpi()

        self.root = tk.Tk()
        self.root.title("Voice Control")

        # Window styling: frameless, acrylic alpha, always-on-top
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-alpha", HUD_ALPHA)
        except Exception:
            pass

        self.root.config(bg=COLOR_BORDER)

        # Position window in top-right corner
        screen_w = self.root.winfo_screenwidth()
        pos_x = screen_w - HUD_WIDTH - HUD_PADDING_X
        pos_y = HUD_PADDING_Y
        self.root.geometry(f"{HUD_WIDTH}x{HUD_HEIGHT}+{pos_x}+{pos_y}")

        # Global hotkey to cleanly exit
        self.root.bind("<Escape>", lambda e: self.close())

        self._build_ui()
        self._is_running = True

        # Process queue at 30Hz cap (33ms)
        self.root.after(33, self._process_queue)
        self.root.mainloop()

    def _build_ui(self):
        """Construct precision obsidian voice command HUD."""
        # 1. Outer 1px precision hairline border
        self.outer_frame = tk.Frame(self.root, bg=COLOR_BORDER, bd=0)
        self.outer_frame.pack(fill="both", expand=True)

        # 2. Main obsidian container
        self.container = tk.Frame(self.outer_frame, bg=COLOR_BG, bd=0)
        self.container.pack(fill="both", expand=True, padx=1, pady=1)

        # Make entire HUD surface draggable (convenience for floating bar)
        self.container.bind("<ButtonPress-1>", self._on_drag_start)
        self.container.bind("<B1-Motion>", self._on_drag_motion)

        # =========================================================================
        # SECTION 1: HEADER STRIP (Height: 32px)
        # =========================================================================
        self.header = tk.Frame(self.container, bg=COLOR_PANEL, height=32)
        self.header.pack(fill="x", side="top")
        self.header.pack_propagate(False)
        self.header.bind("<ButtonPress-1>", self._on_drag_start)
        self.header.bind("<B1-Motion>", self._on_drag_motion)

        # Left: Drag grip & Brand title
        brand_frame = tk.Frame(self.header, bg=COLOR_PANEL)
        brand_frame.pack(side="left", padx=(10, 6), pady=4)
        brand_frame.bind("<ButtonPress-1>", self._on_drag_start)
        brand_frame.bind("<B1-Motion>", self._on_drag_motion)

        lbl_grip = tk.Label(
            brand_frame,
            text="⠿",
            font=("Segoe UI", 9),
            fg=COLOR_TEXT_SUBTLE,
            bg=COLOR_PANEL,
        )
        lbl_grip.pack(side="left", padx=(0, 6))
        lbl_grip.bind("<ButtonPress-1>", self._on_drag_start)
        lbl_grip.bind("<B1-Motion>", self._on_drag_motion)

        lbl_brand = tk.Label(
            brand_frame,
            text="VOICE CONTROL",
            font=FONT_BRAND,
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_PANEL,
        )
        lbl_brand.pack(side="left")
        lbl_brand.bind("<ButtonPress-1>", self._on_drag_start)
        lbl_brand.bind("<B1-Motion>", self._on_drag_motion)

        # Tactile Status Pill Badge
        self.status_pill = tk.Frame(self.header, bg="#181B24", padx=7, pady=2)
        self.status_pill.pack(side="left", padx=8, pady=4)
        self.status_pill.bind("<ButtonPress-1>", self._on_drag_start)
        self.status_pill.bind("<B1-Motion>", self._on_drag_motion)

        self.status_dot = tk.Label(
            self.status_pill,
            text="●",
            font=("Segoe UI", 7, "bold"),
            fg=COLOR_STATUS_LISTENING,
            bg="#181B24",
        )
        self.status_dot.pack(side="left", padx=(0, 4))

        self.status_text = tk.Label(
            self.status_pill,
            text="STARTING",
            font=FONT_STATUS,
            fg=COLOR_STATUS_LISTENING,
            bg="#181B24",
        )
        self.status_text.pack(side="left")

        # Center / Right: Live Multi-Bar Audio VU Visualizer
        vu_frame = tk.Frame(self.header, bg=COLOR_PANEL)
        vu_frame.pack(side="left", padx=(10, 4), pady=4)
        vu_frame.bind("<ButtonPress-1>", self._on_drag_start)
        vu_frame.bind("<B1-Motion>", self._on_drag_motion)

        self.canvas_vu = tk.Canvas(
            vu_frame,
            width=50,
            height=14,
            bg=COLOR_PANEL,
            bd=0,
            highlightthickness=0,
        )
        self.canvas_vu.pack(side="left")
        self.canvas_vu.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas_vu.bind("<B1-Motion>", self._on_drag_motion)

        # Pre-create 7 equalizer bars
        self.vu_bar_ids = []
        bar_w = 4
        bar_gap = 3
        for i in range(7):
            bx1 = i * (bar_w + bar_gap) + 2
            bx2 = bx1 + bar_w
            bid = self.canvas_vu.create_rectangle(
                bx1, 12, bx2, 14, fill="#1E2330", width=0
            )
            self.vu_bar_ids.append(bid)

        # Far Right: Precision Window Controls (Minimize, Reset, Close)
        ctrl_frame = tk.Frame(self.header, bg=COLOR_PANEL)
        ctrl_frame.pack(side="right", padx=(4, 8), pady=4)

        # Reset button (Green circle compatibility / Reset geometry)
        self.btn_reset = tk.Canvas(ctrl_frame, width=12, height=12, bg=COLOR_PANEL, bd=0, highlightthickness=0, cursor="hand2")
        self.btn_reset.pack(side="left", padx=3)
        self.btn_reset_circle = self.btn_reset.create_oval(1, 1, 11, 11, fill="#10B981", outline=COLOR_TL_BORDER_GREEN, width=1)
        self.btn_reset.bind("<Button-1>", lambda e: self.reset_window())
        self.btn_reset.bind("<Enter>", lambda e: self.btn_reset.itemconfig(self.btn_reset_circle, fill="#34D399"))
        self.btn_reset.bind("<Leave>", lambda e: self.btn_reset.itemconfig(self.btn_reset_circle, fill="#10B981"))

        # Minimize to Tray button (Yellow circle compatibility / Minimize)
        self.btn_min = tk.Canvas(ctrl_frame, width=12, height=12, bg=COLOR_PANEL, bd=0, highlightthickness=0, cursor="hand2")
        self.btn_min.pack(side="left", padx=3)
        self.btn_min_circle = self.btn_min.create_oval(1, 1, 11, 11, fill="#F59E0B", outline=COLOR_TL_BORDER_YELLOW, width=1)
        self.btn_min.bind("<Button-1>", lambda e: self.minimize_to_tray())
        self.btn_min.bind("<Enter>", lambda e: self.btn_min.itemconfig(self.btn_min_circle, fill="#FBBF24"))
        self.btn_min.bind("<Leave>", lambda e: self.btn_min.itemconfig(self.btn_min_circle, fill="#F59E0B"))

        # Close / Quit button (Red circle compatibility / Close)
        self.btn_close = tk.Canvas(ctrl_frame, width=12, height=12, bg=COLOR_PANEL, bd=0, highlightthickness=0, cursor="hand2")
        self.btn_close.pack(side="left", padx=3)
        self.btn_close_circle = self.btn_close.create_oval(1, 1, 11, 11, fill="#EF4444", outline=COLOR_TL_BORDER_RED, width=1)
        self.btn_close.bind("<Button-1>", lambda e: self.close())
        self.btn_close.bind("<Enter>", lambda e: self.btn_close.itemconfig(self.btn_close_circle, fill="#F87171"))
        self.btn_close.bind("<Leave>", lambda e: self.btn_close.itemconfig(self.btn_close_circle, fill="#EF4444"))

        # =========================================================================
        # SECTION 2: BODY AREA (Swappable Loading vs Live Command View)
        # =========================================================================
        self.body_area = tk.Frame(self.container, bg=COLOR_BG)
        self.body_area.pack(fill="both", expand=True)
        self.body_area.bind("<ButtonPress-1>", self._on_drag_start)
        self.body_area.bind("<B1-Motion>", self._on_drag_motion)

        # --- VIEW A: Sleek Obsidian Loader ---
        self.loading_view = tk.Frame(self.body_area, bg=COLOR_BG)
        self.loading_view.pack(fill="both", expand=True)

        load_inner = tk.Frame(self.loading_view, bg=COLOR_BG)
        load_inner.place(relx=0.5, rely=0.5, anchor="center")

        self.lbl_load_title = tk.Label(
            load_inner,
            text="Initializing Voice Control...",
            font=FONT_LOAD_TITLE,
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_BG,
        )
        self.lbl_load_title.pack(pady=(0, 3))

        self.lbl_load_sub = tk.Label(
            load_inner,
            text="Warming up local STT & Intent models...",
            font=FONT_LOAD_SUB,
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_BG,
        )
        self.lbl_load_sub.pack(pady=(0, 8))

        # Precision Progress Track
        self.canvas_load_prog = tk.Canvas(
            load_inner,
            bg="#1A1D27",
            height=3,
            width=220,
            bd=0,
            highlightthickness=0,
        )
        self.canvas_load_prog.pack()
        self.load_bar = self.canvas_load_prog.create_rectangle(
            0, 0, 40, 3, fill=COLOR_ACCENT, width=0
        )
        self._animate_loading_step = 0
        self._animate_loader()

        # --- VIEW B: Live Voice Command Bar ---
        self.hud_view = tk.Frame(self.body_area, bg=COLOR_BG)

        # 1. Main Live Transcription Capsule (The Centerpiece of the HUD)
        self.card_transcription = tk.Frame(
            self.hud_view,
            bg=COLOR_CARD,
            bd=0,
            highlightthickness=1,
            highlightbackground=COLOR_BORDER_SUBTLE,
        )
        self.card_transcription.pack(fill="x", padx=10, pady=(6, 4))
        self.card_transcription.bind("<ButtonPress-1>", self._on_drag_start)
        self.card_transcription.bind("<B1-Motion>", self._on_drag_motion)

        inner_card = tk.Frame(self.card_transcription, bg=COLOR_CARD, padx=10, pady=7)
        inner_card.pack(fill="both", expand=True)
        inner_card.bind("<ButtonPress-1>", self._on_drag_start)
        inner_card.bind("<B1-Motion>", self._on_drag_motion)

        # Left mic glyph inside capsule
        self.lbl_mic_glyph = tk.Label(
            inner_card,
            text="🎙",
            font=("Segoe UI Emoji", 11),
            fg=COLOR_ACCENT,
            bg=COLOR_CARD,
        )
        self.lbl_mic_glyph.pack(side="left", padx=(0, 8))
        self.lbl_mic_glyph.bind("<ButtonPress-1>", self._on_drag_start)
        self.lbl_mic_glyph.bind("<B1-Motion>", self._on_drag_motion)

        # High-visibility Live Transcription Text Label
        self.lbl_transcription = tk.Label(
            inner_card,
            text="Listening for voice commands...",
            font=FONT_TRANSCRIPTION,
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_CARD,
            anchor="w",
            justify="left",
            wraplength=360,
        )
        self.lbl_transcription.pack(side="left", fill="both", expand=True)
        self.lbl_transcription.bind("<ButtonPress-1>", self._on_drag_start)
        self.lbl_transcription.bind("<B1-Motion>", self._on_drag_motion)

        # 2. Bottom Action Feedback & Telemetry Strip (Height ~26px)
        self.footer = tk.Frame(self.hud_view, bg=COLOR_BG)
        self.footer.pack(fill="x", side="bottom", padx=10, pady=(2, 6))
        self.footer.bind("<ButtonPress-1>", self._on_drag_start)
        self.footer.bind("<B1-Motion>", self._on_drag_motion)

        # Left: Sleek Action Feedback Badge Pill
        self.action_pill = tk.Frame(self.footer, bg="#161922", padx=8, pady=3)
        self.action_pill.pack(side="left")
        self.action_pill.bind("<ButtonPress-1>", self._on_drag_start)
        self.action_pill.bind("<B1-Motion>", self._on_drag_motion)

        self.lbl_action_badge = tk.Label(
            self.action_pill,
            text='e.g. "volume up", "open chrome", "dim screen"',
            font=FONT_FEEDBACK,
            fg=COLOR_TEXT_SUBTLE,
            bg="#161922",
            anchor="w",
        )
        self.lbl_action_badge.pack(side="left")
        self.lbl_action_badge.bind("<ButtonPress-1>", self._on_drag_start)
        self.lbl_action_badge.bind("<B1-Motion>", self._on_drag_motion)

        # Right: Local Model Telemetry Badge
        self.lbl_telemetry = tk.Label(
            self.footer,
            text="100% LOCAL",
            font=FONT_TELEMETRY,
            fg=COLOR_TEXT_SUBTLE,
            bg=COLOR_BG,
        )
        self.lbl_telemetry.pack(side="right", padx=(4, 0))
        self.lbl_telemetry.bind("<ButtonPress-1>", self._on_drag_start)
        self.lbl_telemetry.bind("<B1-Motion>", self._on_drag_motion)

    def _animate_loader(self):
        """Smooth progress pulse for the loading bar."""
        if not self._is_running or not self.is_loading_mode:
            return

        w = 220
        bar_len = 50
        x1 = (self._animate_loading_step * 5) % (w + bar_len) - bar_len
        x2 = x1 + bar_len
        if hasattr(self, "canvas_load_prog") and hasattr(self, "load_bar"):
            self.canvas_load_prog.coords(self.load_bar, max(0, x1), 0, min(w, x2), 3)

        self._animate_loading_step += 1
        if self.root:
            self.root.after(30, self._animate_loader)

    def _on_drag_start(self, event):
        self._drag_start_x = event.x_root - self.root.winfo_x()
        self._drag_start_y = event.y_root - self.root.winfo_y()

    def _on_drag_motion(self, event):
        if self.root:
            new_x = event.x_root - self._drag_start_x
            new_y = event.y_root - self._drag_start_y
            self.root.geometry(f"+{new_x}+{new_y}")

    def minimize_to_tray(self):
        """Minimize overlay to system tray via pystray."""
        logger.info("Minimizing HUD overlay to system tray...")
        if self.root:
            self.root.withdraw()
        self._ensure_tray_icon()

    def _ensure_tray_icon(self):
        """Create and run pystray icon in background thread if not already running."""
        if self._tray_icon is not None:
            return

        try:
            icon_img = _generate_tray_icon()
            menu = pystray.Menu(
                pystray.MenuItem("Open Voice Control", self.restore_from_tray, default=True),
                pystray.MenuItem("Reset Size & Position", lambda icon, item: self.root.after(0, self.reset_window)),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Quit", lambda icon, item: self.root.after(0, self.close)),
            )
            self._tray_icon = pystray.Icon("VoiceControl", icon_img, "Voice Controller", menu)
            tray_thread = threading.Thread(target=self._tray_icon.run, daemon=True)
            tray_thread.start()
            logger.info("System tray icon active.")
        except Exception as err:
            logger.error("Failed to initialize system tray icon: %s", err)

    def restore_from_tray(self, icon=None, item=None):
        """Restore window from system tray (thread-safe, callable from pystray thread)."""
        if self.root:
            self.root.after(0, self._do_restore)

    def _do_restore(self):
        """Actual Tkinter restoration on main GUI thread."""
        if self.root:
            self.root.deiconify()
            self.root.lift()
            self.root.attributes("-topmost", True)
        if self._tray_icon:
            try:
                self._tray_icon.stop()
            except Exception:
                pass
            self._tray_icon = None
        logger.info("HUD overlay restored from system tray.")

    def reset_window(self):
        """Reset window geometry to default top-right position and size."""
        if self.root:
            screen_w = self.root.winfo_screenwidth()
            pos_x = screen_w - HUD_WIDTH - HUD_PADDING_X
            pos_y = HUD_PADDING_Y
            self.root.geometry(f"{HUD_WIDTH}x{HUD_HEIGHT}+{pos_x}+{pos_y}")
            self.root.deiconify()
            self.root.lift()
            self.root.attributes("-topmost", True)
            logger.info("HUD overlay geometry reset to %dx%d at (+%d,+%d).", HUD_WIDTH, HUD_HEIGHT, pos_x, pos_y)

    def update_state(
        self,
        status: Optional[str] = None,
        transcription: Optional[str] = None,
        action: Optional[str] = None,
        confidence: Optional[float] = None,
        feedback: Optional[str] = None,
        audio_level: Optional[float] = None,
        is_loading: Optional[bool] = None,
        loading_message: Optional[str] = None,
        live_transcription: Optional[str] = None,
    ):
        """Thread-safe state update method to be called from background threads."""
        self.msg_queue.put({
            "status": status,
            "transcription": transcription,
            "action": action,
            "confidence": confidence,
            "feedback": feedback,
            "audio_level": audio_level,
            "is_loading": is_loading,
            "loading_message": loading_message,
            "live_transcription": live_transcription,
        })

    def _process_queue(self):
        """Main-thread queue drainer applying updates at 30Hz."""
        try:
            while not self.msg_queue.empty():
                msg = self.msg_queue.get_nowait()
                self._apply_update(msg)
        except Exception as err:
            logger.error("Error processing HUD queue: %s", err)
        finally:
            if self._is_running and self.root:
                self.root.after(33, self._process_queue)

    def _apply_update(self, msg: Dict[str, Any]):
        # Switch between Loading View and HUD View
        if msg.get("is_loading") is not None:
            self.is_loading_mode = bool(msg["is_loading"])
            if self.is_loading_mode:
                self.hud_view.pack_forget()
                self.loading_view.pack(fill="both", expand=True)
                self._animate_loader()
            else:
                self.loading_view.pack_forget()
                self.hud_view.pack(fill="both", expand=True)

        if msg.get("loading_message") and hasattr(self, "lbl_load_sub"):
            self.lbl_load_sub.config(text=msg["loading_message"])

        # Status Pill Update
        status = msg.get("status")
        if status:
            self.current_status = status.upper()
            if self.current_status == "LISTENING":
                self._set_status_pill("LISTENING", COLOR_STATUS_LISTENING)
            elif self.current_status == "HEARING":
                self._set_status_pill("HEARING", COLOR_STATUS_HEARING)
            elif self.current_status == "PROCESSING":
                self._set_status_pill("THINKING", COLOR_STATUS_PROCESSING)
            elif self.current_status == "EXECUTED":
                self._set_status_pill("DONE", COLOR_STATUS_EXECUTED)
            elif self.current_status == "IGNORED":
                self._set_status_pill("UNRECOGNIZED", COLOR_STATUS_IGNORED)
            else:
                self._set_status_pill(self.current_status[:10], COLOR_STATUS_OFFLINE)

        # Continuous Live Streaming Transcription Update
        if "live_transcription" in msg and msg["live_transcription"] is not None:
            self._update_live_transcription(msg["live_transcription"])

        # Committed Action Feedback
        feedback = msg.get("feedback")
        transcription = msg.get("transcription")
        if feedback and not self.is_loading_mode:
            self._commit_command_result(transcription, feedback, status=self.current_status)

        # Real-time Multi-bar Audio VU Visualizer Update
        audio_level = msg.get("audio_level")
        if audio_level is not None:
            self._current_audio_level = max(0.0, min(1.0, float(audio_level)))
            self._update_vu_meter(self._current_audio_level)

    def _set_status_pill(self, label: str, color: str):
        """Update top status pill text and glowing dot."""
        if hasattr(self, "status_dot") and hasattr(self, "status_text"):
            self.status_dot.config(fg=color)
            self.status_text.config(text=label, fg=color)

    def _update_vu_meter(self, level: float):
        """Animate 7-bar audio equalizer based on live mic input level with smooth falloff."""
        if not hasattr(self, "canvas_vu") or not hasattr(self, "vu_bar_ids"):
            return

        bar_h_max = 12
        bar_w = 4
        bar_gap = 3

        for i, bid in enumerate(self.vu_bar_ids):
            # Target height with natural bell-curve spectrum shaping
            mult = self._vu_multipliers[i]
            target_h = max(2.0, level * bar_h_max * mult * 1.5)

            # Smooth decay / falloff
            if target_h > self._vu_bars[i]:
                self._vu_bars[i] = target_h
            else:
                self._vu_bars[i] = max(2.0, self._vu_bars[i] * 0.75 + target_h * 0.25)

            bh = int(min(bar_h_max, max(2, self._vu_bars[i])))
            bx1 = i * (bar_w + bar_gap) + 2
            bx2 = bx1 + bar_w
            by2 = 13
            by1 = by2 - bh

            fill_color = "#1E2330" if level < 0.04 else (COLOR_STATUS_LISTENING if level < 0.60 else COLOR_ACCENT_AMBER)
            self.canvas_vu.coords(bid, bx1, by1, bx2, by2)
            self.canvas_vu.itemconfig(bid, fill=fill_color)

    def _update_live_transcription(self, text: Optional[str]):
        """Render streaming live speech transcription directly in the primary display box."""
        cleaned = (text or "").strip()
        if hasattr(self, "lbl_transcription"):
            if cleaned:
                # Active speech in progress: render live stream in bright high-contrast white
                # with streaming cursor indicator
                self.lbl_transcription.config(
                    text=f"{cleaned} ▍",
                    fg=COLOR_TEXT_MAIN,
                )
                self.lbl_mic_glyph.config(fg=COLOR_STATUS_HEARING)
                self._set_status_pill("HEARING", COLOR_STATUS_HEARING)
                self.card_transcription.config(highlightbackground=COLOR_ACCENT)
            else:
                if self.current_status == "LISTENING" and not self._last_committed_text:
                    self.lbl_transcription.config(
                        text="Listening for voice commands...",
                        fg=COLOR_TEXT_MUTED,
                    )
                    self.lbl_mic_glyph.config(fg=COLOR_ACCENT)
                    self.card_transcription.config(highlightbackground=COLOR_BORDER_SUBTLE)

    def _commit_command_result(self, spoken_text: Optional[str], feedback_text: str, status: str = "EXECUTED"):
        """Display finalized speech text and action result badge."""
        self._last_committed_text = spoken_text or ""

        # 1. Update main transcription box with finalized spoken words
        if hasattr(self, "lbl_transcription"):
            if spoken_text:
                self.lbl_transcription.config(
                    text=spoken_text,
                    fg=COLOR_TEXT_MAIN,
                )
            else:
                self.lbl_transcription.config(
                    text="Listening for voice commands...",
                    fg=COLOR_TEXT_MUTED,
                )
            self.card_transcription.config(highlightbackground=COLOR_BORDER_SUBTLE)
            self.lbl_mic_glyph.config(fg=COLOR_ACCENT)

        # 2. Update action feedback badge pill
        if hasattr(self, "lbl_action_badge") and hasattr(self, "action_pill"):
            if status == "EXECUTED":
                self.lbl_action_badge.config(
                    text=f"✓  {feedback_text}",
                    fg=COLOR_STATUS_LISTENING,
                    bg="#0D281E",
                )
                self.action_pill.config(bg="#0D281E")
            elif status == "IGNORED":
                self.lbl_action_badge.config(
                    text=f"—  {feedback_text}",
                    fg=COLOR_STATUS_IGNORED,
                    bg="#2B1115",
                )
                self.action_pill.config(bg="#2B1115")
            else:
                self.lbl_action_badge.config(
                    text=f"•  {feedback_text}",
                    fg=COLOR_TEXT_MAIN,
                    bg="#181B24",
                )
                self.action_pill.config(bg="#181B24")

        # 3. Schedule auto-fade back to idle ready prompt after 3.0 seconds
        if self._reset_timer is not None and self.root:
            try:
                self.root.after_cancel(self._reset_timer)
            except Exception:
                pass
        if self.root:
            self._reset_timer = self.root.after(3000, self._fade_to_idle)

    def _fade_to_idle(self):
        """Smoothly reset prompt text to idle state while keeping last feedback visible."""
        if self.current_status in ("LISTENING", "READY"):
            if hasattr(self, "lbl_transcription"):
                self.lbl_transcription.config(
                    text="Listening for voice commands...",
                    fg=COLOR_TEXT_MUTED,
                )
                self.lbl_mic_glyph.config(fg=COLOR_ACCENT)
                self.card_transcription.config(highlightbackground=COLOR_BORDER_SUBTLE)
            if hasattr(self, "lbl_action_badge") and hasattr(self, "action_pill"):
                self.lbl_action_badge.config(
                    text='e.g. "volume up", "open chrome", "dim screen"',
                    fg=COLOR_TEXT_SUBTLE,
                    bg="#161922",
                )
                self.action_pill.config(bg="#161922")
            self._last_committed_text = ""

    def close(self):
        """Safely destroy the overlay and terminate tray icon."""
        self._is_running = False
        if self._reset_timer is not None and self.root:
            try:
                self.root.after_cancel(self._reset_timer)
            except Exception:
                pass
            self._reset_timer = None

        if self._tray_icon:
            try:
                self._tray_icon.stop()
            except Exception:
                pass
            self._tray_icon = None

        if self.on_close_callback:
            try:
                self.on_close_callback()
            except Exception:
                pass
        if self.root:
            try:
                self.root.destroy()
            except Exception:
                pass
