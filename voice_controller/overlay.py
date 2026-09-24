"""macOS-styled Desktop Notepad overlay for Voice Control with High-DPI, Traffic Lights, and Tray restore."""

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
    COLOR_BORDER,
    COLOR_TEXT_MAIN,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_NOTE,
    COLOR_ACCENT,
    COLOR_ACCENT_GOLD,
    COLOR_TL_RED,
    COLOR_TL_YELLOW,
    COLOR_TL_GREEN,
    COLOR_TL_BORDER_RED,
    COLOR_TL_BORDER_YELLOW,
    COLOR_TL_BORDER_GREEN,
    COLOR_STATUS_LISTENING,
    COLOR_STATUS_PROCESSING,
    COLOR_STATUS_EXECUTED,
    COLOR_STATUS_IGNORED,
    COLOR_STATUS_OFFLINE,
    HUD_WIDTH,
    HUD_HEIGHT,
    HUD_ALPHA,
    HUD_PADDING_X,
    HUD_PADDING_Y,
)

logger = logging.getLogger("MacOSNotepadOverlay")

# macOS Typography stack
FONT_TITLE = ("Segoe UI", 10, "bold")
FONT_SUBTITLE = ("Segoe UI", 8)
FONT_NOTE_DATE = ("Segoe UI", 8)
FONT_NOTE_TITLE = ("Segoe UI", 11, "bold")
FONT_NOTE_BODY = ("Segoe UI", 9)
FONT_NOTE_FEEDBACK = ("Segoe UI", 8, "italic")
FONT_STATUS_PILL = ("Segoe UI", 8, "bold")


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
    """Generate crisp 64x64 RGBA system tray icon."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Dark rounded background
    draw.rounded_rectangle((4, 4, 60, 60), radius=14, fill="#1C1C1E", outline="#FFD60A", width=3)
    # macOS note lines
    draw.line((18, 22, 46, 22), fill="#FFFFFF", width=4)
    draw.line((18, 32, 46, 32), fill="#FFFFFF", width=4)
    draw.line((18, 42, 34, 42), fill="#FFD60A", width=4)
    return img


class HUDOverlay:
    """macOS-styled minimal Notepad overlay pinned to the desktop with Traffic Lights and System Tray."""

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

        # Throttling trackers
        self._last_pulse_draw_time = 0.0
        self._last_pulse_level = 0.0

    def start(self):
        """Initialize Tkinter root with High-DPI awareness and enter event loop."""
        _enable_high_dpi()

        self.root = tk.Tk()
        self.root.title("Notes")

        # Window styling: frameless, transparent, always-on-top
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
        """Construct macOS dark notepad interface with traffic lights."""
        self.container = tk.Frame(self.root, bg=COLOR_BG, bd=0)
        self.container.pack(fill="both", expand=True, padx=1, pady=1)

        # 1. macOS Window Header (Only draggable surface)
        self.header = tk.Frame(self.container, bg=COLOR_PANEL, height=36)
        self.header.pack(fill="x", side="top")
        self.header.pack_propagate(False)

        # Drag bindings restricted strictly to header
        self.header.bind("<ButtonPress-1>", self._on_drag_start)
        self.header.bind("<B1-Motion>", self._on_drag_motion)

        # --- macOS Traffic Lights (🔴 🟡 🟢) on far left ---
        tl_frame = tk.Frame(self.header, bg=COLOR_PANEL)
        tl_frame.pack(side="left", padx=(10, 4), pady=6)
        tl_frame.bind("<ButtonPress-1>", self._on_drag_start)
        tl_frame.bind("<B1-Motion>", self._on_drag_motion)

        # 🔴 Red button: Quit
        self.btn_close = tk.Canvas(tl_frame, width=12, height=12, bg=COLOR_PANEL, bd=0, highlightthickness=0, cursor="hand2")
        self.btn_close.pack(side="left", padx=3)
        self.btn_close_circle = self.btn_close.create_oval(1, 1, 11, 11, fill=COLOR_TL_RED, outline=COLOR_TL_BORDER_RED, width=1)
        self.btn_close.bind("<Button-1>", lambda e: self.close())
        self.btn_close.bind("<Enter>", lambda e: self.btn_close.itemconfig(self.btn_close_circle, fill="#FF7B73"))
        self.btn_close.bind("<Leave>", lambda e: self.btn_close.itemconfig(self.btn_close_circle, fill=COLOR_TL_RED))

        # 🟡 Yellow button: Minimize to System Tray (with pystray restore)
        self.btn_min = tk.Canvas(tl_frame, width=12, height=12, bg=COLOR_PANEL, bd=0, highlightthickness=0, cursor="hand2")
        self.btn_min.pack(side="left", padx=3)
        self.btn_min_circle = self.btn_min.create_oval(1, 1, 11, 11, fill=COLOR_TL_YELLOW, outline=COLOR_TL_BORDER_YELLOW, width=1)
        self.btn_min.bind("<Button-1>", lambda e: self.minimize_to_tray())
        self.btn_min.bind("<Enter>", lambda e: self.btn_min.itemconfig(self.btn_min_circle, fill="#FFCE52"))
        self.btn_min.bind("<Leave>", lambda e: self.btn_min.itemconfig(self.btn_min_circle, fill=COLOR_TL_YELLOW))

        # 🟢 Green button: Reset size & position to default top-right
        self.btn_reset = tk.Canvas(tl_frame, width=12, height=12, bg=COLOR_PANEL, bd=0, highlightthickness=0, cursor="hand2")
        self.btn_reset.pack(side="left", padx=3)
        self.btn_reset_circle = self.btn_reset.create_oval(1, 1, 11, 11, fill=COLOR_TL_GREEN, outline=COLOR_TL_BORDER_GREEN, width=1)
        self.btn_reset.bind("<Button-1>", lambda e: self.reset_window())
        self.btn_reset.bind("<Enter>", lambda e: self.btn_reset.itemconfig(self.btn_reset_circle, fill="#4CD964"))
        self.btn_reset.bind("<Leave>", lambda e: self.btn_reset.itemconfig(self.btn_reset_circle, fill=COLOR_TL_GREEN))

        # App Icon & Title (Left aligned)
        title_frame = tk.Frame(self.header, bg=COLOR_PANEL)
        title_frame.pack(side="left", padx=(6, 8), pady=6)
        title_frame.bind("<ButtonPress-1>", self._on_drag_start)
        title_frame.bind("<B1-Motion>", self._on_drag_motion)

        icon_lbl = tk.Label(
            title_frame,
            text="📝",
            font=("Segoe UI Emoji", 10),
            bg=COLOR_PANEL,
            fg=COLOR_ACCENT_GOLD,
        )
        icon_lbl.pack(side="left", padx=(0, 6))
        icon_lbl.bind("<ButtonPress-1>", self._on_drag_start)
        icon_lbl.bind("<B1-Motion>", self._on_drag_motion)

        title_lbl = tk.Label(
            title_frame,
            text="Notes",
            font=FONT_TITLE,
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_PANEL,
        )
        title_lbl.pack(side="left")
        title_lbl.bind("<ButtonPress-1>", self._on_drag_start)
        title_lbl.bind("<B1-Motion>", self._on_drag_motion)

        # Discreet macOS Status Pill (Right aligned)
        self.status_pill = tk.Frame(self.header, bg="#202023", padx=8, pady=3)
        self.status_pill.pack(side="right", padx=12, pady=6)
        self.status_pill.bind("<ButtonPress-1>", self._on_drag_start)
        self.status_pill.bind("<B1-Motion>", self._on_drag_motion)

        self.status_dot = tk.Label(
            self.status_pill,
            text="●",
            font=FONT_STATUS_PILL,
            fg=COLOR_STATUS_LISTENING,
            bg="#202023",
        )
        self.status_dot.pack(side="left", padx=(0, 4))

        self.status_text = tk.Label(
            self.status_pill,
            text="Starting...",
            font=FONT_STATUS_PILL,
            fg=COLOR_STATUS_LISTENING,
            bg="#202023",
        )
        self.status_text.pack(side="left")

        # Subtle thin audio meter / accent bar below header
        self.canvas_pulse = tk.Canvas(
            self.container,
            bg=COLOR_BORDER,
            height=2,
            bd=0,
            highlightthickness=0,
        )
        self.canvas_pulse.pack(fill="x", side="top")
        self.pulse_bar = self.canvas_pulse.create_rectangle(
            0, 0, 0, 2, fill=COLOR_ACCENT, width=0
        )

        # 2. Body Views Container
        self.body_area = tk.Frame(self.container, bg=COLOR_BG)
        self.body_area.pack(fill="both", expand=True)

        # --- VIEW A: macOS Loading Screen ---
        self.loading_view = tk.Frame(self.body_area, bg=COLOR_BG)
        self.loading_view.pack(fill="both", expand=True)

        loader_inner = tk.Frame(self.loading_view, bg=COLOR_BG)
        loader_inner.place(relx=0.5, rely=0.5, anchor="center")

        lbl_load_icon = tk.Label(
            loader_inner,
            text="🎙️",
            font=("Segoe UI Emoji", 26),
            bg=COLOR_BG,
        )
        lbl_load_icon.pack(pady=(0, 6))

        self.lbl_load_title = tk.Label(
            loader_inner,
            text="Starting Voice Assistant...",
            font=FONT_NOTE_TITLE,
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_BG,
        )
        self.lbl_load_title.pack()

        self.lbl_load_sub = tk.Label(
            loader_inner,
            text="Preparing speech recognition...",
            font=FONT_SUBTITLE,
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_BG,
        )
        self.lbl_load_sub.pack(pady=(2, 8))

        # Minimalist progress track
        self.canvas_load_prog = tk.Canvas(
            loader_inner,
            bg="#2A2A2E",
            height=3,
            width=180,
            bd=0,
            highlightthickness=0,
        )
        self.canvas_load_prog.pack()
        self.load_bar = self.canvas_load_prog.create_rectangle(
            0, 0, 40, 3, fill=COLOR_ACCENT, width=0
        )
        self._animate_loading_step = 0
        self._animate_loader()

        # --- VIEW B: macOS Notepad Interface ---
        self.notepad_view = tk.Frame(self.body_area, bg=COLOR_BG)

        # Notepad header (Date / Subtitle)
        now_str = datetime.datetime.now().strftime("%B %d at %I:%M %p")
        self.lbl_note_date = tk.Label(
            self.notepad_view,
            text=now_str,
            font=FONT_NOTE_DATE,
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_BG,
        )
        self.lbl_note_date.pack(anchor="w", padx=14, pady=(8, 2))

        # Live Speech Feedback Bar (Continuous real-time speech transcription display)
        self.live_bar = tk.Frame(self.notepad_view, bg="#252528", padx=8, pady=4)
        self.live_bar.pack(fill="x", padx=14, pady=(2, 4))

        self.lbl_live_mic = tk.Label(
            self.live_bar,
            text="🎙️",
            font=("Segoe UI Emoji", 9),
            fg=COLOR_ACCENT_GOLD,
            bg="#252528",
        )
        self.lbl_live_mic.pack(side="left", padx=(0, 6))

        self.lbl_live_transcription = tk.Label(
            self.live_bar,
            text="Listening for speech...",
            font=("Segoe UI", 9, "italic"),
            fg=COLOR_TEXT_MUTED,
            bg="#252528",
            anchor="w",
            justify="left",
        )
        self.lbl_live_transcription.pack(side="left", fill="x", expand=True)

        # Scrolled text area styled like macOS Notes (Read-only, allows text selection)
        self.note_text = tk.Text(
            self.notepad_view,
            bg=COLOR_BG,
            fg=COLOR_TEXT_NOTE,
            insertbackground=COLOR_ACCENT,
            selectbackground="#3A3A3C",
            selectforeground=COLOR_TEXT_MAIN,
            font=FONT_NOTE_BODY,
            wrap="word",
            bd=0,
            padx=14,
            pady=4,
            highlightthickness=0,
            height=6,
        )
        self.note_text.pack(fill="both", expand=True)

        # Rich text tags
        self.note_text.tag_configure("timestamp", foreground=COLOR_TEXT_MUTED, font=FONT_NOTE_DATE)
        self.note_text.tag_configure("user_speech", foreground=COLOR_TEXT_MAIN, font=FONT_NOTE_BODY)
        self.note_text.tag_configure("action_ok", foreground=COLOR_STATUS_LISTENING, font=FONT_NOTE_FEEDBACK)
        self.note_text.tag_configure("action_warn", foreground=COLOR_STATUS_PROCESSING, font=FONT_NOTE_FEEDBACK)
        self.note_text.tag_configure("live_draft", foreground=COLOR_ACCENT, font=FONT_NOTE_FEEDBACK)

        # Initial placeholder note
        self.note_text.insert("end", "Speak a command (e.g. \"Turn up volume\", \"Open Chrome\", \"Dim screen\")...\n\n", "timestamp")
        self.note_text.config(state="disabled")

        # Bottom footer note
        self.footer = tk.Frame(self.notepad_view, bg=COLOR_BG, height=22)
        self.footer.pack(fill="x", side="bottom", padx=14, pady=(0, 6))

        self.lbl_footer = tk.Label(
            self.footer,
            text="🔴 Quit  🟡 Minimize to Tray  🟢 Reset",
            font=FONT_SUBTITLE,
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_BG,
        )
        self.lbl_footer.pack(side="left")

    def _animate_loader(self):
        """Smooth progress pulse for the loading bar."""
        if not self._is_running or not self.is_loading_mode:
            return

        w = 180
        bar_len = 50
        x1 = (self._animate_loading_step * 4) % (w + bar_len) - bar_len
        x2 = x1 + bar_len
        if hasattr(self, "canvas_load_prog") and hasattr(self, "load_bar"):
            self.canvas_load_prog.coords(self.load_bar, max(0, x1), 0, min(w, x2), 3)

        self._animate_loading_step += 1
        if self.root:
            self.root.after(30, self._animate_loader)

    def _on_drag_start(self, event):
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def _on_drag_motion(self, event):
        if self.root:
            deltax = event.x - self._drag_start_x
            deltay = event.y - self._drag_start_y
            new_x = self.root.winfo_x() + deltax
            new_y = self.root.winfo_y() + deltay
            self.root.geometry(f"+{new_x}+{new_y}")

    def minimize_to_tray(self):
        """Minimize overlay to system tray via pystray (Yellow traffic light)."""
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
                pystray.MenuItem("Open Voice Notes", self.restore_from_tray, default=True),
                pystray.MenuItem("Reset Size & Position", lambda icon, item: self.root.after(0, self.reset_window)),
                pystray.MenuItem("Clear Notes", lambda icon, item: self.root.after(0, self._clear_notes)),
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
        """Reset window geometry to default top-right position and size (Green traffic light)."""
        if self.root:
            screen_w = self.root.winfo_screenwidth()
            pos_x = screen_w - HUD_WIDTH - HUD_PADDING_X
            pos_y = HUD_PADDING_Y
            self.root.geometry(f"{HUD_WIDTH}x{HUD_HEIGHT}+{pos_x}+{pos_y}")
            self.root.deiconify()
            self.root.lift()
            self.root.attributes("-topmost", True)
            logger.info("HUD overlay geometry reset to %dx%d at (+%d,+%d).", HUD_WIDTH, HUD_HEIGHT, pos_x, pos_y)

    def _clear_notes(self):
        """Clear user notes in notepad."""
        if hasattr(self, "note_text"):
            self.note_text.config(state="normal")
            self.note_text.delete("1.0", "end")
            self.note_text.insert("end", "Notes reset.\n\n", "timestamp")
            self.note_text.config(state="disabled")

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
        # Switch between Loading View and Notepad View
        if msg.get("is_loading") is not None:
            self.is_loading_mode = bool(msg["is_loading"])
            if self.is_loading_mode:
                self.notepad_view.pack_forget()
                self.loading_view.pack(fill="both", expand=True)
                self._animate_loader()
            else:
                self.loading_view.pack_forget()
                self.notepad_view.pack(fill="both", expand=True)

        if msg.get("loading_message") and hasattr(self, "lbl_load_sub"):
            self.lbl_load_sub.config(text=msg["loading_message"])

        # Status Pill Update
        status = msg.get("status")
        if status:
            self.current_status = status.upper()

            if self.current_status == "LISTENING":
                self.status_dot.config(fg=COLOR_STATUS_LISTENING)
                self.status_text.config(text="Listening", fg=COLOR_STATUS_LISTENING)
            elif self.current_status == "PROCESSING":
                self.status_dot.config(fg=COLOR_STATUS_PROCESSING)
                self.status_text.config(text="Thinking...", fg=COLOR_STATUS_PROCESSING)
            elif self.current_status == "EXECUTED":
                self.status_dot.config(fg=COLOR_STATUS_EXECUTED)
                self.status_text.config(text="Done", fg=COLOR_STATUS_EXECUTED)
            elif self.current_status == "IGNORED":
                self.status_dot.config(fg=COLOR_STATUS_IGNORED)
                self.status_text.config(text="Unrecognized", fg=COLOR_STATUS_IGNORED)
            else:
                self.status_dot.config(fg=COLOR_STATUS_OFFLINE)
                self.status_text.config(text="Ready", fg=COLOR_STATUS_OFFLINE)

        # Continuous Live Transcription Update
        if "live_transcription" in msg and msg["live_transcription"] is not None:
            self._update_live_transcription(msg["live_transcription"])

        # Notepad Note Commit
        feedback = msg.get("feedback")
        transcription = msg.get("transcription")

        if feedback and not self.is_loading_mode:
            self._append_notepad_entry(transcription, feedback, status=self.current_status)

        # Throttled Audio Level Pulse (only redraw if delta > 0.02 or >=100ms elapsed)
        audio_level = msg.get("audio_level")
        if audio_level is not None and hasattr(self, "canvas_pulse"):
            now = time.perf_counter()
            if abs(audio_level - self._last_pulse_level) > 0.02 or (now - self._last_pulse_draw_time) >= 0.08:
                self._last_pulse_level = audio_level
                self._last_pulse_draw_time = now

                w = HUD_WIDTH
                bar_w = int(max(0.0, min(1.0, audio_level)) * w)
                bar_color = COLOR_STATUS_LISTENING
                if self.current_status == "PROCESSING":
                    bar_color = COLOR_STATUS_PROCESSING
                elif self.current_status == "EXECUTED":
                    bar_color = COLOR_STATUS_EXECUTED

                self.canvas_pulse.coords(self.pulse_bar, 0, 0, bar_w, 2)
                self.canvas_pulse.itemconfig(self.pulse_bar, fill=bar_color)

    def _update_live_transcription(self, text: Optional[str]):
        """Update live speech bubble and active draft line in notepad body."""
        cleaned = (text or "").strip()
        if hasattr(self, "lbl_live_transcription"):
            if cleaned:
                self.lbl_live_transcription.config(
                    text=f'"{cleaned}..."',
                    fg=COLOR_ACCENT_GOLD,
                )
                self.lbl_live_mic.config(fg=COLOR_STATUS_LISTENING)
            else:
                self.lbl_live_transcription.config(
                    text="Listening for speech...",
                    fg=COLOR_TEXT_MUTED,
                )
                self.lbl_live_mic.config(fg=COLOR_ACCENT_GOLD)

        if hasattr(self, "note_text"):
            self.note_text.config(state="normal")
            try:
                # Remove prior draft line if present
                if self.note_text.tag_ranges("live_draft"):
                    self.note_text.delete("live_draft.first", "live_draft.last")
                if cleaned:
                    self.note_text.insert("end", f"🎙️ \"{cleaned}...\"\n", "live_draft")
                    self.note_text.see("end")
            except Exception as err:
                logger.debug("Error updating live draft: %s", err)
            finally:
                self.note_text.config(state="disabled")

    def _append_notepad_entry(self, spoken_text: Optional[str], feedback_text: str, status: str = "EXECUTED"):
        """Append a clean, human-readable note entry into the read-only notepad, capped at 200 lines."""
        self.note_text.config(state="normal")

        # Remove prior draft line if present before committing permanent note
        try:
            if self.note_text.tag_ranges("live_draft"):
                self.note_text.delete("live_draft.first", "live_draft.last")
        except Exception:
            pass

        # Reset live speech bubble
        if hasattr(self, "lbl_live_transcription"):
            self.lbl_live_transcription.config(
                text="Listening for speech...",
                fg=COLOR_TEXT_MUTED,
            )
            self.lbl_live_mic.config(fg=COLOR_ACCENT_GOLD)

        # Capping history: if line count > 200, delete oldest lines
        try:
            line_count = int(self.note_text.index("end-1c").split(".")[0])
            if line_count > 200:
                self.note_text.delete("1.0", "50.0")
        except Exception:
            pass

        # Time tag
        time_str = datetime.datetime.now().strftime("%I:%M %p")
        self.note_text.insert("end", f"{time_str}  ", "timestamp")

        # Spoken text
        if spoken_text:
            self.note_text.insert("end", f'"{spoken_text}"\n', "user_speech")
        else:
            self.note_text.insert("end", "\n")

        # Result feedback
        tag = "action_ok" if status == "EXECUTED" else "action_warn"
        prefix = "  ✓ " if status == "EXECUTED" else "  — "
        self.note_text.insert("end", f"{prefix}{feedback_text}\n\n", tag)

        # Auto scroll to bottom & lock read-only state
        self.note_text.see("end")
        self.note_text.config(state="disabled")

    def close(self):
        """Safely destroy the overlay and terminate tray icon."""
        self._is_running = False
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
