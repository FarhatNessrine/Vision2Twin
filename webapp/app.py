"""
Four-workstation YOLO web application — smooth-browser-playback edition
=====================================================================

Key design change versus the previous server-preview version:
- The browser plays the four videos natively at 1× speed.
- The browser captures a frame from each active video at a target of 10 Hz.
- One multipart request sends the active station frames together.
- A single shared YOLO model performs one batch inference.
- Only JSON detections/counts are returned for the live overlay.
- The browser playback never waits for YOLO. If inference is slower than 10 Hz,
  frames are skipped rather than slowing the videos.
- Aggregation is still organized in 2-second source-video windows, targeting
  20 predictions/window. The workbook records the actual sample count and marks
  incomplete windows explicitly.

Outputs:
- Hub_raw / Body_raw / RearCap_raw / Final_raw
- Latest_State
- State_History
- Cycles
- OPC_Snapshots
- OPC_Mapping
- Configuration
- One clickable representative annotated JPEG per station per 2-second window
- Fast ROI-based cycle-duration detection from raw prediction samples (2 confirmations)
- Simplified Excel workbook: Latest_State, State_History, Cycles, OPC_Snapshots, OPC_Mapping, Configuration
- Smart FFmpeg browser preparation: fast remux when possible, re-encode only when required
- On-demand OPC UA snapshot publishing from the latest CLEANED 2-second ROI aggregates
- Embedded asyncua OPC UA server started from the webapp only when the user clicks the OPC button
- Stable mapping: YOLO/ROI values -> HubStockCurrent, HubAssembledCurrent, ...

OPC UA is NOT streamed continuously. The webapp publishes one snapshot only when requested.
Cycle events are detected independently from the 2-second state aggregation.
"""

from __future__ import annotations

import asyncio
import io
import json
import math
import mimetypes
import os
import shutil
import subprocess
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import StatisticsError, mode
from typing import Any

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_file, send_from_directory
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from ultralytics import YOLO

try:
    from asyncua import Server, ua
    ASYNCUA_AVAILABLE = True
    ASYNCUA_IMPORT_ERROR = ""
except Exception as _asyncua_exc:
    Server = None  # type: ignore[assignment]
    ua = None      # type: ignore[assignment]
    ASYNCUA_AVAILABLE = False
    ASYNCUA_IMPORT_ERROR = str(_asyncua_exc)


# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_videos")
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
EXCEL_PATH = os.path.join(BASE_DIR, "assembly_4stations.xlsx")
ROI_CONFIG_PATH = os.path.join(BASE_DIR, "roi_config.json")

MODEL_PATH = os.environ.get(
    "YOLO_MODEL",
    r"C:\Users\farhatne\Desktop\webApp\best_spont.pt",
)

TARGET_SAMPLE_FPS = float(os.environ.get("TARGET_SAMPLE_FPS", "10"))
AGGREGATION_WINDOW_S = float(os.environ.get("AGGREGATION_WINDOW_S", "2"))
SAMPLES_PER_WINDOW = int(round(TARGET_SAMPLE_FPS * AGGREGATION_WINDOW_S))

# Browser capture remains 640 px wide, while YOLO inference is reduced to 512
# by default for a performance test. Set YOLO_IMGSZ=640 to restore the standard
# inference resolution if the accuracy drop is unacceptable.
CAPTURE_WIDTH = int(os.environ.get("CAPTURE_WIDTH", "640"))
INFERENCE_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "512"))
CAPTURE_JPEG_QUALITY = float(os.environ.get("CAPTURE_JPEG_QUALITY", "0.72"))
EVIDENCE_JPEG_QUALITY = int(os.environ.get("EVIDENCE_JPEG_QUALITY", "88"))

# Save the complete workbook less frequently than the previous version.
# With four stations and 2-second windows, 120 aggregate rows ~= 60 source sec.
EXCEL_SAVE_EVERY_AGGREGATED_ROWS = int(
    os.environ.get("EXCEL_SAVE_EVERY_AGGREGATED_ROWS", "120")
)

# Browser preparation. "smart" is recommended:
# - every upload is normalized to a browser-friendly MP4;
# - H.264/yuv420p sources (including MKV) are REMUXED with stream copy, which is fast;
# - incompatible codecs/profiles are re-encoded with libx264 ultrafast;
# - trimmed MP4 files are also remuxed so timestamps/moov metadata are rebuilt.
# Set VIDEO_PREP_MODE=never only if every source already plays correctly in the browser.
VIDEO_PREP_MODE = os.environ.get("VIDEO_PREP_MODE", "smart").strip().lower()
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")

# On-demand OPC UA snapshot server. It is intentionally NOT started at app launch.
# Clicking the OPC button starts it (if needed), writes the latest cleaned values,
# then toggles StartControl False -> True so FlexSim can use the new snapshot.
OPC_ENDPOINT_BIND = os.environ.get("OPC_ENDPOINT_BIND", "opc.tcp://0.0.0.0:4840/flexsim/server/")
OPC_ENDPOINT_PUBLIC = os.environ.get("OPC_ENDPOINT_PUBLIC", "opc.tcp://localhost:4840/flexsim/server/")
OPC_NAMESPACE_URI = os.environ.get("OPC_NAMESPACE_URI", "http://flexsim.project")
OPC_SERVER_NAME = os.environ.get("OPC_SERVER_NAME", "FlexSim Custom OPC UA Server")
OPC_START_TIMEOUT_S = float(os.environ.get("OPC_START_TIMEOUT_S", "10"))
OPC_WRITE_TIMEOUT_S = float(os.environ.get("OPC_WRITE_TIMEOUT_S", "10"))

# Confirmed working mapping structure. The values come directly from the latest
# 2-second cleaned ROI aggregates kept in memory by this web application.
OPC_MAPPING = [
    {"station_id": "hub", "zone": "stock", "yolo_class": "Hub-not-assembled", "opc_variable": "HubStockCurrent"},
    {"station_id": "hub", "zone": "output", "yolo_class": "Hub-assembled", "opc_variable": "HubAssembledCurrent"},
    {"station_id": "body", "zone": "stock", "yolo_class": "Body-not-assembled", "opc_variable": "BodyStockCurrent"},
    {"station_id": "body", "zone": "output", "yolo_class": "Body-assembled", "opc_variable": "BodyAssembledCurrent"},
    {"station_id": "rear_cap", "zone": "stock", "yolo_class": "Rear-cap-not-assembled", "opc_variable": "RearCapStockCurrent"},
    {"station_id": "rear_cap", "zone": "output", "yolo_class": "Rear-cap-assembled", "opc_variable": "RearCapAssembledCurrent"},
    {"station_id": "final", "zone": "stock", "yolo_class": "Hub-assembled", "opc_variable": "FinalHubCurrent"},
    {"station_id": "final", "zone": "stock", "yolo_class": "Body-assembled", "opc_variable": "FinalBodyCurrent"},
    {"station_id": "final", "zone": "stock", "yolo_class": "Rear-cap-assembled", "opc_variable": "FinalRearCapCurrent"},
    {"station_id": "final", "zone": "output", "yolo_class": "Tidal-turbine", "opc_variable": "TidalTurbineCurrent"},
]

# The 2-second cleaned state is still used for Latest_State / OPC snapshots.
# These thresholds only describe snapshot quality; they do NOT gate cycle timing.
STATE_MIN_SUPPORT = float(os.environ.get("STATE_MIN_SUPPORT", "0.60"))
STATE_MIN_SAMPLE_RATIO = float(os.environ.get("STATE_MIN_SAMPLE_RATIO", "0.50"))
STATE_MIN_SAMPLES = max(2, int(math.ceil(SAMPLES_PER_WINDOW * STATE_MIN_SAMPLE_RATIO)))

# Cycle timing is intentionally faster than state aggregation. START/END events are
# confirmed from consecutive raw ROI-count predictions. With a 10 Hz target, two
# confirmations usually add only ~0.1-0.3 s, while still rejecting one-frame misses.
CYCLE_CONFIRMATIONS = max(2, int(os.environ.get("CYCLE_CONFIRMATIONS", "2")))
CYCLE_MIN_DURATION_S = float(os.environ.get("CYCLE_MIN_DURATION_S", "1.0"))

# Device can be overridden, e.g. YOLO_DEVICE=cpu or YOLO_DEVICE=0.
_env_device = os.environ.get("YOLO_DEVICE")
if _env_device is None or _env_device == "":
    YOLO_DEVICE: int | str = 0 if torch.cuda.is_available() else "cpu"
else:
    YOLO_DEVICE = int(_env_device) if _env_device.isdigit() else _env_device

STATIONS: dict[str, dict[str, Any]] = {
    "hub": {
        "name": "Hub Assembly",
        "sheet": "Hub_raw",
        "folder": "hub",
        "relevant_classes": ["Hub-not-assembled", "Hub-assembled"],
        "cycle": {
            "input_zone": "stock",
            "input_classes": ["Hub-not-assembled"],
            "output_zone": "output",
            "output_class": "Hub-assembled",
        },
    },
    "body": {
        "name": "Body Assembly",
        "sheet": "Body_raw",
        "folder": "body",
        "relevant_classes": ["Body-not-assembled", "Body-assembled"],
        "cycle": {
            "input_zone": "stock",
            "input_classes": ["Body-not-assembled"],
            "output_zone": "output",
            "output_class": "Body-assembled",
        },
    },
    "rear_cap": {
        "name": "Rear-cap Assembly",
        "sheet": "RearCap_raw",
        "folder": "rear_cap",
        "relevant_classes": ["Rear-cap-not-assembled", "Rear-cap-assembled"],
        "cycle": {
            "input_zone": "stock",
            "input_classes": ["Rear-cap-not-assembled"],
            "output_zone": "output",
            "output_class": "Rear-cap-assembled",
        },
    },
    "final": {
        "name": "Final Tidal-Turbine Assembly",
        "sheet": "Final_raw",
        "folder": "final",
        "relevant_classes": [
            "Hub-assembled",
            "Body-assembled",
            "Rear-cap-assembled",
            "Tidal-turbine",
        ],
        "cycle": {
            "input_zone": "stock",
            "input_classes": ["Hub-assembled", "Body-assembled", "Rear-cap-assembled"],
            "output_zone": "output",
            "output_class": "Tidal-turbine",
        },
    },
}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
for cfg in STATIONS.values():
    os.makedirs(os.path.join(FRAMES_DIR, cfg["folder"]), exist_ok=True)

app = Flask(__name__, static_folder=BASE_DIR)


# =============================================================================
# Embedded on-demand OPC UA server
# =============================================================================
class EmbeddedOpcUaServer:
    """Run asyncua in its own asyncio loop/thread and publish snapshots on demand."""

    def __init__(self) -> None:
        self.thread: threading.Thread | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.server: Any = None
        self.nodes: dict[str, Any] = {}
        self.start_control_node: Any = None
        self.namespace_index: int | None = None
        self.ready_event = threading.Event()
        self.stop_event = threading.Event()
        self.guard = threading.Lock()
        self.error: str | None = None
        self.last_snapshot_id: str | None = None
        self.last_publish_at: str | None = None
        self.last_values: dict[str, int] = {}

    def is_running(self) -> bool:
        return bool(self.thread and self.thread.is_alive() and self.ready_event.is_set() and not self.error)

    def public_status(self) -> dict[str, Any]:
        with self.guard:
            return {
                "available": ASYNCUA_AVAILABLE,
                "running": self.is_running(),
                "endpoint": OPC_ENDPOINT_PUBLIC,
                "namespace_uri": OPC_NAMESPACE_URI,
                "namespace_index": self.namespace_index,
                "start_control": bool(self.is_running()),
                "last_snapshot_id": self.last_snapshot_id,
                "last_publish_at": self.last_publish_at,
                "last_values": dict(self.last_values),
                "error": self.error or (None if ASYNCUA_AVAILABLE else ASYNCUA_IMPORT_ERROR),
            }

    def start(self, timeout: float = OPC_START_TIMEOUT_S) -> None:
        if not ASYNCUA_AVAILABLE:
            raise RuntimeError(
                "asyncua is not available. Install it in the same Python environment with: pip install asyncua"
            )
        with self.guard:
            if self.is_running():
                return
            if self.thread and self.thread.is_alive():
                # A previous start is still in progress.
                pass
            else:
                self.error = None
                self.ready_event.clear()
                self.stop_event.clear()
                self.thread = threading.Thread(
                    target=self._thread_main,
                    daemon=True,
                    name="embedded-opcua-server",
                )
                self.thread.start()
        if not self.ready_event.wait(timeout):
            raise RuntimeError(f"OPC UA server did not become ready within {timeout:.1f}s")
        if self.error:
            raise RuntimeError(self.error)

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self.loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._server_main())
        except Exception as exc:
            with self.guard:
                self.error = f"OPC UA server error: {exc}"
            self.ready_event.set()
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            loop.close()
            self.loop = None

    async def _server_main(self) -> None:
        assert Server is not None and ua is not None
        server = Server()
        await server.init()
        server.set_endpoint(OPC_ENDPOINT_BIND)
        server.set_server_name(OPC_SERVER_NAME)
        idx = await server.register_namespace(OPC_NAMESPACE_URI)
        objects = server.nodes.objects

        vision = await objects.add_folder(ua.NodeId("vision_variables", idx), "vision_variables")
        control = await objects.add_folder(ua.NodeId("control_variables", idx), "control_variables")

        start_control = await control.add_variable(
            ua.NodeId("control.StartControl", idx),
            "StartControl",
            ua.Variant(False, ua.VariantType.Boolean),
        )
        await start_control.set_writable()

        nodes: dict[str, Any] = {}
        for item in OPC_MAPPING:
            name = item["opc_variable"]
            node = await vision.add_variable(
                ua.NodeId(f"vision.{name}", idx),
                name,
                ua.Variant(0, ua.VariantType.Int64),
            )
            await node.set_writable()
            nodes[name] = node

        self.server = server
        self.nodes = nodes
        self.start_control_node = start_control
        self.namespace_index = idx

        await server.start()
        self.ready_event.set()
        print(f"[OPC] Server running: {OPC_ENDPOINT_PUBLIC}  namespace={idx}")
        try:
            while not self.stop_event.is_set():
                await asyncio.sleep(0.25)
        finally:
            await server.stop()
            print("[OPC] Server stopped")

    async def _publish_async(self, values: dict[str, int], snapshot_id: str) -> None:
        assert ua is not None
        if not self.start_control_node:
            raise RuntimeError("StartControl node is not available")

        # Force a new trigger edge each time the user publishes a snapshot.
        await self.start_control_node.write_value(ua.Variant(False, ua.VariantType.Boolean))
        for name, value in values.items():
            node = self.nodes.get(name)
            if node is None:
                raise RuntimeError(f"OPC node is missing: {name}")
            await node.write_value(ua.Variant(int(value), ua.VariantType.Int64))
        await self.start_control_node.write_value(ua.Variant(True, ua.VariantType.Boolean))

        with self.guard:
            self.last_snapshot_id = snapshot_id
            self.last_publish_at = datetime.now().isoformat(timespec="milliseconds")
            self.last_values = {k: int(v) for k, v in values.items()}

    def publish(self, values: dict[str, int], snapshot_id: str) -> None:
        self.start()
        if self.loop is None:
            raise RuntimeError("OPC UA event loop is not available")
        future = asyncio.run_coroutine_threadsafe(
            self._publish_async(values, snapshot_id), self.loop
        )
        future.result(timeout=OPC_WRITE_TIMEOUT_S)

    def stop(self, timeout: float = 5.0) -> None:
        with self.guard:
            thread = self.thread
            if not thread or not thread.is_alive():
                self.ready_event.clear()
                return
            self.stop_event.set()
        thread.join(timeout=timeout)
        with self.guard:
            if thread.is_alive():
                self.error = "OPC UA server did not stop cleanly within timeout"
            else:
                self.ready_event.clear()
                self.thread = None
                self.server = None
                self.nodes = {}
                self.start_control_node = None
                self.namespace_index = None


opc_server = EmbeddedOpcUaServer()


def _opc_mapping_public() -> list[dict[str, Any]]:
    rows = []
    ns = opc_server.namespace_index
    for item in OPC_MAPPING:
        sid = item["station_id"]
        row = dict(item)
        row["workstation"] = STATIONS[sid]["name"]
        row["node_id"] = (
            f"ns={ns};s=vision.{item['opc_variable']}" if ns is not None
            else f"ns=<dynamic>;s=vision.{item['opc_variable']}"
        )
        rows.append(row)
    return rows


def _build_opc_snapshot() -> dict[str, Any]:
    """Map latest CLEANED in-memory aggregates directly to OPC variables."""
    rows: list[dict[str, Any]] = []
    values: dict[str, int] = {}
    missing_stations: set[str] = set()

    for mapping in OPC_MAPPING:
        sid = mapping["station_id"]
        agg = stations[sid].latest_aggregate
        if not agg or agg.get("window_index") is None:
            missing_stations.add(sid)
            continue
        zone = mapping["zone"]
        cls = mapping["yolo_class"]
        value = int(agg.get("roi_mode_counts", {}).get(zone, {}).get(cls, 0))
        support = float(agg.get("roi_support", {}).get(zone, {}).get(cls, 0.0))
        values[mapping["opc_variable"]] = value
        rows.append({
            "station_id": sid,
            "workstation": STATIONS[sid]["name"],
            "zone": zone,
            "yolo_class": cls,
            "opc_variable": mapping["opc_variable"],
            "value": value,
            "support": round(support, 4),
            "window_index": int(agg["window_index"]),
            "window_start_s": float(agg.get("timestamp_start_s", 0.0)),
            "window_end_s": float(agg.get("timestamp_end_s", 0.0)),
            "window_complete": bool(agg.get("window_complete", False)),
            "sample_count": int(agg.get("sample_count", 0)),
            "expected_samples": int(agg.get("expected_samples", SAMPLES_PER_WINDOW)),
            "quality_ok": bool(
                int(agg.get("sample_count", 0)) >= STATE_MIN_SAMPLES
                and support >= STATE_MIN_SUPPORT
            ),
        })

    if missing_stations:
        names = ", ".join(STATIONS[s]["name"] for s in sorted(missing_stations))
        raise RuntimeError(
            f"No cleaned aggregate is available yet for: {names}. "
            "Let the analysis run until each station has completed at least one 2-second window."
        )

    snapshot_id = datetime.now().strftime("snapshot_%Y%m%d_%H%M%S")
    return {
        "snapshot_id": snapshot_id,
        "created_at": datetime.now().isoformat(timespec="milliseconds"),
        "session_id": session.get("session_id"),
        "values": values,
        "rows": rows,
        "warnings": [
            f"{r['workstation']} / {r['opc_variable']}: latest window is partial or support is low"
            for r in rows if not r["quality_ok"]
        ],
    }


# =============================================================================
# Excel style helpers
# =============================================================================
_HDR_FILL = PatternFill("solid", fgColor="1F3864")
_HDR_FONT = Font(bold=True, color="FFFFFF", name="Calibri", size=10)
_ALT_FILL = PatternFill("solid", fgColor="DCE6F1")
_GREEN_FILL = PatternFill("solid", fgColor="E2F0D9")
_BORDER = Border(
    left=Side(style="thin", color="B4C6E7"),
    right=Side(style="thin", color="B4C6E7"),
    top=Side(style="thin", color="B4C6E7"),
    bottom=Side(style="thin", color="B4C6E7"),
)

BOX_COLORS_BGR = [
    (255, 229, 0),
    (113, 61, 255),
    (0, 230, 118),
    (102, 209, 255),
    (255, 61, 113),
    (60, 154, 255),
    (178, 86, 255),
    (255, 130, 0),
]
_color_map: dict[str, tuple[int, int, int]] = {}
_color_index = 0


def class_color_bgr(label: str) -> tuple[int, int, int]:
    global _color_index
    if label not in _color_map:
        _color_map[label] = BOX_COLORS_BGR[_color_index % len(BOX_COLORS_BGR)]
        _color_index += 1
    return _color_map[label]


# =============================================================================
# Runtime state
# =============================================================================
@dataclass
class CycleRuntime:
    status: str = "IDLE"
    active: bool = False
    completed_cycles: int = 0
    current_cycle_id: int | None = None
    start_window: int | None = None
    start_time_s: float | None = None
    start_frame: str | None = None
    input_before: int | None = None
    input_after_start: int | None = None
    output_min_since_start: int | None = None
    prev_input_count: int | None = None
    prev_output_count: int | None = None
    last_cycle_id: int | None = None
    last_cycle_duration_s: float | None = None
    last_event: str = "Waiting for first detection baseline"

    # Two-sample confirmation state. Candidate timestamps/frames keep the FIRST
    # observation, so the reported event time is not delayed by confirmation.
    start_candidate_count: int = 0
    start_candidate_time_s: float | None = None
    start_candidate_input: int | None = None
    start_candidate_jpeg: bytes | None = None
    start_candidate_detections: list[dict[str, Any]] = field(default_factory=list)

    end_candidate_count: int = 0
    end_candidate_time_s: float | None = None
    end_candidate_output: int | None = None
    end_candidate_jpeg: bytes | None = None
    end_candidate_detections: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StationRuntime:
    station_id: str
    name: str
    video_path: str | None = None
    video_url: str | None = None
    original_name: str | None = None
    stored_paths: list[str] = field(default_factory=list)
    source_fps: float = 0.0
    frame_count: int = 0
    duration_s: float = 0.0
    width: int = 0
    height: int = 0
    loaded: bool = False
    status: str = "empty"
    warning: str | None = None
    transcoded: bool = False
    prepare_method: str = ""
    prepare_seconds: float = 0.0
    rois: dict[str, dict[str, float]] = field(default_factory=dict)
    cycle: CycleRuntime = field(default_factory=CycleRuntime)

    current_video_time_s: float = 0.0
    current_source_frame_index: int = 0
    samples_processed: int = 0
    aggregated_windows: int = 0
    progress_pct: float = 0.0
    latest_counts: dict[str, int] = field(default_factory=dict)
    latest_roi_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    latest_total: int = 0
    latest_average_confidence: float = 0.0
    latest_aggregate: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


state_lock = threading.RLock()
analysis_lock = threading.Lock()
model_load_lock = threading.Lock()
model: YOLO | None = None
model_classes: list[str] = []

stations: dict[str, StationRuntime] = {
    sid: StationRuntime(sid, cfg["name"]) for sid, cfg in STATIONS.items()
}

session: dict[str, Any] = {
    "session_id": None,
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "min_confidence": 0.50,
    "source_time_s": 0.0,
    "wall_elapsed_s": 0.0,
    "predict_batches": 0,
    "images_processed": 0,
    "last_batch_inference_ms": None,
    "last_batch_size": 0,
    "excel_ready": False,
    "error": None,
}

excel_writer: "ExcelSessionWriter | None" = None
station_buffers: dict[str, list[dict[str, Any]]] = {sid: [] for sid in STATIONS}
previous_window: dict[str, int | None] = {sid: None for sid in STATIONS}


# =============================================================================
# Model / utility helpers
# =============================================================================
def ensure_model_loaded() -> None:
    global model, model_classes
    if model is not None:
        return
    with model_load_lock:
        if model is not None:
            return
        print(f"[INFO] Loading YOLO model: {MODEL_PATH}")
        print(f"[INFO] YOLO device: {YOLO_DEVICE} | imgsz={INFERENCE_IMGSZ}")
        model = YOLO(MODEL_PATH)
        names = model.names
        if isinstance(names, dict):
            model_classes = [str(names[i]) for i in sorted(names)]
        else:
            model_classes = [str(x) for x in names]
        print(f"[INFO] Model ready with {len(model_classes)} classes: {model_classes}")


def _safe_mode(values: list[int]) -> int:
    if not values:
        return 0
    try:
        return int(mode(values))
    except StatisticsError:
        return int(Counter(values).most_common(1)[0][0])


def _probe_video(path: str) -> dict[str, Any]:
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not open this video")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or frame_count <= 0:
            raise RuntimeError("Could not read video FPS/frame count")
        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration_s": frame_count / fps,
            "width": width,
            "height": height,
        }
    finally:
        cap.release()


def ffmpeg_available() -> bool:
    return shutil.which(FFMPEG_BIN) is not None


def ffprobe_available() -> bool:
    return shutil.which(FFPROBE_BIN) is not None


def _run_process(cmd: list[str], timeout: int) -> tuple[bool, str]:
    print(f"[INFO] media command: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout or "command failed")[-1500:]
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"command timed out after {timeout}s"
    except FileNotFoundError:
        return False, f"executable not found: {cmd[0]}"
    except Exception as exc:
        return False, str(exc)


def probe_video_info(src_path: str) -> dict[str, Any]:
    """Probe codec details used to choose fast remux vs re-encode."""
    info: dict[str, Any] = {
        "codec": "", "profile": "", "pix_fmt": "", "height": 0, "width": 0,
        "has_audio": False, "audio_codec": "",
    }
    if not ffprobe_available():
        return info

    v_cmd = [
        FFPROBE_BIN, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,profile,pix_fmt,height,width",
        "-of", "json", src_path,
    ]
    try:
        v = subprocess.run(v_cmd, capture_output=True, text=True, timeout=30)
        streams = json.loads(v.stdout or "{}").get("streams", [])
        if streams:
            st = streams[0]
            info["codec"] = str(st.get("codec_name", "")).lower()
            info["profile"] = str(st.get("profile", "")).lower()
            info["pix_fmt"] = str(st.get("pix_fmt", "")).lower()
            info["height"] = int(st.get("height", 0) or 0)
            info["width"] = int(st.get("width", 0) or 0)
    except Exception as exc:
        print(f"[WARN] ffprobe video probe failed: {exc}")

    a_cmd = [
        FFPROBE_BIN, "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_name", "-of", "json", src_path,
    ]
    try:
        a = subprocess.run(a_cmd, capture_output=True, text=True, timeout=30)
        streams = json.loads(a.stdout or "{}").get("streams", [])
        if streams:
            info["has_audio"] = True
            info["audio_codec"] = str(streams[0].get("codec_name", "")).lower()
    except Exception as exc:
        print(f"[WARN] ffprobe audio probe failed: {exc}")
    return info


def is_browser_copy_compatible(info: dict[str, Any]) -> bool:
    """True when the video stream can normally be copied into MP4 without re-encoding."""
    if info.get("codec") not in ("h264", "avc"):
        return False
    bad_profiles = (
        "high 4:4:4", "high 10", "high 4:2:2", "cavlc 4:4:4",
        "multiview high", "stereo high",
    )
    profile = str(info.get("profile", ""))
    if any(x in profile for x in bad_profiles):
        return False
    pix_fmt = str(info.get("pix_fmt", ""))
    if pix_fmt and pix_fmt != "yuv420p":
        return False
    return True


def _make_tmp_mp4(dst_path: str) -> str:
    base = dst_path[:-4] if dst_path.lower().endswith(".mp4") else dst_path
    return f"{base}.{uuid.uuid4().hex[:8]}.tmp.mp4"


def _finalize_media(tmp_path: str, dst_path: str) -> bool:
    if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
        os.replace(tmp_path, dst_path)
        return True
    try:
        os.remove(tmp_path)
    except OSError:
        pass
    return False


def prepare_video_for_browser(src_path: str, dst_path: str) -> tuple[bool, str, str]:
    """
    Normalize any MKV/MP4/etc. into a browser-friendly MP4.

    Returns: (ok, error, method)
      method = remux-copy | reencode-ultrafast

    The fast path copies an H.264/yuv420p video stream instead of re-encoding it.
    This is why compatible MKV files become almost as fast to prepare as MP4 files.
    """
    if not ffmpeg_available():
        return False, "ffmpeg not found in PATH (or FFMPEG_BIN)", ""

    info = probe_video_info(src_path)
    print(
        f"[INFO] Probe {os.path.basename(src_path)} -> codec={info['codec']!r}, "
        f"profile={info['profile']!r}, pix_fmt={info['pix_fmt']!r}, "
        f"size={info['width']}x{info['height']}, audio={info['audio_codec']!r}"
    )

    base_flags = ["-err_detect", "ignore_err", "-fflags", "+genpts+igndts"]
    audio_args = (
        ["-c:a", "copy"] if info.get("has_audio") and info.get("audio_codec") == "aac"
        else (["-c:a", "aac", "-b:a", "128k"] if info.get("has_audio") else ["-an"])
    )

    # Fast path: remux compatible H.264 video into a fresh MP4 container. This also
    # fixes many trimmed-MP4 timestamp/moov problems without re-encoding the video.
    if is_browser_copy_compatible(info):
        tmp = _make_tmp_mp4(dst_path)
        cmd = [
            FFMPEG_BIN, "-y", *base_flags, "-i", src_path,
            "-map", "0:v:0", "-map", "0:a?",
            "-c:v", "copy", *audio_args,
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart", "-f", "mp4", tmp,
        ]
        ok, err = _run_process(cmd, timeout=300)
        if ok and _finalize_media(tmp, dst_path):
            return True, "", "remux-copy"
        print(f"[WARN] Fast remux failed; falling back to re-encode: {err[-300:]}")
        try:
            os.remove(tmp)
        except OSError:
            pass

    # Compatibility path. If ffprobe is unavailable, or the stream is HEVC/VP9/
    # unusual H.264, re-encode once using the old application's fast strategy.
    tmp = _make_tmp_mp4(dst_path)
    height = int(info.get("height") or 0)
    vf_args = ["-vf", "scale=-2:720"] if height > 720 else []
    cmd = [
        FFMPEG_BIN, "-y", *base_flags, "-i", src_path,
        "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        *vf_args, "-pix_fmt", "yuv420p", "-avoid_negative_ts", "make_zero",
        *audio_args, "-movflags", "+faststart", "-f", "mp4", tmp,
    ]
    ok, err = _run_process(cmd, timeout=1800)
    if ok and _finalize_media(tmp, dst_path):
        return True, "", "reencode-ultrafast"
    try:
        os.remove(tmp)
    except OSError:
        pass
    return False, err or "FFmpeg output missing/empty", ""


def _validate_normalized_rect(rect: dict[str, Any]) -> dict[str, float]:
    try:
        x1 = float(rect["x1"]); y1 = float(rect["y1"])
        x2 = float(rect["x2"]); y2 = float(rect["y2"])
    except Exception as exc:
        raise ValueError("ROI must contain x1, y1, x2, y2") from exc
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    if x2 - x1 < 0.01 or y2 - y1 < 0.01:
        raise ValueError("ROI is too small")
    return {
        "x1": round(x1, 6), "y1": round(y1, 6),
        "x2": round(x2, 6), "y2": round(y2, 6),
    }


def _load_roi_config() -> dict[str, dict[str, dict[str, float]]]:
    data = {sid: {} for sid in STATIONS}
    if not os.path.exists(ROI_CONFIG_PATH):
        return data
    try:
        with open(ROI_CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for sid in STATIONS:
            if not isinstance(raw.get(sid), dict):
                continue
            for zone in ("stock", "output"):
                rect = raw[sid].get(zone)
                if isinstance(rect, dict):
                    data[sid][zone] = _validate_normalized_rect(rect)
    except Exception as exc:
        print(f"[WARN] Could not load ROI config: {exc}")
    return data


def _save_roi_config() -> None:
    payload = {sid: dict(stations[sid].rois) for sid in STATIONS}
    tmp = ROI_CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, ROI_CONFIG_PATH)


def _cleanup_previous_upload(st: StationRuntime) -> None:
    for path in list(st.stored_paths):
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    st.stored_paths = []


def _reset_station_processing(st: StationRuntime) -> None:
    st.current_video_time_s = 0.0
    st.current_source_frame_index = 0
    st.samples_processed = 0
    st.aggregated_windows = 0
    st.progress_pct = 0.0
    st.latest_counts = {}
    st.latest_roi_counts = {}
    st.latest_total = 0
    st.latest_average_confidence = 0.0
    st.latest_aggregate = {}
    st.cycle = CycleRuntime()
    st.error = None
    st.status = "ready" if st.loaded else "empty"


def _cycle_public_dict(st: StationRuntime) -> dict[str, Any]:
    c = st.cycle
    current_duration = None
    if c.active and c.start_time_s is not None:
        current_duration = max(0.0, st.current_video_time_s - c.start_time_s)
    return {
        "status": c.status,
        "active": c.active,
        "completed_cycles": c.completed_cycles,
        "current_cycle_id": c.current_cycle_id,
        "start_window": c.start_window,
        "start_time_s": None if c.start_time_s is None else round(c.start_time_s, 3),
        "current_duration_s": None if current_duration is None else round(current_duration, 3),
        "last_cycle_id": c.last_cycle_id,
        "last_cycle_duration_s": None if c.last_cycle_duration_s is None else round(c.last_cycle_duration_s, 3),
        "prev_input_count": c.prev_input_count,
        "prev_output_count": c.prev_output_count,
        "last_event": c.last_event,
    }


def _station_public_dict(st: StationRuntime) -> dict[str, Any]:
    return {
        "station_id": st.station_id,
        "name": st.name,
        "loaded": st.loaded,
        "status": st.status,
        "original_name": st.original_name,
        "video_url": st.video_url,
        "source_fps": round(st.source_fps, 3),
        "frame_count": st.frame_count,
        "duration_s": round(st.duration_s, 3),
        "width": st.width,
        "height": st.height,
        "warning": st.warning,
        "transcoded": st.transcoded,
        "prepare_method": st.prepare_method,
        "prepare_seconds": round(st.prepare_seconds, 3),
        "rois": dict(st.rois),
        "roi_ready": all(zone in st.rois for zone in ("stock", "output")),
        "current_video_time_s": round(st.current_video_time_s, 3),
        "current_source_frame_index": st.current_source_frame_index,
        "samples_processed": st.samples_processed,
        "aggregated_windows": st.aggregated_windows,
        "progress_pct": round(st.progress_pct, 2),
        "latest_counts": dict(st.latest_counts),
        "latest_roi_counts": {z: dict(v) for z, v in st.latest_roi_counts.items()},
        "latest_total": st.latest_total,
        "latest_average_confidence": round(st.latest_average_confidence, 4),
        "latest_aggregate": dict(st.latest_aggregate),
        "cycle_state": _cycle_public_dict(st),
        "error": st.error,
    }


def _parse_result(result, min_confidence: float) -> tuple[list[dict[str, Any]], dict[str, int], float]:
    detections: list[dict[str, Any]] = []
    counts = {cls: 0 for cls in model_classes}
    confidences: list[float] = []
    if result.boxes is not None:
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < min_confidence:
                continue
            cls_id = int(box.cls[0])
            label = str(model.names[cls_id])  # type: ignore[union-attr]
            x1, y1, x2, y2 = [round(v) for v in box.xyxy[0].tolist()]
            detections.append({
                "label": label,
                "confidence": round(confidence, 4),
                "bbox": [x1, y1, x2, y2],
            })
            counts[label] = counts.get(label, 0) + 1
            confidences.append(confidence)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return detections, counts, avg_conf


def _roi_counts(
    detections: list[dict[str, Any]],
    frame_shape,
    rois: dict[str, dict[str, float]],
) -> dict[str, dict[str, int]]:
    h, w = frame_shape[:2]
    out = {
        "stock": {cls: 0 for cls in model_classes},
        "output": {cls: 0 for cls in model_classes},
    }
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx = ((x1 + x2) * 0.5) / max(w, 1)
        cy = ((y1 + y2) * 0.5) / max(h, 1)
        for zone in ("stock", "output"):
            rect = rois.get(zone)
            if rect and rect["x1"] <= cx <= rect["x2"] and rect["y1"] <= cy <= rect["y2"]:
                out[zone][det["label"]] = out[zone].get(det["label"], 0) + 1
    return out


def _draw_boxes(frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        label = det["label"]
        confidence = det["confidence"]
        x1, y1, x2, y2 = det["bbox"]
        color = class_color_bgr(label)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {confidence:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        ty = y1 - 5 if y1 > th + 8 else y2 + th + 8
        top = max(0, ty - th - baseline - 3)
        bottom = min(out.shape[0] - 1, ty + baseline + 2)
        right = min(out.shape[1] - 1, x1 + tw + 6)
        cv2.rectangle(out, (x1, top), (right, bottom), color, -1)
        cv2.putText(out, text, (x1 + 3, max(th + 1, ty)), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return out


def _draw_rois(frame: np.ndarray, rois: dict[str, dict[str, float]]) -> np.ndarray:
    out = frame
    h, w = out.shape[:2]
    styles = {
        "stock": ((0, 230, 118), "STOCK / INPUT ROI"),
        "output": ((60, 154, 255), "OUTPUT ROI"),
    }
    for zone, (color, label) in styles.items():
        r = rois.get(zone)
        if not r:
            continue
        x1 = int(round(r["x1"] * w)); y1 = int(round(r["y1"] * h))
        x2 = int(round(r["x2"] * w)); y2 = int(round(r["y2"] * h))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        cv2.putText(out, label, (x1 + 4, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
    return out


# =============================================================================
# Excel writer — compact experiment workbook
# =============================================================================
def _style_header(ws) -> None:
    for cell in ws[1]:
        cell.fill = _HDR_FILL
        cell.font = _HDR_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = _BORDER
    ws.freeze_panes = "A2"


def _style_row(ws, row_idx: int, green: bool = False) -> None:
    fill = _GREEN_FILL if green else (_ALT_FILL if row_idx % 2 == 0 else None)
    for cell in ws[row_idx]:
        if fill is not None:
            cell.fill = fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = _BORDER


def _state_fields(station_id: str, agg: dict[str, Any]) -> list[Any]:
    """Return a compact, station-specific state row (max 3 inputs + 1 output)."""
    cfg = STATIONS[station_id]["cycle"]
    in_zone = cfg["input_zone"]
    out_zone = cfg["output_zone"]
    inputs = list(cfg["input_classes"])
    values: list[Any] = []
    for idx in range(3):
        if idx < len(inputs):
            cls = inputs[idx]
            values += [
                cls,
                int(agg["roi_mode_counts"][in_zone].get(cls, 0)),
                round(float(agg["roi_support"][in_zone].get(cls, 0.0)), 4),
            ]
        else:
            values += [None, None, None]
    out_cls = cfg["output_class"]
    values += [
        out_cls,
        int(agg["roi_mode_counts"][out_zone].get(out_cls, 0)),
        round(float(agg["roi_support"][out_zone].get(out_cls, 0.0)), 4),
    ]
    return values


class ExcelSessionWriter:
    """Compact workbook: no frame-by-frame raw sheets and no verbose Cycle_State sheet."""

    def __init__(self, path: str, session_id: str):
        self.path = path
        self.session_id = session_id
        self.wb = Workbook()
        self.wb.remove(self.wb.active)
        self.aggregated_rows_since_save = 0

        state_headers = [
            "session_id", "station_id", "workstation", "window_index",
            "window_start_s", "window_end_s", "sample_count", "expected_samples",
            "window_complete",
            "input_1_class", "input_1_count", "input_1_support",
            "input_2_class", "input_2_count", "input_2_support",
            "input_3_class", "input_3_count", "input_3_support",
            "output_class", "output_count", "output_support", "representative_frame",
        ]
        self.state_headers = state_headers

        self.ws_latest = self.wb.create_sheet("Latest_State")
        self.ws_latest.append(state_headers)
        _style_header(self.ws_latest)
        for sid, cfg in STATIONS.items():
            initial = [session_id, sid, cfg["name"], -1, 0.0, 0.0, 0, SAMPLES_PER_WINDOW, False]
            initial += [None, None, None] * 3
            initial += [cfg["cycle"]["output_class"], 0, 0.0, ""]
            self.ws_latest.append(initial)
            _style_row(self.ws_latest, self.ws_latest.max_row, green=True)

        self.ws_history = self.wb.create_sheet("State_History")
        self.ws_history.append(state_headers)
        _style_header(self.ws_history)

        self.cycle_headers = [
            "session_id", "station_id", "workstation", "cycle_id",
            "start_time_s", "end_time_s", "duration_s",
            "input_classes", "input_before", "input_after_start",
            "output_class", "output_before", "output_after_end",
            "start_confirmations", "end_confirmations", "start_frame", "end_frame",
        ]
        self.ws_cycles = self.wb.create_sheet("Cycles")
        self.ws_cycles.append(self.cycle_headers)
        _style_header(self.ws_cycles)

        self.opc_snapshot_headers = [
            "snapshot_id", "published_at", "session_id", "opc_endpoint",
            "start_control", "station_id", "workstation", "roi_zone",
            "yolo_class", "opc_variable", "value", "support",
            "window_index", "window_start_s", "window_end_s",
            "window_complete", "sample_count", "expected_samples", "quality_ok"
        ]
        self.ws_opc_snapshots = self.wb.create_sheet("OPC_Snapshots")
        self.ws_opc_snapshots.append(self.opc_snapshot_headers)
        _style_header(self.ws_opc_snapshots)

        self.ws_opc_mapping = self.wb.create_sheet("OPC_Mapping")
        self.ws_opc_mapping.append([
            "station_id", "workstation", "roi_zone", "yolo_class",
            "opc_variable", "node_id_template", "namespace_uri"
        ])
        _style_header(self.ws_opc_mapping)
        for item in OPC_MAPPING:
            sid = item["station_id"]
            self.ws_opc_mapping.append([
                sid, STATIONS[sid]["name"], item["zone"], item["yolo_class"],
                item["opc_variable"], f"vision.{item['opc_variable']}", OPC_NAMESPACE_URI
            ])

        self.ws_config = self.wb.create_sheet("Configuration")
        self.ws_config.append(["section", "parameter", "value"])
        _style_header(self.ws_config)
        config_rows = [
            ["session", "session_id", session_id],
            ["session", "created_at", datetime.now().isoformat(timespec="milliseconds")],
            ["model", "model_path", MODEL_PATH],
            ["model", "inference_imgsz", INFERENCE_IMGSZ],
            ["model", "yolo_device", str(YOLO_DEVICE)],
            ["sampling", "target_sample_fps", TARGET_SAMPLE_FPS],
            ["sampling", "aggregation_window_s", AGGREGATION_WINDOW_S],
            ["sampling", "samples_per_window_target", SAMPLES_PER_WINDOW],
            ["sampling", "capture_width", CAPTURE_WIDTH],
            ["cycle", "cycle_confirmations", CYCLE_CONFIRMATIONS],
            ["cycle", "cycle_min_duration_s", CYCLE_MIN_DURATION_S],
            ["cycle", "timing_basis", "raw video timestamps; first sample of confirmed event"],
            ["state_quality", "min_support", STATE_MIN_SUPPORT],
            ["state_quality", "min_samples", STATE_MIN_SAMPLES],
            ["video", "video_prep_mode", VIDEO_PREP_MODE],
            ["opc", "mode", "ON-DEMAND SNAPSHOT"],
            ["opc", "endpoint", OPC_ENDPOINT_PUBLIC],
            ["opc", "namespace_uri", OPC_NAMESPACE_URI],
        ]
        for row in config_rows:
            self.ws_config.append(row)
        for sid, cfg in STATIONS.items():
            for zone in ("stock", "output"):
                rect = stations[sid].rois.get(zone)
                if rect:
                    self.ws_config.append([
                        "roi", f"{sid}.{zone}",
                        f"x1={rect['x1']:.6f}; y1={rect['y1']:.6f}; x2={rect['x2']:.6f}; y2={rect['y2']:.6f}"
                    ])

        for ws in self.wb.worksheets:
            self._autosize_reasonably(ws)

    @staticmethod
    def _autosize_reasonably(ws) -> None:
        for idx, cell in enumerate(ws[1], start=1):
            text = str(cell.value or "")
            ws.column_dimensions[get_column_letter(idx)].width = min(max(12, len(text) + 2), 28)

    # Kept as a no-op so the inference path stays simple; raw predictions are no longer
    # written to Excel. This is the main workbook-size reduction.
    def append_raw(self, row: dict[str, Any]) -> None:
        return

    @staticmethod
    def _set_frame_hyperlink(ws, row_idx: int, col_idx: int, station_id: str, frame_file: str) -> None:
        if not frame_file:
            return
        frame_path = os.path.join(FRAMES_DIR, STATIONS[station_id]["folder"], frame_file)
        cell = ws.cell(row=row_idx, column=col_idx)
        cell.value = frame_file
        try:
            cell.hyperlink = Path(frame_path).resolve().as_uri()
            cell.style = "Hyperlink"
        except Exception:
            pass

    def append_aggregate(self, agg: dict[str, Any]) -> None:
        row_values = [
            self.session_id,
            agg["station_id"],
            STATIONS[agg["station_id"]]["name"],
            agg["window_index"],
            round(agg["timestamp_start_s"], 4),
            round(agg["timestamp_end_s"], 4),
            agg["sample_count"],
            agg["expected_samples"],
            agg["window_complete"],
        ] + _state_fields(agg["station_id"], agg) + [agg["representative_frame"]]

        self.ws_history.append(row_values)
        hrow = self.ws_history.max_row
        _style_row(self.ws_history, hrow, green=True)
        frame_col = self.state_headers.index("representative_frame") + 1
        self._set_frame_hyperlink(self.ws_history, hrow, frame_col, agg["station_id"], agg["representative_frame"])

        latest_row = list(STATIONS.keys()).index(agg["station_id"]) + 2
        for col_idx, value in enumerate(row_values, start=1):
            self.ws_latest.cell(row=latest_row, column=col_idx, value=value)
        _style_row(self.ws_latest, latest_row, green=True)
        self._set_frame_hyperlink(self.ws_latest, latest_row, frame_col, agg["station_id"], agg["representative_frame"])

        self.aggregated_rows_since_save += 1
        if self.aggregated_rows_since_save >= EXCEL_SAVE_EVERY_AGGREGATED_ROWS:
            self.save()

    # Current cycle state is shown in the web app; it is intentionally not duplicated
    # as a verbose Excel sheet anymore.
    def update_cycle_state(self, st: StationRuntime) -> None:
        return

    def append_cycle(self, cycle: dict[str, Any]) -> None:
        values = [cycle.get(h) for h in self.cycle_headers]
        self.ws_cycles.append(values)
        row = self.ws_cycles.max_row
        _style_row(self.ws_cycles, row, green=True)
        for key in ("start_frame", "end_frame"):
            col = self.cycle_headers.index(key) + 1
            frame_file = cycle.get(key)
            if frame_file:
                self._set_frame_hyperlink(self.ws_cycles, row, col, cycle["station_id"], frame_file)
        self.save()

    def append_opc_snapshot(self, snapshot: dict[str, Any]) -> None:
        published_at = datetime.now().isoformat(timespec="milliseconds")
        for row in snapshot.get("rows", []):
            values = [
                snapshot.get("snapshot_id"), published_at, snapshot.get("session_id"),
                OPC_ENDPOINT_PUBLIC, True, row.get("station_id"), row.get("workstation"),
                row.get("zone"), row.get("yolo_class"), row.get("opc_variable"),
                row.get("value"), row.get("support"), row.get("window_index"),
                row.get("window_start_s"), row.get("window_end_s"),
                row.get("window_complete"), row.get("sample_count"),
                row.get("expected_samples"), row.get("quality_ok"),
            ]
            self.ws_opc_snapshots.append(values)
            _style_row(self.ws_opc_snapshots, self.ws_opc_snapshots.max_row, green=True)
        self.save()

    def save(self) -> None:
        tmp_path = self.path + ".tmp.xlsx"
        self.wb.save(tmp_path)
        os.replace(tmp_path, self.path)
        self.aggregated_rows_since_save = 0
        with state_lock:
            session["excel_ready"] = True

    def close(self) -> None:
        self.save()
        self.wb.close()


# =============================================================================
# Fast ROI-based cycle detection (independent from 2-second aggregation)
# =============================================================================
def _raw_cycle_metric(roi_counts: dict[str, dict[str, int]], zone: str, classes: list[str]) -> int:
    return sum(int(roi_counts.get(zone, {}).get(cls, 0)) for cls in classes)


def _reset_start_candidate(c: CycleRuntime) -> None:
    c.start_candidate_count = 0
    c.start_candidate_time_s = None
    c.start_candidate_input = None
    c.start_candidate_jpeg = None
    c.start_candidate_detections = []


def _reset_end_candidate(c: CycleRuntime) -> None:
    c.end_candidate_count = 0
    c.end_candidate_time_s = None
    c.end_candidate_output = None
    c.end_candidate_jpeg = None
    c.end_candidate_detections = []


def _save_cycle_evidence(
    station_id: str,
    cycle_id: int,
    event: str,
    t_video: float,
    jpeg_bytes: bytes | None,
    detections: list[dict[str, Any]],
) -> str | None:
    if not jpeg_bytes:
        return None
    safe_t = int(round(max(0.0, t_video) * 1000.0))
    name = f"cycle_{cycle_id:04d}_{event}_{safe_t:010d}ms.jpg"
    path = os.path.join(FRAMES_DIR, STATIONS[station_id]["folder"], name)
    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        try:
            with open(path, "wb") as f:
                f.write(jpeg_bytes)
            return name
        except OSError:
            return None
    annotated = _draw_boxes(frame, detections)
    annotated = _draw_rois(annotated, stations[station_id].rois)
    cv2.imwrite(path, annotated, [cv2.IMWRITE_JPEG_QUALITY, EVIDENCE_JPEG_QUALITY])
    return name


def _process_cycle_sample(
    station_id: str,
    t_video: float,
    roi_counts: dict[str, dict[str, int]],
    jpeg_bytes: bytes | None,
    detections: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Detect START/END quickly from consecutive raw ROI-count predictions.

    START = input count below the idle baseline for CYCLE_CONFIRMATIONS consecutive samples.
    END   = output count above the minimum seen since START for CYCLE_CONFIRMATIONS samples.

    The timestamp stored is the FIRST sample of the confirmed event, so confirmation does
    not artificially lengthen/shorten the measured cycle.
    """
    st = stations[station_id]
    c = st.cycle
    cfg = STATIONS[station_id]["cycle"]
    input_value = _raw_cycle_metric(roi_counts, cfg["input_zone"], cfg["input_classes"])
    output_value = _raw_cycle_metric(roi_counts, cfg["output_zone"], [cfg["output_class"]])
    c.prev_input_count = input_value
    c.prev_output_count = output_value

    # First sample gives us an immediate baseline; no 2-second wait is required.
    if c.input_before is None and not c.active and c.completed_cycles == 0 and c.start_candidate_count == 0:
        # We use output_min_since_start only while active; input_before doubles as the
        # current idle baseline until a cycle is started.
        c.input_before = input_value
        c.output_min_since_start = output_value
        c.status = "IDLE"
        c.last_event = f"Baseline ready: input={input_value}, output={output_value}"
        return None

    completed: dict[str, Any] | None = None

    if not c.active:
        baseline_input = int(c.input_before if c.input_before is not None else input_value)

        # Restocking / a higher stable count becomes the new baseline immediately. A
        # start still needs two lower samples, so one-frame positive spikes are unlikely
        # to generate a completed cycle by themselves.
        if input_value > baseline_input:
            c.input_before = input_value
            c.output_min_since_start = output_value
            _reset_start_candidate(c)
            c.status = "IDLE"
            c.last_event = f"Idle baseline updated: input={input_value}, output={output_value}"
            return None

        if input_value < baseline_input:
            if c.start_candidate_count == 0:
                c.start_candidate_time_s = float(t_video)
                c.start_candidate_input = int(input_value)
                c.start_candidate_jpeg = bytes(jpeg_bytes) if jpeg_bytes else None
                c.start_candidate_detections = [dict(d) for d in detections]
            c.start_candidate_count += 1
            c.status = "START_CANDIDATE"
            c.last_event = (
                f"Start candidate {c.start_candidate_count}/{CYCLE_CONFIRMATIONS}: "
                f"input {baseline_input}->{input_value} at {t_video:.2f}s"
            )

            if c.start_candidate_count >= CYCLE_CONFIRMATIONS:
                cycle_id = c.completed_cycles + 1
                start_time = float(c.start_candidate_time_s if c.start_candidate_time_s is not None else t_video)
                c.active = True
                c.status = "ASSEMBLING"
                c.current_cycle_id = cycle_id
                c.start_time_s = start_time
                c.start_window = int(math.floor(start_time / AGGREGATION_WINDOW_S))
                c.input_after_start = int(c.start_candidate_input if c.start_candidate_input is not None else input_value)
                start_jpeg = c.start_candidate_jpeg
                start_dets = list(c.start_candidate_detections)
                c.start_frame = _save_cycle_evidence(
                    station_id, cycle_id, "start", start_time, start_jpeg, start_dets
                )
                c.output_min_since_start = int(output_value)
                c.last_event = (
                    f"START cycle {cycle_id}: input {baseline_input}->{c.input_after_start} "
                    f"confirmed x{CYCLE_CONFIRMATIONS} at {start_time:.2f}s"
                )
                _reset_start_candidate(c)
                _reset_end_candidate(c)
        else:
            _reset_start_candidate(c)
            c.status = "IDLE"
            c.last_event = f"Idle: input={input_value}, output={output_value}"

    else:
        if c.output_min_since_start is None:
            c.output_min_since_start = output_value
        else:
            c.output_min_since_start = min(int(c.output_min_since_start), int(output_value))
        output_baseline = int(c.output_min_since_start)

        if output_value > output_baseline:
            if c.end_candidate_count == 0:
                c.end_candidate_time_s = float(t_video)
                c.end_candidate_output = int(output_value)
                c.end_candidate_jpeg = bytes(jpeg_bytes) if jpeg_bytes else None
                c.end_candidate_detections = [dict(d) for d in detections]
            c.end_candidate_count += 1
            c.last_event = (
                f"End candidate {c.end_candidate_count}/{CYCLE_CONFIRMATIONS}: "
                f"output {output_baseline}->{output_value} at {t_video:.2f}s"
            )

            if c.end_candidate_count >= CYCLE_CONFIRMATIONS:
                end_time = float(c.end_candidate_time_s if c.end_candidate_time_s is not None else t_video)
                duration = max(0.0, end_time - float(c.start_time_s or end_time))
                if duration >= CYCLE_MIN_DURATION_S:
                    cycle_id = int(c.current_cycle_id or (c.completed_cycles + 1))
                    end_frame = _save_cycle_evidence(
                        station_id,
                        cycle_id,
                        "end",
                        end_time,
                        c.end_candidate_jpeg,
                        list(c.end_candidate_detections),
                    )
                    completed = {
                        "session_id": session.get("session_id"),
                        "station_id": station_id,
                        "workstation": st.name,
                        "cycle_id": cycle_id,
                        "start_time_s": round(float(c.start_time_s or 0.0), 3),
                        "end_time_s": round(end_time, 3),
                        "duration_s": round(duration, 3),
                        "input_classes": ", ".join(cfg["input_classes"]),
                        "input_before": c.input_before,
                        "input_after_start": c.input_after_start,
                        "output_class": cfg["output_class"],
                        "output_before": output_baseline,
                        "output_after_end": int(c.end_candidate_output if c.end_candidate_output is not None else output_value),
                        "start_confirmations": CYCLE_CONFIRMATIONS,
                        "end_confirmations": CYCLE_CONFIRMATIONS,
                        "start_frame": c.start_frame,
                        "end_frame": end_frame,
                    }
                    c.completed_cycles += 1
                    c.last_cycle_id = cycle_id
                    c.last_cycle_duration_s = duration
                    c.last_event = f"END cycle {cycle_id}: duration={duration:.2f}s at {end_time:.2f}s"
                    c.active = False
                    c.status = "COMPLETED"
                    c.current_cycle_id = None
                    c.start_window = None
                    c.start_time_s = None
                    c.start_frame = None
                    # Current input becomes the baseline for the next cycle.
                    c.input_before = int(input_value)
                    c.input_after_start = None
                    c.output_min_since_start = int(output_value)
                    _reset_start_candidate(c)
                    _reset_end_candidate(c)
                else:
                    c.last_event = (
                        f"Output rise ignored: duration {duration:.2f}s < {CYCLE_MIN_DURATION_S:.2f}s"
                    )
                    _reset_end_candidate(c)
        else:
            _reset_end_candidate(c)

    return completed


def _mark_incomplete_cycle(station_id: str, reason: str) -> None:
    st = stations[station_id]
    c = st.cycle
    if c.active:
        c.status = "INCOMPLETE"
        c.last_event = reason


# =============================================================================
# Aggregation
# =============================================================================
def _aggregate_window(station_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot aggregate an empty window")

    mode_counts: dict[str, int] = {}
    support: dict[str, float] = {}
    for cls in model_classes:
        vals = [int(r["counts"].get(cls, 0)) for r in rows]
        m = _safe_mode(vals)
        mode_counts[cls] = m
        support[cls] = sum(v == m for v in vals) / len(vals)

    roi_mode_counts = {"stock": {}, "output": {}}
    roi_support = {"stock": {}, "output": {}}
    for zone in ("stock", "output"):
        for cls in model_classes:
            vals = [int(r["roi_counts"][zone].get(cls, 0)) for r in rows]
            m = _safe_mode(vals)
            roi_mode_counts[zone][cls] = m
            roi_support[zone][cls] = sum(v == m for v in vals) / len(vals)

    total_mode = _safe_mode([int(r["total_objects"]) for r in rows])
    stock_total_mode = _safe_mode([sum(r["roi_counts"]["stock"].values()) for r in rows])
    output_total_mode = _safe_mode([sum(r["roi_counts"]["output"].values()) for r in rows])

    relevant = STATIONS[station_id]["relevant_classes"]

    def distance(row: dict[str, Any]) -> int:
        d = sum(abs(int(row["counts"].get(cls, 0)) - mode_counts[cls]) for cls in relevant)
        for zone in ("stock", "output"):
            d += sum(
                abs(int(row["roi_counts"][zone].get(cls, 0)) - roi_mode_counts[zone][cls])
                for cls in relevant
            )
        return d

    representative = min(rows, key=distance)
    window_index = int(rows[0]["window_index"])
    window_start_s = window_index * AGGREGATION_WINDOW_S
    sample_count = len(rows)
    window_complete = sample_count >= SAMPLES_PER_WINDOW
    window_end_s = (
        window_start_s + AGGREGATION_WINDOW_S
        if window_complete
        else max(r["video_time_s"] for r in rows)
    )

    # Create only ONE annotated evidence frame for this 2-second window.
    raw_jpeg = representative["jpeg_bytes"]
    frame = cv2.imdecode(np.frombuffer(raw_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    evidence_name = f"window_{window_index:06d}_sample_{representative['sample_index']:08d}.jpg"
    evidence_path = os.path.join(FRAMES_DIR, STATIONS[station_id]["folder"], evidence_name)
    if frame is not None:
        annotated = _draw_boxes(frame, representative["detections"])
        annotated = _draw_rois(annotated, stations[station_id].rois)
        cv2.imwrite(evidence_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, EVIDENCE_JPEG_QUALITY])
    else:
        # Keep the evidence filename traceable even if decoding unexpectedly fails.
        with open(evidence_path, "wb") as f:
            f.write(raw_jpeg)

    return {
        "station_id": station_id,
        "window_index": window_index,
        "timestamp_start_s": window_start_s,
        "timestamp_end_s": window_end_s,
        "sample_count": sample_count,
        "expected_samples": SAMPLES_PER_WINDOW,
        "window_complete": window_complete,
        "total_objects_mode": total_mode,
        "stock_roi_total_mode": stock_total_mode,
        "output_roi_total_mode": output_total_mode,
        "mode_counts": mode_counts,
        "support": support,
        "roi_mode_counts": roi_mode_counts,
        "roi_support": roi_support,
        "representative_sample_index": representative["sample_index"],
        "representative_frame": evidence_name,
    }


def _flush_station_buffer(station_id: str) -> None:
    global station_buffers
    rows = station_buffers[station_id]
    if not rows or excel_writer is None:
        return

    agg = _aggregate_window(station_id, rows)
    st = stations[station_id]
    st.aggregated_windows += 1
    st.latest_aggregate = {
        "window_index": agg["window_index"],
        "timestamp_start_s": round(agg["timestamp_start_s"], 3),
        "timestamp_end_s": round(agg["timestamp_end_s"], 3),
        "sample_count": agg["sample_count"],
        "expected_samples": agg["expected_samples"],
        "window_complete": agg["window_complete"],
        "total_objects_mode": agg["total_objects_mode"],
        "stock_roi_total_mode": agg["stock_roi_total_mode"],
        "output_roi_total_mode": agg["output_roi_total_mode"],
        "mode_counts": dict(agg["mode_counts"]),
        "support": {k: round(v, 3) for k, v in agg["support"].items()},
        "roi_mode_counts": {z: dict(v) for z, v in agg["roi_mode_counts"].items()},
        "roi_support": {z: {k: round(v, 3) for k, v in vals.items()} for z, vals in agg["roi_support"].items()},
        "representative_frame": agg["representative_frame"],
    }

    excel_writer.append_aggregate(agg)
    station_buffers[station_id] = []


def _log_opc_snapshot(snapshot: dict[str, Any]) -> None:
    """Persist an OPC initialization snapshot without making Excel the data source."""
    global excel_writer
    if excel_writer is not None:
        # Writer is active during the normal hourly/on-demand workflow.
        with analysis_lock:
            if excel_writer is not None:
                excel_writer.append_opc_snapshot(snapshot)
        return

    # If analysis already ended, append to the saved workbook if available.
    if not os.path.exists(EXCEL_PATH):
        return
    try:
        wb = load_workbook(EXCEL_PATH)
        ws = wb["OPC_Snapshots"] if "OPC_Snapshots" in wb.sheetnames else wb.create_sheet("OPC_Snapshots")
        headers = [
            "snapshot_id", "published_at", "session_id", "opc_endpoint",
            "start_control", "station_id", "workstation", "roi_zone",
            "yolo_class", "opc_variable", "value", "support",
            "window_index", "window_start_s", "window_end_s",
            "window_complete", "sample_count", "expected_samples", "quality_ok"
        ]
        if ws.max_row == 1 and ws.cell(1, 1).value is None:
            ws.append(headers)
            _style_header(ws)
        published_at = datetime.now().isoformat(timespec="milliseconds")
        for row in snapshot.get("rows", []):
            ws.append([
                snapshot.get("snapshot_id"), published_at, snapshot.get("session_id"),
                OPC_ENDPOINT_PUBLIC, True, row.get("station_id"), row.get("workstation"),
                row.get("zone"), row.get("yolo_class"), row.get("opc_variable"),
                row.get("value"), row.get("support"), row.get("window_index"),
                row.get("window_start_s"), row.get("window_end_s"),
                row.get("window_complete"), row.get("sample_count"),
                row.get("expected_samples"), row.get("quality_ok"),
            ])
        tmp = EXCEL_PATH + ".opc.tmp.xlsx"
        wb.save(tmp)
        wb.close()
        os.replace(tmp, EXCEL_PATH)
    except Exception as exc:
        print(f"[WARN] OPC snapshot published but Excel snapshot log failed: {exc}")


# =============================================================================
# Flask routes
# =============================================================================
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/video/<path:filename>")
def serve_video(filename: str):
    # conditional=True enables byte-range requests used by native HTML5 playback.
    return send_from_directory(UPLOAD_DIR, filename, conditional=True)


@app.route("/api/config")
def api_config():
    ensure_model_loaded()
    return jsonify({
        "stations": [
            {
                "id": sid,
                "name": cfg["name"],
                "relevant_classes": cfg["relevant_classes"],
            }
            for sid, cfg in STATIONS.items()
        ],
        "target_sample_fps": TARGET_SAMPLE_FPS,
        "aggregation_window_s": AGGREGATION_WINDOW_S,
        "samples_per_window": SAMPLES_PER_WINDOW,
        "capture_width": CAPTURE_WIDTH,
        "capture_jpeg_quality": CAPTURE_JPEG_QUALITY,
        "inference_imgsz": INFERENCE_IMGSZ,
        "yolo_device": str(YOLO_DEVICE),
        "cuda_available": bool(torch.cuda.is_available()),
        "model_classes": model_classes,
        "roi_zones": ["stock", "output"],
        "video_prep_mode": VIDEO_PREP_MODE,
        "state_min_support": STATE_MIN_SUPPORT,
        "state_min_samples": STATE_MIN_SAMPLES,
        "cycle_confirmations": CYCLE_CONFIRMATIONS,
        "cycle_min_duration_s": CYCLE_MIN_DURATION_S,
        "opc_ua": "on-demand snapshot",
        "opc_available": ASYNCUA_AVAILABLE,
        "opc_endpoint": OPC_ENDPOINT_PUBLIC,
        "opc_namespace_uri": OPC_NAMESPACE_URI,
        "opc_mapping": _opc_mapping_public(),
    })


@app.route("/api/upload/<station_id>", methods=["POST"])
def upload_station_video(station_id: str):
    if station_id not in STATIONS:
        return jsonify({"error": "Unknown station"}), 404
    with state_lock:
        if session["status"] == "running":
            return jsonify({"error": "Stop the analysis before replacing a video."}), 409

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(file.filename)[1].lower() or ".mp4"
    uid = uuid.uuid4().hex[:10]
    raw_path = os.path.join(UPLOAD_DIR, f"{station_id}_raw_{uid}{ext}")
    ready_path = os.path.join(UPLOAD_DIR, f"{station_id}_ready_{uid}.mp4")
    file.save(raw_path)

    served_path = raw_path
    transcoded = False
    warning = None
    prepare_method = "direct"
    t_prepare = time.perf_counter()

    if VIDEO_PREP_MODE != "never":
        if ffmpeg_available():
            ok, err, method = prepare_video_for_browser(raw_path, ready_path)
            if ok:
                served_path = ready_path
                transcoded = method == "reencode-ultrafast"
                prepare_method = method
                try:
                    os.remove(raw_path)
                except OSError:
                    pass
            else:
                # MP4 may still play directly; MKV usually will not play consistently in browsers.
                if ext == ".mp4":
                    warning = f"FFmpeg preparation failed; trying original MP4. {err[:260]}"
                    prepare_method = "fallback-original-mp4"
                else:
                    try:
                        os.remove(raw_path)
                    except OSError:
                        pass
                    return jsonify({
                        "error": (
                            "Could not prepare this non-MP4 video for browser playback. "
                            f"FFmpeg error: {err[:500]}"
                        )
                    }), 400
        else:
            if ext != ".mp4":
                try:
                    os.remove(raw_path)
                except OSError:
                    pass
                return jsonify({
                    "error": (
                        "FFmpeg is required for MKV/non-MP4 uploads but was not found. "
                        "Install FFmpeg and ensure ffmpeg/ffprobe are in PATH, or set FFMPEG_BIN/FFPROBE_BIN."
                    )
                }), 400
            warning = "FFmpeg not found; serving original MP4 without timestamp/container repair."
            prepare_method = "direct-no-ffmpeg"

    prepare_seconds = time.perf_counter() - t_prepare

    try:
        info = _probe_video(served_path)
    except Exception as exc:
        for path in (raw_path, ready_path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        return jsonify({"error": f"Invalid/unreadable video after preparation: {exc}"}), 400

    with state_lock:
        st = stations[station_id]
        _cleanup_previous_upload(st)
        st.video_path = served_path
        st.video_url = f"/video/{os.path.basename(served_path)}"
        st.original_name = file.filename
        st.stored_paths = [served_path]
        st.source_fps = info["fps"]
        st.frame_count = info["frame_count"]
        st.duration_s = info["duration_s"]
        st.width = info["width"]
        st.height = info["height"]
        st.loaded = True
        st.warning = warning
        st.transcoded = transcoded
        st.prepare_method = prepare_method
        st.prepare_seconds = prepare_seconds
        _reset_station_processing(st)
        public = _station_public_dict(st)

    print(
        f"[INFO] {station_id} upload ready: {file.filename} -> {os.path.basename(served_path)} | "
        f"method={prepare_method} | {prepare_seconds:.2f}s"
    )
    return jsonify({"ok": True, "station": public})


@app.route("/api/roi/<station_id>", methods=["POST", "DELETE"])
def station_roi(station_id: str):
    if station_id not in STATIONS:
        return jsonify({"error": "Unknown station"}), 404
    with state_lock:
        if session["status"] == "running":
            return jsonify({"error": "ROI zones cannot be changed while analysis is running."}), 409
        st = stations[station_id]
        if request.method == "DELETE":
            st.rois = {}
            _save_roi_config()
            return jsonify({"ok": True, "station": _station_public_dict(st)})

        payload = request.get_json(silent=True) or {}
        zone = str(payload.get("zone", "")).lower()
        if zone not in ("stock", "output"):
            return jsonify({"error": "zone must be 'stock' or 'output'"}), 400
        try:
            rect = _validate_normalized_rect(payload.get("rect") or {})
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        st.rois[zone] = rect
        _save_roi_config()
        return jsonify({"ok": True, "station": _station_public_dict(st)})


@app.route("/api/start", methods=["POST"])
def start_analysis():
    global excel_writer, station_buffers, previous_window
    ensure_model_loaded()

    with analysis_lock:
        with state_lock:
            if session["status"] == "running":
                return jsonify({"error": "Analysis is already running"}), 409
            missing = [sid for sid, st in stations.items() if not st.loaded]
            if missing:
                return jsonify({"error": f"Load all four videos first. Missing: {', '.join(missing)}"}), 400
            missing_rois = [sid for sid, st in stations.items() if not all(z in st.rois for z in ("stock", "output"))]
            if missing_rois:
                return jsonify({"error": f"Draw both ROI zones for: {', '.join(missing_rois)}"}), 400

            payload = request.get_json(silent=True) or {}
            min_confidence = max(0.01, min(0.99, float(payload.get("min_confidence", 0.50))))

            for st in stations.values():
                _reset_station_processing(st)
                st.status = "processing"

            if excel_writer is not None:
                try:
                    excel_writer.close()
                except Exception:
                    pass
                excel_writer = None

            if os.path.exists(EXCEL_PATH):
                try:
                    os.remove(EXCEL_PATH)
                except PermissionError:
                    return jsonify({"error": "Close assembly_4stations.xlsx before starting a new session."}), 409

            for cfg in STATIONS.values():
                folder = os.path.join(FRAMES_DIR, cfg["folder"])
                if os.path.isdir(folder):
                    for name in os.listdir(folder):
                        path = os.path.join(folder, name)
                        if os.path.isfile(path):
                            try:
                                os.remove(path)
                            except OSError:
                                pass

            session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
            session.update({
                "session_id": session_id,
                "status": "running",
                "started_at": datetime.now().isoformat(timespec="milliseconds"),
                "finished_at": None,
                "min_confidence": min_confidence,
                "source_time_s": 0.0,
                "wall_elapsed_s": 0.0,
                "predict_batches": 0,
                "images_processed": 0,
                "last_batch_inference_ms": None,
                "last_batch_size": 0,
                "excel_ready": False,
                "error": None,
            })

            station_buffers = {sid: [] for sid in STATIONS}
            previous_window = {sid: None for sid in STATIONS}
            excel_writer = ExcelSessionWriter(EXCEL_PATH, session_id)
            # Create a valid compact workbook immediately; later saves are much less frequent.
            excel_writer.save()

    return jsonify({"ok": True, "session_id": session["session_id"]})


@app.route("/api/predict_batch", methods=["POST"])
def predict_batch():
    global previous_window
    ensure_model_loaded()

    with state_lock:
        if session["status"] != "running":
            return jsonify({"error": "Analysis is not running"}), 409
        session_id = session["session_id"]
        session_conf = float(session["min_confidence"])

    try:
        meta = json.loads(request.form.get("meta", "{}"))
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid multipart meta JSON"}), 400

    frame_meta = meta.get("frames") or []
    if not isinstance(frame_meta, list) or not frame_meta:
        return jsonify({"error": "No frame metadata provided"}), 400

    min_confidence = max(0.01, min(0.99, float(meta.get("min_confidence", session_conf))))

    decoded_frames: list[np.ndarray] = []
    batch_info: list[dict[str, Any]] = []

    for item in frame_meta:
        sid = str(item.get("station_id", ""))
        if sid not in STATIONS:
            continue
        with state_lock:
            st = stations[sid]
            if not st.loaded or st.status == "completed":
                continue
        upload = request.files.get(f"frame_{sid}")
        if upload is None:
            continue
        jpeg_bytes = upload.read()
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        video_time_s = max(0.0, float(item.get("video_time_s", 0.0)))
        decoded_frames.append(frame)
        batch_info.append({
            "station_id": sid,
            "video_time_s": video_time_s,
            "jpeg_bytes": jpeg_bytes,
        })

    if not decoded_frames:
        return jsonify({"error": "No decodable active frames in batch"}), 400

    request_started = time.perf_counter()

    # One lock + one YOLO call = one shared model, one batch.
    with analysis_lock:
        with state_lock:
            if session["status"] != "running" or session["session_id"] != session_id:
                return jsonify({"error": "Session changed while request was waiting"}), 409

        inference_t0 = time.perf_counter()
        results = model(
            decoded_frames,
            conf=min_confidence,
            imgsz=INFERENCE_IMGSZ,
            device=YOLO_DEVICE,
            verbose=False,
        )  # type: ignore[misc]
        inference_ms = (time.perf_counter() - inference_t0) * 1000.0

        output_results: list[dict[str, Any]] = []

        for result, frame, info in zip(results, decoded_frames, batch_info):
            sid = info["station_id"]
            t_video = float(info["video_time_s"])
            st = stations[sid]
            detections, counts, avg_conf = _parse_result(result, min_confidence)
            roi_counts = _roi_counts(detections, frame.shape, st.rois)

            # Cycle timing is independent from the 2-second cleaned state. It reacts
            # after only CYCLE_CONFIRMATIONS consecutive raw ROI-count observations.
            completed_cycle = _process_cycle_sample(
                sid, t_video, roi_counts, info["jpeg_bytes"], detections
            )
            if completed_cycle is not None and excel_writer is not None:
                excel_writer.append_cycle(completed_cycle)

            window_index = int(math.floor(t_video / AGGREGATION_WINDOW_S))

            prev = previous_window[sid]
            if prev is not None and window_index != prev and station_buffers[sid]:
                _flush_station_buffer(sid)
            previous_window[sid] = window_index

            sample_index = st.samples_processed
            sample_in_window = len(station_buffers[sid])
            source_frame_index = int(round(t_video * st.source_fps))
            row = {
                "station_id": sid,
                "sample_index": sample_index,
                "video_time_s": t_video,
                "window_index": window_index,
                "sample_in_window": sample_in_window,
                "source_frame_index": source_frame_index,
                "source_fps": st.source_fps,
                "wall_timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "capture_width": int(frame.shape[1]),
                "capture_height": int(frame.shape[0]),
                "total_objects": len(detections),
                "avg_confidence": avg_conf,
                "counts": counts,
                "roi_counts": roi_counts,
                "detections": detections,
                "jpeg_bytes": info["jpeg_bytes"],
            }
            station_buffers[sid].append(row)
            if excel_writer is not None:
                excel_writer.append_raw(row)

            st.current_video_time_s = t_video
            st.current_source_frame_index = source_frame_index
            st.samples_processed += 1
            st.progress_pct = min(100.0, 100.0 * t_video / max(st.duration_s, 1e-9))
            st.latest_counts = {k: int(v) for k, v in counts.items() if v > 0}
            st.latest_roi_counts = {
                z: {k: int(v) for k, v in vals.items() if v > 0}
                for z, vals in roi_counts.items()
            }
            st.latest_total = len(detections)
            st.latest_average_confidence = avg_conf

            output_results.append({
                "station_id": sid,
                "video_time_s": round(t_video, 4),
                "capture_width": int(frame.shape[1]),
                "capture_height": int(frame.shape[0]),
                "detections": detections,
                "counts": {k: int(v) for k, v in counts.items() if v > 0},
                "roi_counts": {
                    z: {k: int(v) for k, v in vals.items() if v > 0}
                    for z, vals in roi_counts.items()
                },
                "avg_confidence": round(avg_conf, 4),
            })

        now = time.perf_counter()
        with state_lock:
            started_at_wall = session.get("_started_perf")
            if started_at_wall is None:
                session["_started_perf"] = now
                started_at_wall = now
            session["source_time_s"] = round(max(i["video_time_s"] for i in batch_info), 3)
            session["wall_elapsed_s"] = round(now - float(started_at_wall), 3)
            session["predict_batches"] = int(session.get("predict_batches", 0)) + 1
            session["images_processed"] = int(session.get("images_processed", 0)) + len(decoded_frames)
            session["last_batch_inference_ms"] = round(inference_ms, 1)
            session["last_batch_size"] = len(decoded_frames)

    roundtrip_server_ms = (time.perf_counter() - request_started) * 1000.0
    return jsonify({
        "ok": True,
        "session_id": session_id,
        "batch_size": len(decoded_frames),
        "inference_ms": round(inference_ms, 1),
        "server_ms": round(roundtrip_server_ms, 1),
        "results": output_results,
    })


@app.route("/api/finalize_station/<station_id>", methods=["POST"])
def finalize_station(station_id: str):
    if station_id not in STATIONS:
        return jsonify({"error": "Unknown station"}), 404
    with analysis_lock:
        with state_lock:
            if session["status"] != "running":
                return jsonify({"ok": True, "message": "Session is not running"})
        if station_buffers[station_id]:
            _flush_station_buffer(station_id)
        with state_lock:
            st = stations[station_id]
            _mark_incomplete_cycle(station_id, "Video ended before a matching output event completed the active cycle")
            st.status = "completed"
            st.progress_pct = 100.0
            if excel_writer is not None:
                excel_writer.update_cycle_state(st)
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def stop_analysis():
    global excel_writer
    with analysis_lock:
        with state_lock:
            if session["status"] != "running":
                return jsonify({"ok": True, "message": "No analysis is running"})

        for sid in STATIONS:
            if station_buffers[sid]:
                _flush_station_buffer(sid)

        for sid in STATIONS:
            _mark_incomplete_cycle(sid, "Analysis stopped before a matching output event completed the active cycle")
            if excel_writer is not None:
                excel_writer.update_cycle_state(stations[sid])

        if excel_writer is not None:
            excel_writer.close()
            excel_writer = None

        with state_lock:
            session["status"] = "stopped"
            session["finished_at"] = datetime.now().isoformat(timespec="milliseconds")
            session.pop("_started_perf", None)
            for st in stations.values():
                if st.status == "processing":
                    st.status = "stopped"

    return jsonify({"ok": True, "message": "Analysis stopped and Excel saved"})


@app.route("/api/complete", methods=["POST"])
def complete_analysis():
    global excel_writer
    with analysis_lock:
        with state_lock:
            if session["status"] != "running":
                return jsonify({"ok": True})
        for sid in STATIONS:
            if station_buffers[sid]:
                _flush_station_buffer(sid)
        for sid in STATIONS:
            _mark_incomplete_cycle(sid, "All videos ended before a matching output event completed the active cycle")
            if excel_writer is not None:
                excel_writer.update_cycle_state(stations[sid])
        if excel_writer is not None:
            excel_writer.close()
            excel_writer = None
        with state_lock:
            session["status"] = "completed"
            session["finished_at"] = datetime.now().isoformat(timespec="milliseconds")
            session.pop("_started_perf", None)
            for st in stations.values():
                st.status = "completed"
                st.progress_pct = 100.0
    return jsonify({"ok": True})


@app.route("/api/opc/preview")
def opc_preview():
    try:
        with state_lock:
            snapshot = _build_opc_snapshot()
        return jsonify({"ok": True, "snapshot": snapshot, "opc": opc_server.public_status()})
    except Exception as exc:
        return jsonify({"error": str(exc), "opc": opc_server.public_status()}), 409


@app.route("/api/opc/publish", methods=["POST"])
def opc_publish():
    """Start/reuse the embedded OPC server and publish one latest-state snapshot."""
    try:
        # Copy one coherent cleaned snapshot quickly; do not hold the inference lock
        # while the OPC server starts or while network values are written.
        with state_lock:
            snapshot = _build_opc_snapshot()
        opc_server.publish(snapshot["values"], snapshot["snapshot_id"])
        _log_opc_snapshot(snapshot)
        return jsonify({
            "ok": True,
            "message": "Latest cleaned detection state published to OPC UA; StartControl=True.",
            "snapshot": snapshot,
            "opc": opc_server.public_status(),
        })
    except Exception as exc:
        return jsonify({"error": str(exc), "opc": opc_server.public_status()}), 500


@app.route("/api/opc/stop", methods=["POST"])
def opc_stop():
    try:
        opc_server.stop()
        return jsonify({"ok": True, "opc": opc_server.public_status()})
    except Exception as exc:
        return jsonify({"error": str(exc), "opc": opc_server.public_status()}), 500


@app.route("/api/reset", methods=["POST"])
def reset_all():
    global excel_writer, station_buffers, previous_window
    # A full reset also closes the on-demand OPC endpoint.
    try:
        opc_server.stop()
    except Exception:
        pass
    with analysis_lock:
        with state_lock:
            if session["status"] == "running":
                return jsonify({"error": "Stop the analysis before resetting."}), 409

        if excel_writer is not None:
            try:
                excel_writer.close()
            except Exception:
                pass
            excel_writer = None

        with state_lock:
            for sid in list(stations.keys()):
                old = stations[sid]
                _cleanup_previous_upload(old)
                fresh = StationRuntime(sid, old.name)
                fresh.rois = dict(old.rois)
                stations[sid] = fresh

            session.clear()
            session.update({
                "session_id": None,
                "status": "idle",
                "started_at": None,
                "finished_at": None,
                "min_confidence": 0.50,
                "source_time_s": 0.0,
                "wall_elapsed_s": 0.0,
                "predict_batches": 0,
                "images_processed": 0,
                "last_batch_inference_ms": None,
                "last_batch_size": 0,
                "excel_ready": False,
                "error": None,
            })
            station_buffers = {sid: [] for sid in STATIONS}
            previous_window = {sid: None for sid in STATIONS}

        for cfg in STATIONS.values():
            folder = os.path.join(FRAMES_DIR, cfg["folder"])
            if os.path.isdir(folder):
                for name in os.listdir(folder):
                    path = os.path.join(folder, name)
                    if os.path.isfile(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass

        if os.path.exists(EXCEL_PATH):
            try:
                os.remove(EXCEL_PATH)
            except PermissionError:
                return jsonify({"error": "Reset done, but Excel is open and could not be deleted."}), 409

    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    with state_lock:
        public_session = {k: v for k, v in session.items() if not k.startswith("_")}
        return jsonify({
            "session": public_session,
            "stations": {sid: _station_public_dict(st) for sid, st in stations.items()},
            "opc": opc_server.public_status(),
        })


@app.route("/api/download_excel")
def download_excel():
    # During a running session, force one save so the download is current.
    with analysis_lock:
        if excel_writer is not None:
            excel_writer.save()
    if not os.path.exists(EXCEL_PATH):
        return jsonify({"error": "Excel file is not available yet"}), 404
    return send_file(EXCEL_PATH, as_attachment=True, download_name="assembly_4stations.xlsx")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_path": MODEL_PATH,
        "model_loaded": model is not None,
        "target_sample_fps": TARGET_SAMPLE_FPS,
        "aggregation_window_s": AGGREGATION_WINDOW_S,
        "samples_per_window": SAMPLES_PER_WINDOW,
        "capture_width": CAPTURE_WIDTH,
        "inference_imgsz": INFERENCE_IMGSZ,
        "cuda_available": bool(torch.cuda.is_available()),
        "yolo_device": str(YOLO_DEVICE),
        "video_prep_mode": VIDEO_PREP_MODE,
        "ffmpeg_available": ffmpeg_available(),
        "ffprobe_available": ffprobe_available(),
        "state_min_support": STATE_MIN_SUPPORT,
        "state_min_samples": STATE_MIN_SAMPLES,
        "cycle_confirmations": CYCLE_CONFIRMATIONS,
        "opc_ua": "on-demand snapshot",
        "opc_available": ASYNCUA_AVAILABLE,
        "opc_endpoint": OPC_ENDPOINT_PUBLIC,
        "opc_running": opc_server.is_running(),
    })


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    saved_rois = _load_roi_config()
    for sid in STATIONS:
        stations[sid].rois = dict(saved_rois.get(sid, {}))

    print(f"[INFO] BASE_DIR      : {BASE_DIR}")
    print(f"[INFO] MODEL_PATH    : {MODEL_PATH}")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
    print(f"[INFO] YOLO device   : {YOLO_DEVICE}")
    print(f"[INFO] YOLO imgsz    : {INFERENCE_IMGSZ}")
    print(f"[INFO] Target FPS    : {TARGET_SAMPLE_FPS}")
    print(f"[INFO] Window        : {AGGREGATION_WINDOW_S}s / target {SAMPLES_PER_WINDOW} samples")
    print(f"[INFO] Capture width : {CAPTURE_WIDTH}")
    print(f"[INFO] Video prep    : {VIDEO_PREP_MODE}")
    print(f"[INFO] FFmpeg        : {ffmpeg_available()} | FFprobe: {ffprobe_available()}")
    print(f"[INFO] Cycle detect  : {CYCLE_CONFIRMATIONS} consecutive raw detections; min duration={CYCLE_MIN_DURATION_S:.1f}s")
    print(f"[INFO] State quality : samples>={STATE_MIN_SAMPLES}, support>={STATE_MIN_SUPPORT:.2f}")
    print(f"[INFO] OPC UA        : ON-DEMAND ({OPC_ENDPOINT_PUBLIC})")
    if not ASYNCUA_AVAILABLE:
        print(f"[WARN] asyncua unavailable: {ASYNCUA_IMPORT_ERROR}")
    ensure_model_loaded()
    print("[INFO] Open http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
