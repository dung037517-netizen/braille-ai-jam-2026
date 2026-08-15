"""
BrailleGuard AI — Braille label defect inspection (MVP)
=======================================================
AI-JAM US 2026 · The 11th International AI Invention Challenge

A single-file Streamlit app built for a 30-second demo video:
    Tab 1  The Problem
    Tab 2  Live AI Demo
    Tab 3  Impact & Vision

The inference engine in this build is a MOCK (see `MockBrailleInspector`).
It is deterministic, so the same input always produces the same output —
swap in a real model by replacing that one class.

Run:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import io
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ══════════════════════════════════════════════════════════════════════════════
# 0. APP CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

APP_NAME = "BrailleGuard AI"
APP_VERSION = "0.9.0-mvp"
ENGINE_NAME = "BG-Net v0.9 (mock)"
TAGLINE_EN = "Every dot checked before it reaches a fingertip."
TAGLINE_VI = "Kiểm tra từng chấm nổi trước khi chạm tới đầu ngón tay."

st.set_page_config(
    page_title=f"{APP_NAME} · AI-JAM US 2026",
    page_icon="⠿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit renamed the sizing argument in 1.49 (`use_container_width` ->
# `width="stretch"`). Detect once so the demo never shows a deprecation notice.
try:  # pragma: no cover - environment dependent
    from importlib.metadata import version as _pkg_version

    _ST_VER = tuple(int(p) for p in _pkg_version("streamlit").split(".")[:2])
except Exception:  # pragma: no cover
    _ST_VER = (1, 41)
_NEW_WIDTH_API = _ST_VER >= (1, 49)


def show_image(image: Any, caption: str | None = None) -> None:
    """st.image that stretches to the container on any recent Streamlit."""
    if _NEW_WIDTH_API:
        st.image(image, caption=caption, width="stretch")
    else:
        st.image(image, caption=caption, use_container_width=True)


def wide_button(label: str, **kwargs: Any) -> bool:
    """st.button that fills its container on any recent Streamlit."""
    if _NEW_WIDTH_API:
        return st.button(label, width="stretch", **kwargs)
    return st.button(label, use_container_width=True, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# 1. BILINGUAL COPY (EN for international judges / VI for local audience)
# ══════════════════════════════════════════════════════════════════════════════

STR: dict[str, dict[str, str]] = {
    # --- chrome ---------------------------------------------------------
    "tagline": {"en": TAGLINE_EN, "vi": TAGLINE_VI},
    "tab_problem": {"en": "① The Problem", "vi": "① Vấn Đề"},
    "tab_demo": {"en": "② Live AI Demo", "vi": "② Demo AI Trực Tiếp"},
    "tab_impact": {"en": "③ Impact & Vision", "vi": "③ Tác Động & Tầm Nhìn"},
    "sidebar_controls": {"en": "控 Controls", "vi": "Bảng điều khiển"},
    "theme": {"en": "Appearance", "vi": "Giao diện"},
    "language": {"en": "Language", "vi": "Ngôn ngữ"},
    "sensitivity": {"en": "Detection sensitivity", "vi": "Độ nhạy phát hiện"},
    "sensitivity_help": {
        "en": "Higher sensitivity surfaces lower-confidence defects. Factory QC lines tune this per product.",
        "vi": "Độ nhạy cao sẽ hiện cả lỗi có độ tin cậy thấp. Dây chuyền QC thực tế sẽ tinh chỉnh theo từng sản phẩm.",
    },
    "reset": {"en": "Reset demo", "vi": "Đặt lại demo"},
    "mock_chip": {"en": "MVP · simulated inference engine", "vi": "MVP · engine suy luận mô phỏng"},
    # --- problem tab ----------------------------------------------------
    "problem_kicker": {"en": "THE PROBLEM", "vi": "VẤN ĐỀ"},
    "problem_head": {
        "en": "A missing dot is a wrong medicine.",
        "vi": "Một chấm nổi bị thiếu là một viên thuốc sai.",
    },
    # NOTE: these strings are injected into HTML blocks, so emphasis must be
    # written as tags — markdown syntax would render literally.
    "problem_sub": {
        "en": (
            "Braille on medicine packaging is mandatory in the EU and expected in most "
            "regulated markets. But the embossing is verified <b>by hand</b>, <b>by sampling</b> "
            "— and almost never by someone who reads braille."
        ),
        "vi": (
            "Chữ nổi Braille trên bao bì thuốc là <b>bắt buộc</b> ở EU và được kỳ vọng tại hầu hết "
            "thị trường có quản lý. Nhưng khâu dập nổi lại được kiểm tra <b>thủ công</b>, theo "
            "<b>mẫu xác suất</b> — và gần như không bao giờ bởi người biết đọc Braille."
        ),
    },
    "stat1_v": {"en": "2.2B", "vi": "2,2 tỷ"},
    "stat1_l": {
        "en": "people live with a vision impairment (WHO, World Report on Vision)",
        "vi": "người đang sống với suy giảm thị lực (WHO, World Report on Vision)",
    },
    "stat2_v": {"en": "43M", "vi": "43 triệu"},
    "stat2_l": {
        "en": "people are blind worldwide (Vision Loss Expert Group, 2020)",
        "vi": "người mù trên toàn thế giới (Vision Loss Expert Group, 2020)",
    },
    "stat3_v": {"en": "1 dot", "vi": "1 chấm"},
    "stat3_l": {
        "en": "is the entire difference between PARACETAMOL and PARACETAMEL",
        "vi": "là toàn bộ khác biệt giữa PARACETAMOL và PARACETAMEL",
    },
    "why_head": {"en": "Why today's QC misses it", "vi": "Vì sao QC hiện nay bỏ sót"},
    "why1_t": {"en": "Sampling, not coverage", "vi": "Kiểm mẫu, không phải kiểm toàn bộ"},
    "why1_b": {
        "en": "Lines pull a handful of cartons per batch. A worn embossing die degrades gradually — the defect appears between two samples.",
        "vi": "Dây chuyền chỉ lấy vài hộp mỗi lô. Khuôn dập mòn dần — lỗi xuất hiện đúng vào khoảng giữa hai lần lấy mẫu.",
    },
    "why2_t": {"en": "Sighted inspectors, unreadable text", "vi": "Người kiểm tra sáng mắt, ký tự không đọc được"},
    "why2_b": {
        "en": "A QC operator can see that dots exist. They cannot see that the dots spell the wrong drug name.",
        "vi": "Nhân viên QC nhìn thấy các chấm nổi tồn tại. Nhưng không thấy được rằng các chấm đó ghép thành sai tên thuốc.",
    },
    "why3_t": {"en": "Height is invisible to the eye", "vi": "Chiều cao chấm nổi mắt thường không thấy"},
    "why3_b": {
        "en": "A dot flattened below the readable threshold still looks like a dot from above. Under a fingertip, it is not there at all.",
        "vi": "Chấm nổi bị bẹt dưới ngưỡng đọc được vẫn trông như một chấm khi nhìn từ trên. Nhưng dưới đầu ngón tay thì nó không tồn tại.",
    },
    "why4_t": {"en": "The reader cannot verify", "vi": "Người đọc không thể tự kiểm chứng"},
    "why4_b": {
        "en": "The end user is the last line of defence — and is precisely the person who cannot cross-check the print above the braille.",
        "vi": "Người dùng cuối là hàng phòng thủ cuối cùng — và cũng chính là người không thể đối chiếu với dòng chữ in phía trên.",
    },
    "reg_note": {
        "en": "Regulatory anchor — EU Directive 2004/27/EC (amending 2001/83/EC, Art. 56a) requires the product name in braille on medicinal packaging. EN 15823 / ISO 17351 specify the Marburg Medium geometry this MVP measures against.",
        "vi": "Cơ sở pháp lý — Chỉ thị EU 2004/27/EC (sửa đổi 2001/83/EC, Điều 56a) yêu cầu tên thuốc phải có chữ nổi Braille trên bao bì. EN 15823 / ISO 17351 quy định hình học Marburg Medium mà MVP này dùng làm chuẩn đối chiếu.",
    },
    # --- demo tab -------------------------------------------------------
    "demo_kicker": {"en": "LIVE AI DEMO", "vi": "DEMO AI TRỰC TIẾP"},
    "demo_head": {"en": "Point. Scan. Verdict in under a second.", "vi": "Chụp. Quét. Kết luận dưới một giây."},
    "btn_pass": {"en": "▶  RUN DEMO — GOOD LABEL", "vi": "▶  CHẠY DEMO — NHÃN CHUẨN"},
    "btn_fail": {"en": "▶  RUN DEMO — DEFECTIVE LABEL", "vi": "▶  CHẠY DEMO — NHÃN LỖI"},
    "btn_auto": {"en": "🎬  AUTO-PITCH (30s)", "vi": "🎬  PITCH TỰ ĐỘNG 30S"},
    "upload_head": {"en": "Or inspect a real photo", "vi": "Hoặc kiểm tra ảnh thật"},
    "upload_label": {
        "en": "Drag & drop a photo of a braille label (PNG / JPG)",
        "vi": "Kéo & thả ảnh nhãn Braille (PNG / JPG)",
    },
    "col_input": {"en": "INPUT · Captured label", "vi": "ĐẦU VÀO · Ảnh nhãn"},
    "col_output": {"en": "OUTPUT · AI inspection", "vi": "ĐẦU RA · Phân tích AI"},
    "verdict_pass": {"en": "PASS", "vi": "ĐẠT"},
    "verdict_fail": {"en": "ERROR", "vi": "LỖI"},
    "verdict_pass_sub": {
        "en": "All braille cells match the reference string. Batch cleared for release.",
        "vi": "Toàn bộ ô Braille khớp chuỗi tham chiếu. Lô hàng được thông qua.",
    },
    "verdict_fail_sub": {
        "en": "Defects detected. Batch held — do not release to packaging.",
        "vi": "Phát hiện lỗi. Giữ lô hàng — không chuyển sang đóng gói.",
    },
    "findings_head": {"en": "Defect report", "vi": "Báo cáo lỗi"},
    "no_findings": {
        "en": "No defects above the current sensitivity threshold.",
        "vi": "Không có lỗi nào vượt ngưỡng độ nhạy hiện tại.",
    },
    "m_latency": {"en": "Inference time", "vi": "Thời gian xử lý"},
    "m_cells": {"en": "Cells scanned", "vi": "Số ô đã quét"},
    "m_defects": {"en": "Defects found", "vi": "Số lỗi phát hiện"},
    "m_conf": {"en": "Top confidence", "vi": "Độ tin cậy cao nhất"},
    "expected": {"en": "Expected", "vi": "Mong đợi"},
    "observed": {"en": "Observed", "vi": "Thực tế"},
    "empty_state": {
        "en": "Press one of the two buttons above — or drop in your own photo — to run an inspection.",
        "vi": "Nhấn một trong hai nút phía trên — hoặc thả ảnh của bạn vào — để bắt đầu kiểm tra.",
    },
    "stage_load": {"en": "Loading frame…", "vi": "Đang nạp khung hình…"},
    "stage_norm": {"en": "Normalising illumination…", "vi": "Đang chuẩn hoá ánh sáng…"},
    "stage_grid": {"en": "Locating braille cell grid…", "vi": "Đang định vị lưới ô Braille…"},
    "stage_dots": {"en": "Measuring dot geometry…", "vi": "Đang đo hình học chấm nổi…"},
    "stage_dec": {"en": "Decoding cells → text…", "vi": "Đang giải mã ô → ký tự…"},
    "stage_cmp": {"en": "Comparing against reference…", "vi": "Đang đối chiếu chuỗi tham chiếu…"},
    "decoded": {"en": "Decoded braille", "vi": "Chuỗi Braille giải mã"},
    "reference": {"en": "Reference string", "vi": "Chuỗi tham chiếu"},
    "upload_note": {
        "en": "Uploaded photos run through the heuristic placeholder detector (contrast-anomaly grid). The scripted buttons show the full cell-level pipeline.",
        "vi": "Ảnh tải lên chạy qua bộ phát hiện placeholder (lưới bất thường tương phản). Hai nút kịch bản sẽ minh hoạ đầy đủ pipeline cấp ô.",
    },
    # --- impact tab -----------------------------------------------------
    "impact_kicker": {"en": "IMPACT & VISION", "vi": "TÁC ĐỘNG & TẦM NHÌN"},
    "impact_head": {
        "en": "One phone camera replaces a braille-literate inspector.",
        "vi": "Một camera điện thoại thay thế một chuyên viên kiểm tra biết đọc Braille.",
    },
    "k_speed": {"en": "Processing speed", "vi": "Tốc độ xử lý"},
    "k_speed_d": {"en": "per label, on-device", "vi": "mỗi nhãn, chạy ngay trên thiết bị"},
    "k_acc": {"en": "Target accuracy", "vi": "Độ chính xác mục tiêu"},
    "k_acc_d": {"en": "cell-level, projected", "vi": "cấp độ ô, dự kiến"},
    "k_cost": {"en": "Hardware cost", "vi": "Chi phí phần cứng"},
    "k_cost_d": {"en": "any smartphone — no scanner rig", "vi": "mọi smartphone — không cần máy quét chuyên dụng"},
    "k_scale": {"en": "Coverage", "vi": "Phạm vi kiểm tra"},
    "k_scale_d": {"en": "of the batch, not a sample", "vi": "của cả lô, thay vì lấy mẫu"},
    "road_head": {"en": "Roadmap", "vi": "Lộ trình"},
    "r1_t": {"en": "Now · MVP", "vi": "Hiện tại · MVP"},
    "r1_b": {
        "en": "Cell-grid detection + geometry rules on flat, well-lit labels. This demo.",
        "vi": "Phát hiện lưới ô + luật hình học trên nhãn phẳng, đủ sáng. Chính là demo này.",
    },
    "r2_t": {"en": "Next · Real capture", "vi": "Kế tiếp · Ảnh chụp thực tế"},
    "r2_b": {
        "en": "CNN trained on embossed-dot crops; glare, curvature and skew correction for handheld phone shots.",
        "vi": "CNN huấn luyện trên ảnh cắt chấm nổi; hiệu chỉnh loá sáng, độ cong và nghiêng cho ảnh chụp cầm tay.",
    },
    "r3_t": {"en": "Then · Line integration", "vi": "Sau đó · Tích hợp dây chuyền"},
    "r3_b": {
        "en": "Fixed camera over the packaging line, 100% of cartons checked, auto-reject signal to the PLC.",
        "vi": "Camera cố định trên dây chuyền đóng gói, kiểm 100% số hộp, tín hiệu loại bỏ tự động tới PLC.",
    },
    "r4_t": {"en": "Vision · Beyond pharma", "vi": "Tầm nhìn · Vượt khỏi ngành dược"},
    "r4_b": {
        "en": "Food allergen labels, elevator panels, public signage, exam papers — anywhere braille is produced without a braille reader in the loop.",
        "vi": "Nhãn dị ứng thực phẩm, bảng thang máy, biển báo công cộng, đề thi — mọi nơi Braille được tạo ra mà không có người đọc Braille kiểm tra.",
    },
    "feas_head": {"en": "Feasibility note", "vi": "Ghi chú tính khả thi"},
    "feas_body": {
        "en": "MVP build: the algorithm is currently being optimised for lighting noise from real smartphone cameras. Accuracy figures on this page are engineering targets for the trained model, not measurements from this simulated build.",
        "vi": "Bản MVP: Thuật toán đang được tối ưu hóa cho nhiễu ánh sáng từ camera smartphone thực tế. Các chỉ số độ chính xác trên trang này là mục tiêu kỹ thuật của mô hình khi được huấn luyện, không phải kết quả đo từ bản mô phỏng này.",
    },
    "who_head": {"en": "Who it serves", "vi": "Phục vụ ai"},
    "w1_t": {"en": "Packaging lines", "vi": "Dây chuyền đóng gói"},
    "w1_b": {
        "en": "A QC step that runs on a phone already on the floor — no new capital equipment.",
        "vi": "Một bước QC chạy ngay trên chiếc điện thoại đã có sẵn tại xưởng — không cần đầu tư thiết bị mới.",
    },
    "w2_t": {"en": "Pharmacies & clinics", "vi": "Nhà thuốc & phòng khám"},
    "w2_b": {
        "en": "Spot-check stock on arrival, before it reaches a patient who cannot verify it.",
        "vi": "Kiểm tra ngẫu nhiên hàng khi nhập, trước khi tới tay bệnh nhân không thể tự kiểm chứng.",
    },
    "w3_t": {"en": "Braille readers", "vi": "Người đọc Braille"},
    "w3_b": {
        "en": "The end user finally gets a way to confirm the box says what it claims.",
        "vi": "Người dùng cuối cùng cũng có cách xác nhận hộp thuốc đúng như những gì nó ghi.",
    },
    "w4_t": {"en": "Regulators & NGOs", "vi": "Cơ quan quản lý & tổ chức phi lợi nhuận"},
    "w4_b": {
        "en": "Auditable, timestamped evidence of compliance instead of a sampling log.",
        "vi": "Bằng chứng tuân thủ có dấu thời gian, kiểm toán được — thay cho sổ ghi lấy mẫu.",
    },
    "dl_report": {
        "en": "Download inspection report (PNG)",
        "vi": "Tải ảnh báo cáo kiểm tra (PNG)",
    },
    "sdg_note": {
        "en": "Aligned with UN SDG 3 (Good Health & Well-being), SDG 10 (Reduced Inequalities) and the WHO/UN CRPD accessibility mandate.",
        "vi": "Phù hợp với Mục tiêu Phát triển Bền vững của LHQ: SDG 3 (Sức khỏe), SDG 10 (Giảm bất bình đẳng) và cam kết tiếp cận theo Công ước CRPD.",
    },
    "script_head": {"en": "30-second pitch script", "vi": "Kịch bản pitch 30 giây"},
}


def t(key: str) -> str:
    lang = st.session_state.get("lang", "en")
    entry = STR.get(key, {})
    return entry.get(lang) or entry.get("en") or key


# 30-second teleprompter beats: (start_s, end_s, en, vi)
PITCH_BEATS: list[tuple[int, int, str, str]] = [
    (0, 4, "Braille on medicine boxes is checked by hand — by people who cannot read it.",
           "Chữ nổi trên hộp thuốc được kiểm tra thủ công — bởi những người không đọc được nó."),
    (4, 8, "One flattened dot turns PARACETAMOL into PARACETAMEL.",
           "Một chấm nổi bị bẹt biến PARACETAMOL thành PARACETAMEL."),
    (8, 14, "BrailleGuard scans the label with a phone camera. Good label: PASS.",
            "BrailleGuard quét nhãn bằng camera điện thoại. Nhãn chuẩn: ĐẠT."),
    (14, 22, "Defective label: the AI boxes the exact cell, names the defect, holds the batch.",
             "Nhãn lỗi: AI khoanh đúng ô sai, gọi tên lỗi, và giữ lô hàng lại."),
    (22, 27, "Under half a second. Any smartphone. One hundred percent of the batch.",
             "Dưới nửa giây. Mọi smartphone. Kiểm tra 100% lô hàng."),
    (27, 30, "BrailleGuard AI — every dot checked before it reaches a fingertip.",
             "BrailleGuard AI — kiểm tra từng chấm nổi trước khi chạm tới đầu ngón tay."),
]


# ══════════════════════════════════════════════════════════════════════════════
# 2. THEME + CSS
# ══════════════════════════════════════════════════════════════════════════════

PALETTES: dict[str, dict[str, str]] = {
    "dark": {
        "bg": "#060912",
        "bg2": "#0B1220",
        "glow1": "rgba(56,189,248,0.16)",
        "glow2": "rgba(167,139,250,0.14)",
        "surface": "rgba(255,255,255,0.045)",
        "surface2": "rgba(255,255,255,0.075)",
        "border": "rgba(255,255,255,0.11)",
        "text": "#E8EEF9",
        "muted": "#94A3B8",
        "accent": "#38BDF8",
        "accent2": "#A78BFA",
        "ok": "#22C55E",
        "okbg": "rgba(34,197,94,0.13)",
        "err": "#FB3B5C",
        "errbg": "rgba(251,59,92,0.13)",
        "warn": "#FBBF24",
        "shadow": "0 18px 48px rgba(0,0,0,0.55)",
    },
    "light": {
        "bg": "#F4F7FC",
        "bg2": "#FFFFFF",
        "glow1": "rgba(2,132,199,0.10)",
        "glow2": "rgba(124,58,237,0.09)",
        "surface": "#FFFFFF",
        "surface2": "#F1F5F9",
        "border": "rgba(15,23,42,0.10)",
        "text": "#0B1220",
        "muted": "#52627A",
        "accent": "#0284C7",
        "accent2": "#7C3AED",
        "ok": "#15803D",
        "okbg": "rgba(21,128,61,0.10)",
        "err": "#DC2626",
        "errbg": "rgba(220,38,38,0.09)",
        "warn": "#B45309",
        "shadow": "0 14px 38px rgba(15,23,42,0.10)",
    },
}


def inject_css() -> None:
    p = PALETTES[st.session_state.get("theme", "dark")]

    # Vietnamese typography guard. Monospace fallbacks (DejaVu Sans Mono and
    # friends) lack precomposed glyphs such as Ế/Ề, so the browser decomposes
    # them and the tone mark lands in its own cell — "VẤN ĐỀ" renders as
    # "VÂN ĐÊ`". Wide tracking makes it worse. So in Vietnamese the UI labels
    # drop to the sans stack with tight tracking; genuinely tabular ASCII
    # (timings, dot numbers, decoded strings) stays monospace either way.
    vi = st.session_state.get("lang", "en") == "vi"
    font_ui = (
        "'Inter',-apple-system,'Segoe UI',Roboto,sans-serif" if vi
        else "'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace"
    )
    ls_wide = "0.06em" if vi else "0.26em"
    ls_mid = "0.03em" if vi else "0.15em"
    ls_chip = "0.02em" if vi else "0.05em"

    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&family=JetBrains+Mono:wght@500;700&display=swap');

:root {{
  --bg:{p['bg']}; --bg2:{p['bg2']}; --surface:{p['surface']}; --surface2:{p['surface2']};
  --border:{p['border']}; --text:{p['text']}; --muted:{p['muted']};
  --accent:{p['accent']}; --accent2:{p['accent2']};
  --ok:{p['ok']}; --okbg:{p['okbg']}; --err:{p['err']}; --errbg:{p['errbg']}; --warn:{p['warn']};
  --shadow:{p['shadow']};
  --ls-wide:{ls_wide}; --ls-mid:{ls_mid}; --ls-chip:{ls_chip};
  --font-ui:{font_ui};
  --font-mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
}}

html, body, [class*="css"], .stApp {{
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
}}

.stApp {{
  background:
    radial-gradient(1100px 620px at 8% -8%, {p['glow1']}, transparent 60%),
    radial-gradient(1000px 560px at 96% 4%, {p['glow2']}, transparent 62%),
    linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%);
  background-attachment: fixed;
  color: var(--text);
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stToolbar"] {{ right: 1rem; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1500px; }}
html {{ scroll-behavior: smooth; }}

/* ---------- typography: sized for screen recording ---------- */
.stApp, .stMarkdown p, .stMarkdown li {{ font-size: 1.08rem; line-height: 1.65; color: var(--text); }}
.stMarkdown strong {{ color: var(--text); font-weight: 800; }}
h1,h2,h3,h4 {{ color: var(--text) !important; letter-spacing:-0.02em; }}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, var(--bg2), var(--bg));
  border-right: 1px solid var(--border);
}}
section[data-testid="stSidebar"] * {{ color: var(--text); }}

/* ---------- hero ---------- */
.bg-hero {{
  position:relative; overflow:hidden;
  border:1px solid var(--border); border-radius:26px;
  background: var(--surface);
  backdrop-filter: blur(14px);
  padding: 30px 34px; margin-bottom: 20px; box-shadow: var(--shadow);
}}
.bg-hero::after {{
  content:""; position:absolute; inset:-2px;
  background: conic-gradient(from 0deg, transparent 0 62%, {p['glow1']} 76%, transparent 88%);
  animation: sweep 9s linear infinite; pointer-events:none;
}}
@keyframes sweep {{ to {{ transform: rotate(360deg); }} }}
.bg-hero-inner {{ position:relative; z-index:2; display:flex; flex-wrap:wrap;
  align-items:center; justify-content:space-between; gap:18px; }}
.bg-logo {{ display:flex; align-items:center; gap:16px; }}
.bg-dots {{
  font-family:var(--font-mono); font-size:2.6rem; line-height:1;
  color:var(--accent); text-shadow:0 0 26px {p['glow1']};
  animation: pulseglow 2.8s ease-in-out infinite;
}}
@keyframes pulseglow {{ 0%,100%{{opacity:.75}} 50%{{opacity:1}} }}
.bg-title {{ font-size:2.5rem; font-weight:900; letter-spacing:-0.035em; margin:0;
  background:linear-gradient(100deg,var(--text) 8%,var(--accent) 52%,var(--accent2) 96%);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }}
.bg-tag {{ color:var(--muted); font-size:1.02rem; margin-top:4px; font-weight:600; }}
.bg-badges {{ display:flex; flex-wrap:wrap; gap:8px; justify-content:flex-end; }}
.chip {{
  display:inline-flex; align-items:center; gap:7px;
  padding:7px 14px; border-radius:999px; border:1px solid var(--border);
  background:var(--surface2); color:var(--muted);
  font-size:.82rem; font-weight:700; letter-spacing:var(--ls-chip); text-transform:uppercase;
  white-space:nowrap;
}}
.chip.accent {{ color:var(--accent); border-color:var(--accent); }}
.chip.warn {{ color:var(--warn); border-color:var(--warn); }}

/* ---------- section headers ---------- */
.kicker {{
  font-family:var(--font-ui); font-size:.86rem; font-weight:700;
  letter-spacing:var(--ls-wide); color:var(--accent); text-transform:uppercase; margin-bottom:6px;
}}
.sec-head {{ font-size:2.6rem; font-weight:900; letter-spacing:-0.035em;
  line-height:1.12; margin:0 0 12px 0; color:var(--text); }}
.sec-sub {{ font-size:1.2rem; color:var(--muted); max-width:900px; line-height:1.6; }}

/* ---------- cards ---------- */
.card {{
  border:1px solid var(--border); border-radius:20px; background:var(--surface);
  padding:22px 24px; height:100%; backdrop-filter: blur(10px);
  transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}}
.card:hover {{ transform:translateY(-4px); border-color:var(--accent); box-shadow:var(--shadow); }}
.card-t {{ font-size:1.14rem; font-weight:800; margin:10px 0 8px; color:var(--text); }}
.card-b {{ font-size:1rem; color:var(--muted); line-height:1.6; }}
.card-ico {{ font-size:1.7rem; }}

/* ---------- big stats ---------- */
.stat {{
  border:1px solid var(--border); border-radius:20px; background:var(--surface);
  padding:26px 22px; text-align:center; height:100%;
}}
.stat-v {{
  font-size:3.4rem; font-weight:900; letter-spacing:-0.045em; line-height:1;
  background:linear-gradient(120deg,var(--accent),var(--accent2));
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}}
.stat-l {{ font-size:.98rem; color:var(--muted); margin-top:12px; line-height:1.5; }}

/* ---------- KPI ---------- */
.kpi {{
  border:1px solid var(--border); border-radius:20px; background:var(--surface);
  padding:24px 20px; height:100%; position:relative; overflow:hidden;
}}
.kpi::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:4px;
  background:linear-gradient(180deg,var(--accent),var(--accent2)); }}
.kpi-l {{ font-size:.8rem; font-weight:800; letter-spacing:var(--ls-mid); text-transform:uppercase; color:var(--muted); }}
.kpi-v {{ font-size:2.9rem; font-weight:900; letter-spacing:-0.04em; color:var(--text); margin:8px 0 2px; line-height:1; }}
.kpi-d {{ font-size:.95rem; color:var(--muted); }}

/* ---------- verdict badge ---------- */
.verdict {{
  display:flex; align-items:center; gap:20px; padding:24px 28px;
  border-radius:22px; border:2px solid; margin:6px 0 18px;
}}
.verdict-pass {{ background:var(--okbg); border-color:var(--ok);
  animation: popin .45s cubic-bezier(.2,1.5,.4,1) both; }}
.verdict-fail {{ background:var(--errbg); border-color:var(--err);
  animation: popin .45s cubic-bezier(.2,1.5,.4,1) both, shake .55s .35s ease-in-out; }}
@keyframes popin {{ from{{transform:scale(.9);opacity:0}} to{{transform:scale(1);opacity:1}} }}
@keyframes shake {{ 0%,100%{{transform:translateX(0)}} 20%{{transform:translateX(-9px)}}
  40%{{transform:translateX(9px)}} 60%{{transform:translateX(-5px)}} 80%{{transform:translateX(5px)}} }}
.verdict-icon {{ font-size:3rem; line-height:1; }}
.verdict-word {{ font-size:2.9rem; font-weight:900; letter-spacing:-0.04em; line-height:1; }}
.verdict-pass .verdict-word {{ color:var(--ok); }}
.verdict-fail .verdict-word {{ color:var(--err); }}
.verdict-sub {{ font-size:1.05rem; color:var(--muted); margin-top:6px; }}
.dotpulse {{ width:14px;height:14px;border-radius:50%;display:inline-block;
  margin-right:12px;vertical-align:middle; }}
.dotpulse.ok {{ background:var(--ok); box-shadow:0 0 0 0 var(--ok); animation:ring 1.8s infinite; }}
.dotpulse.err {{ background:var(--err); box-shadow:0 0 0 0 var(--err); animation:ring 1.1s infinite; }}
@keyframes ring {{ to {{ box-shadow:0 0 0 18px transparent; }} }}

/* ---------- findings ---------- */
.finding {{
  border:1px solid var(--border); border-left:5px solid var(--err);
  border-radius:14px; background:var(--surface); padding:16px 18px; margin-bottom:12px;
  animation: slidein .4s ease both;
}}
@keyframes slidein {{ from{{opacity:0;transform:translateX(-14px)}} to{{opacity:1;transform:none}} }}
.finding.major {{ border-left-color:var(--warn); }}
.finding.minor {{ border-left-color:var(--accent); }}
.f-top {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
.f-code {{ font-family:var(--font-mono); font-weight:700; font-size:.9rem;
  color:var(--text); background:var(--surface2); padding:3px 9px; border-radius:7px; }}
.f-sev {{ font-size:.72rem; font-weight:800; letter-spacing:var(--ls-chip); text-transform:uppercase;
  padding:3px 10px; border-radius:999px; }}
.sev-critical {{ background:var(--errbg); color:var(--err); border:1px solid var(--err); }}
.sev-major {{ background:rgba(251,191,36,.14); color:var(--warn); border:1px solid var(--warn); }}
.sev-minor {{ background:var(--surface2); color:var(--muted); border:1px solid var(--border); }}
.f-conf {{ margin-left:auto; font-family:var(--font-mono);
  font-weight:700; color:var(--accent); font-size:.95rem; }}
.f-msg {{ font-size:1.04rem; font-weight:700; margin:9px 0 4px; color:var(--text); }}
.f-meta {{ font-size:.92rem; color:var(--muted); font-family:var(--font-ui); }}

/* ---------- panels / notes ---------- */
.panel {{ border:1px solid var(--border); border-radius:20px; background:var(--surface);
  padding:20px 24px; }}
.note {{ border:1px dashed var(--warn); border-radius:16px;
  background:rgba(251,191,36,.08); padding:18px 22px; }}
.note-t {{ font-weight:800; color:var(--warn); font-size:1rem; letter-spacing:var(--ls-chip);
  text-transform:uppercase; margin-bottom:6px; }}
.note-b {{ color:var(--muted); font-size:1.02rem; line-height:1.6; }}
.mono {{ font-family:var(--font-mono); }}
.strike {{ text-decoration:line-through; opacity:.65; }}

/* ---------- timeline ---------- */
.tl {{ position:relative; padding-left:34px; }}
.tl::before {{ content:""; position:absolute; left:11px; top:6px; bottom:6px; width:2px;
  background:linear-gradient(180deg,var(--accent),var(--accent2),transparent); }}
.tl-item {{ position:relative; margin-bottom:22px; }}
.tl-item::before {{ content:""; position:absolute; left:-28px; top:5px; width:13px; height:13px;
  border-radius:50%; background:var(--bg2); border:3px solid var(--accent); }}
.tl-t {{ font-weight:800; font-size:1.1rem; color:var(--text); }}
.tl-b {{ color:var(--muted); font-size:1rem; margin-top:3px; line-height:1.55; }}

/* ---------- tabs -------------------------------------------------------
   Streamlit has shipped tabs as BaseWeb <button> and as react-aria <div>;
   [role="tab"] matches both, so the styling survives either DOM.          */
.stTabs [role="tablist"] {{
  gap:10px; border-bottom:1px solid var(--border);
}}
.stTabs [role="tab"] {{
  height:auto; padding:13px 28px; border-radius:14px 14px 0 0;
  background:transparent; cursor:pointer; transition:background .18s ease;
}}
.stTabs [role="tab"] p {{
  font-size:1.16rem !important; font-weight:800 !important;
  letter-spacing:-0.01em; color:var(--muted); margin:0;
}}
.stTabs [role="tab"]:hover {{ background:var(--surface2); }}
.stTabs [role="tab"][aria-selected="true"] {{ background:var(--surface); }}
.stTabs [role="tab"][aria-selected="true"] p {{ color:var(--accent) !important; }}
.stTabs [data-baseweb="tab-highlight"],
.stTabs .react-aria-SelectionIndicator {{ background:var(--accent) !important; height:3px; }}

/* ---------- buttons ---------- */
.stButton > button {{
  border-radius:16px; font-weight:800; font-size:1.12rem; padding:1.15rem 1rem;
  min-height:66px;                     /* keeps the demo buttons identical */
  border:1px solid var(--border); background:var(--surface2); color:var(--text);
  transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
}}
.stButton > button p {{ font-size:1.12rem !important; font-weight:800 !important; }}
.stButton > button:hover {{ transform:translateY(-3px); box-shadow:var(--shadow); border-color:var(--accent); }}
.stButton > button:active {{ transform:translateY(0); }}
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {{
  background:linear-gradient(100deg,var(--accent),var(--accent2));
  border:1px solid transparent; color:#03070F; letter-spacing:.02em;
}}
.stButton > button[kind="primary"] p {{ color:#03070F; }}
.stButton > button[kind="primary"]:hover {{ filter:brightness(1.09); }}

/* ---------- radios as segmented pills -----------------------------------
   Streamlit's own radio marker is painted from the config.toml base theme,
   so it clashes when the in-app palette flips. Hiding the marker and
   styling the label itself keeps both themes consistent. If a future DOM
   change stops these selectors matching, the widget simply falls back to
   its stock appearance — nothing breaks.                                  */
[data-testid="stRadioGroup"] {{ gap:8px; }}
[data-testid="stRadioOption"] {{
  border:1px solid var(--border); border-radius:12px; padding:7px 13px;
  background:var(--surface2); cursor:pointer;
  transition:border-color .18s ease, background .18s ease;
}}
[data-testid="stRadioOption"]:hover {{ border-color:var(--accent); }}
[data-testid="stRadioOption"][data-selected="true"],
[data-testid="stRadioOption"]:has(input:checked) {{
  border-color:var(--accent); background:var(--surface);
  box-shadow:inset 0 0 0 1px var(--accent);
}}
[data-testid="stRadioOption"] > div > div > div:first-child {{ display:none !important; }}
[data-testid="stRadioOption"] p {{
  font-size:.98rem !important; font-weight:700 !important; color:var(--muted); margin:0;
}}
[data-testid="stRadioOption"][data-selected="true"] p,
[data-testid="stRadioOption"]:has(input:checked) p {{ color:var(--text); }}

/* ---------- misc widgets ---------- */
[data-testid="stFileUploader"] section {{
  border:2px dashed var(--border); border-radius:20px; background:var(--surface);
  padding:22px; transition:border-color .2s ease;
}}
[data-testid="stFileUploader"] section:hover {{ border-color:var(--accent); }}
[data-testid="stImage"] img {{ border-radius:18px; border:1px solid var(--border); box-shadow:var(--shadow); }}
.stProgress > div > div > div > div {{
  background-image:linear-gradient(90deg,var(--accent),var(--accent2));
}}
hr {{ border-color:var(--border); }}
[data-testid="stMetricValue"] {{ color:var(--text); }}
.col-cap {{ font-family:var(--font-ui); font-size:.85rem; font-weight:700;
  letter-spacing:var(--ls-mid); text-transform:uppercase; color:var(--muted); margin-bottom:8px; }}
.footer {{ text-align:center; color:var(--muted); font-size:.92rem; padding:30px 0 8px; }}
</style>
""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. BRAILLE ENGINE — reference geometry + label renderer
# ══════════════════════════════════════════════════════════════════════════════
#
#   Braille cell dot numbering        Marburg Medium reference (EN 15823)
#        1 ● ● 4                      dot base diameter   1.5 mm
#        2 ● ● 5                      dot spacing in cell 2.5 mm
#        3 ● ● 6                      cell pitch          6.0 mm
#                                     dot height       >= 0.12 mm

BRAILLE: dict[str, tuple[int, ...]] = {
    "a": (1,), "b": (1, 2), "c": (1, 4), "d": (1, 4, 5), "e": (1, 5),
    "f": (1, 2, 4), "g": (1, 2, 4, 5), "h": (1, 2, 5), "i": (2, 4), "j": (2, 4, 5),
    "k": (1, 3), "l": (1, 2, 3), "m": (1, 3, 4), "n": (1, 3, 4, 5), "o": (1, 3, 5),
    "p": (1, 2, 3, 4), "q": (1, 2, 3, 4, 5), "r": (1, 2, 3, 5), "s": (2, 3, 4),
    "t": (2, 3, 4, 5), "u": (1, 3, 6), "v": (1, 2, 3, 6), "w": (2, 4, 5, 6),
    "x": (1, 3, 4, 6), "y": (1, 3, 4, 5, 6), "z": (1, 3, 5, 6), " ": (),
}
REVERSE_BRAILLE: dict[tuple[int, ...], str] = {v: k for k, v in BRAILLE.items()}

PX_PER_MM = 13
DOT_D = round(1.5 * PX_PER_MM)        # 20 px
DOT_R = DOT_D // 2
DOT_PITCH = round(2.5 * PX_PER_MM)    # 33 px
CELL_PITCH = round(6.0 * PX_PER_MM)   # 78 px
MARGIN_X = 62
BRAILLE_TOP = 306
# Extra bottom headroom so the AI's HUD bar never covers the printed footer.
LABEL_H = 624
FOOTER_Y = LABEL_H - 152

PAPER = (243, 241, 233)
INK = (24, 27, 34)
INK_SOFT = (118, 122, 132)

# Defect kinds the mock engine can inject / detect ---------------------------
DEFECT_META = {
    "MISSING_DOT": ("critical", "⛔"),
    "EXTRA_DOT": ("critical", "⛔"),
    "LOW_DOT_HEIGHT": ("major", "⚠️"),
    "CELL_PITCH": ("minor", "📐"),
    "SURFACE_ANOMALY": ("major", "⚠️"),
}


def _font_path(bold: bool = True) -> str | None:
    cands = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ]
        if bold
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/Library/Fonts/Arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
    )
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def load_font(size: int, bold: bool = True) -> ImageFont.ImageFont:
    path = _font_path(bold)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    try:  # Pillow >= 10.1 supports sizing the bitmap fallback
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - very old Pillow
        return ImageFont.load_default()


@dataclass
class Defect:
    """A defect injected into a rendered label (ground truth for the demo)."""

    cell: int
    kind: str
    dot: int | None = None
    detail_en: str = ""
    detail_vi: str = ""
    expected: str = ""
    observed: str = ""


@dataclass
class Cell:
    index: int
    char: str
    dots: tuple[int, ...]
    box: tuple[int, int, int, int]


@dataclass
class Finding:
    code: str
    severity: str
    icon: str
    box: tuple[int, int, int, int]
    confidence: float
    message_en: str
    message_vi: str
    expected: str = ""
    observed: str = ""

    def message(self, lang: str) -> str:
        return self.message_vi if lang == "vi" else self.message_en


def _draw_dot(d: ImageDraw.ImageDraw, cx: int, cy: int, height: float = 1.0) -> None:
    """Draw one embossed braille dot. `height` < 1 simulates a flattened dot."""
    r = DOT_R * (0.62 + 0.38 * height)
    contrast = 0.35 + 0.65 * height

    def mix(c: tuple[int, int, int]) -> tuple[int, int, int]:
        return tuple(int(PAPER[i] + (c[i] - PAPER[i]) * contrast) for i in range(3))  # type: ignore[return-value]

    d.ellipse([cx - r + 2, cy - r + 3.5, cx + r + 2, cy + r + 3.5], fill=mix((176, 172, 160)))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=mix((231, 228, 218)),
              outline=mix((156, 151, 138)), width=2)
    d.ellipse([cx - r * 0.62, cy - r * 0.68, cx + r * 0.12, cy + r * 0.06],
              fill=mix((253, 252, 249)))


def _dot_center(cell_x: int, dot: int) -> tuple[int, int]:
    col = 0 if dot in (1, 2, 3) else 1
    row = (dot - 1) % 3
    return (cell_x + col * DOT_PITCH + DOT_R, BRAILLE_TOP + row * DOT_PITCH + DOT_R)


def render_label(
    product: str = "PARACETAMOL",
    strength: str = "500 mg  ·  20 film-coated tablets",
    lot: str = "LOT A-2026-118",
    defects: list[Defect] | None = None,
) -> tuple[Image.Image, list[Cell]]:
    """Render a synthetic pharmaceutical carton face with a braille line."""
    defects = defects or []
    text = product.lower()
    n = len(text)
    width = MARGIN_X * 2 + n * CELL_PITCH
    img = Image.new("RGB", (width, LABEL_H), PAPER)

    # paper grain -----------------------------------------------------------
    rng = np.random.default_rng(7)
    grain = rng.normal(0, 3.1, (LABEL_H, width, 1)).repeat(3, axis=2)
    img = Image.fromarray(np.clip(np.asarray(img, dtype=np.float32) + grain, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)

    # carton framing --------------------------------------------------------
    d.rounded_rectangle([16, 16, width - 17, LABEL_H - 17], radius=22,
                        outline=(206, 202, 190), width=3)
    d.rounded_rectangle([28, 28, width - 29, LABEL_H - 29], radius=16,
                        outline=(224, 220, 209), width=1)

    f_pill = load_font(23, bold=True)
    f_name = load_font(74, bold=True)
    f_sub = load_font(30, bold=False)
    f_small = load_font(21, bold=False)
    f_tiny = load_font(18, bold=False)

    # Rx pill
    pill = "Rx  ONLY"
    pw = int(d.textlength(pill, font=f_pill))
    d.rounded_rectangle([MARGIN_X, 52, MARGIN_X + pw + 34, 96], radius=22,
                        fill=(29, 78, 138))
    d.text((MARGIN_X + 17, 62), pill, font=f_pill, fill=(255, 255, 255))

    # brand block on the right
    d.text((width - MARGIN_X - 210, 60), "VINPHARMA", font=f_small, fill=INK_SOFT)
    d.text((width - MARGIN_X - 210, 86), "EU/1/26/0118", font=f_tiny, fill=INK_SOFT)

    # product name + strength
    d.text((MARGIN_X, 118), product.upper(), font=f_name, fill=INK)
    d.text((MARGIN_X + 4, 214), strength, font=f_sub, fill=INK_SOFT)

    # divider
    d.line([MARGIN_X, 272, width - MARGIN_X, 272], fill=(212, 208, 196), width=2)

    # braille line ----------------------------------------------------------
    by_cell: dict[int, list[Defect]] = {}
    for df in defects:
        by_cell.setdefault(df.cell, []).append(df)

    cells: list[Cell] = []
    for i, ch in enumerate(text):
        cell_defects = by_cell.get(i, [])
        shift = 0
        for df in cell_defects:
            if df.kind == "CELL_PITCH":
                shift = 7
        cell_x = MARGIN_X + i * CELL_PITCH + shift

        dots = set(BRAILLE.get(ch, ()))
        low: set[int] = set()
        for df in cell_defects:
            if df.kind == "MISSING_DOT" and df.dot:
                dots.discard(df.dot)
            elif df.kind == "EXTRA_DOT" and df.dot:
                dots.add(df.dot)
            elif df.kind == "LOW_DOT_HEIGHT" and df.dot:
                low.add(df.dot)

        for dot in sorted(dots):
            cx, cy = _dot_center(cell_x, dot)
            _draw_dot(d, cx, cy, height=0.24 if dot in low else 1.0)

        pad = 13
        box = (
            MARGIN_X + i * CELL_PITCH - pad,
            BRAILLE_TOP - pad,
            MARGIN_X + i * CELL_PITCH + DOT_PITCH + DOT_D + pad,
            BRAILLE_TOP + 2 * DOT_PITCH + DOT_D + pad,
        )
        cells.append(Cell(index=i, char=ch, dots=tuple(sorted(dots)), box=box))

    # regulatory footer -----------------------------------------------------
    d.line([MARGIN_X, FOOTER_Y, width - MARGIN_X, FOOTER_Y], fill=(212, 208, 196), width=2)
    d.text((MARGIN_X, FOOTER_Y + 18),
           f"EN 15823 · Marburg Medium · {lot} · EXP 2028-04",
           font=f_tiny, fill=INK_SOFT)
    d.text((width - MARGIN_X - 128, FOOTER_Y + 18), "MADE IN VN", font=f_tiny, fill=INK_SOFT)

    img = img.filter(ImageFilter.GaussianBlur(0.35))
    return img, cells


# --- the two scripted demo labels -------------------------------------------

GOOD_DEFECTS: list[Defect] = []

BAD_DEFECTS: list[Defect] = [
    Defect(
        cell=9, kind="MISSING_DOT", dot=3,
        detail_en="Dot 3 not embossed — cell decodes as 'E' instead of 'O'.",
        detail_vi="Chấm số 3 không được dập — ô này giải mã thành 'E' thay vì 'O'.",
        expected="O  (dots 1·3·5)", observed="E  (dots 1·5)",
    ),
    Defect(
        cell=4, kind="LOW_DOT_HEIGHT", dot=4,
        detail_en="Dot 4 height below the 0.12 mm readable threshold — embossing pressure loss.",
        detail_vi="Chấm số 4 thấp hơn ngưỡng đọc được 0,12 mm — mất lực dập nổi.",
        expected="height ≥ 0.12 mm", observed="height ≈ 0.05 mm",
    ),
    Defect(
        cell=6, kind="CELL_PITCH",
        detail_en="Cell drifted right — the gap to cell 8 is under tolerance, so a fingertip reads cells 7 and 8 as one.",
        detail_vi="Ô bị lệch sang phải — khoảng cách tới ô 8 dưới dung sai, đầu ngón tay sẽ đọc ô 7 và 8 thành một.",
        expected="6.00 mm ± 0.10", observed="5.46 mm (cell 7→8)",
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# 4. INFERENCE — MOCK ENGINE
# ══════════════════════════════════════════════════════════════════════════════
#
#  ┌──────────────────────────────────────────────────────────────────────┐
#  │  TODO(real-model): replace MockBrailleInspector with the trained     │
#  │  detector. Keep the `predict()` signature and the Finding dataclass  │
#  │  and the entire UI below keeps working unchanged.                    │
#  └──────────────────────────────────────────────────────────────────────┘


@dataclass
class Inspection:
    verdict: str                     # "PASS" | "ERROR"
    findings: list[Finding]
    cells_scanned: int
    latency_ms: float
    reference: str = ""
    decoded: str = ""
    source: str = "demo"             # "demo" | "upload"
    meta: dict[str, Any] = field(default_factory=dict)


class MockBrailleInspector:
    """Deterministic stand-in for the trained BG-Net model.

    Two code paths:
      * `predict_scripted` — ground truth is known (the two demo buttons), so
        the engine reports exactly the injected defects with plausible
        confidences. This is what gets recorded in the pitch video.
      * `predict_image` — for arbitrary uploads: a genuine (but naive)
        contrast-anomaly grid detector. No randomness: the same photo always
        yields the same boxes.
    """

    version = ENGINE_NAME

    # ------------------------------------------------------------------ #
    def predict_scripted(
        self, cells: list[Cell], defects: list[Defect], reference: str, threshold: float
    ) -> Inspection:
        t0 = time.perf_counter()
        rng = random.Random(len(defects) * 97 + len(cells))
        findings: list[Finding] = []
        base_conf = {"critical": 0.975, "major": 0.928, "minor": 0.861}

        for df in defects:
            sev, icon = DEFECT_META[df.kind]
            conf = min(0.998, base_conf[sev] + rng.uniform(-0.018, 0.018))
            if conf < threshold:
                continue
            findings.append(
                Finding(
                    code=df.kind,
                    severity=sev,
                    icon=icon,
                    box=cells[df.cell].box,
                    confidence=conf,
                    message_en=f"Cell {df.cell + 1}: {df.detail_en}",
                    message_vi=f"Ô {df.cell + 1}: {df.detail_vi}",
                    expected=df.expected,
                    observed=df.observed,
                )
            )

        # Decode straight from the dots that actually got embossed: a missing
        # dot changes the character, a flat dot or a pitch error does not.
        decoded = "".join(REVERSE_BRAILLE.get(c.dots, "?") for c in cells).upper()
        latency = (time.perf_counter() - t0) * 1000 + rng.uniform(180, 320)
        mismatch = decoded != reference.upper()
        return Inspection(
            verdict="ERROR" if (findings or mismatch) else "PASS",
            findings=findings,
            cells_scanned=len(cells),
            latency_ms=latency,
            reference=reference.upper(),
            decoded=decoded,
            source="demo",
        )

    # ------------------------------------------------------------------ #
    def predict_image(self, img: Image.Image, threshold: float) -> Inspection:
        """Naive placeholder detector for user-supplied photos.

        Splits the frame into blocks, scores each block's local contrast
        against the robust median, and flags statistical outliers. It is not
        a braille model — it is an honest placeholder that behaves
        deterministically for the same input.
        """
        t0 = time.perf_counter()
        W, H = img.size
        work_w = 640
        work_h = max(1, int(H * work_w / W))
        g = np.asarray(img.convert("L").resize((work_w, work_h)), dtype=np.float32)

        bs = 32
        rows, cols = work_h // bs, work_w // bs
        if rows < 2 or cols < 2:
            return Inspection("PASS", [], 0, (time.perf_counter() - t0) * 1000,
                              source="upload")

        blocks = g[: rows * bs, : cols * bs].reshape(rows, bs, cols, bs)
        std = blocks.std(axis=(1, 3))
        med = float(np.median(std))
        mad = float(np.median(np.abs(std - med))) * 1.4826 + 1e-6
        z = (std - med) / mad

        flagged = np.argwhere(z > 3.2)
        # group adjacent flagged blocks into regions (4-neighbour flood fill)
        flags = {(int(r), int(c)) for r, c in flagged}
        regions: list[list[tuple[int, int]]] = []
        seen: set[tuple[int, int]] = set()
        for cellpos in sorted(flags):
            if cellpos in seen:
                continue
            stack, group = [cellpos], []
            seen.add(cellpos)
            while stack:
                r, c = stack.pop()
                group.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = (r + dr, c + dc)
                    if nb in flags and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            regions.append(group)

        sx, sy = W / work_w, H / work_h
        scored = []
        for grp in regions:
            zs = float(max(z[r, c] for r, c in grp))
            r0 = min(r for r, _ in grp); r1 = max(r for r, _ in grp)
            c0 = min(c for _, c in grp); c1 = max(c for _, c in grp)
            box = (
                int(c0 * bs * sx) - 8, int(r0 * bs * sy) - 8,
                int((c1 + 1) * bs * sx) + 8, int((r1 + 1) * bs * sy) + 8,
            )
            conf = float(min(0.97, 0.55 + 0.11 * (zs - 3.2)))
            scored.append((conf, box, len(grp)))
        scored.sort(reverse=True, key=lambda x: x[0])

        findings: list[Finding] = []
        for conf, box, area in scored[:4]:
            if conf < threshold:
                continue
            sev, icon = DEFECT_META["SURFACE_ANOMALY"]
            findings.append(
                Finding(
                    code="SURFACE_ANOMALY",
                    severity=sev if area > 1 else "minor",
                    icon=icon,
                    box=box,
                    confidence=conf,
                    message_en="Local contrast anomaly in the embossed region — candidate dot-height or spacing defect.",
                    message_vi="Bất thường tương phản cục bộ ở vùng dập nổi — nghi ngờ lỗi chiều cao chấm hoặc khoảng cách.",
                    expected="uniform dot relief",
                    observed=f"outlier block ({area} cell{'s' if area > 1 else ''})",
                )
            )

        latency = (time.perf_counter() - t0) * 1000
        return Inspection(
            verdict="ERROR" if findings else "PASS",
            findings=findings,
            cells_scanned=rows * cols,
            latency_ms=latency,
            source="upload",
        )


INSPECTOR = MockBrailleInspector()


# ══════════════════════════════════════════════════════════════════════════════
# 5. ANNOTATOR — draw the AI overlay
# ══════════════════════════════════════════════════════════════════════════════

RED = (244, 45, 82)
AMBER = (251, 176, 36)
GREEN = (32, 197, 108)
CYAN = (34, 189, 248)

# Short forms keep the on-image chips readable at video resolution; the full
# code still appears in the defect report under the images.
SHORT_CODE = {
    "MISSING_DOT": "MISSING DOT",
    "EXTRA_DOT": "EXTRA DOT",
    "LOW_DOT_HEIGHT": "DOT HEIGHT",
    "CELL_PITCH": "CELL PITCH",
    "SURFACE_ANOMALY": "ANOMALY",
}


def _brackets(d: ImageDraw.ImageDraw, box, color, width=5, arm=22) -> None:
    x0, y0, x1, y1 = box
    for (ax, ay, bx, by) in (
        (x0, y0, x0 + arm, y0), (x0, y0, x0, y0 + arm),
        (x1 - arm, y0, x1, y0), (x1, y0, x1, y0 + arm),
        (x0, y1, x0 + arm, y1), (x0, y1 - arm, x0, y1),
        (x1 - arm, y1, x1, y1), (x1, y1 - arm, x1, y1),
    ):
        d.line([ax, ay, bx, by], fill=color, width=width)


def annotate(
    base: Image.Image,
    insp: Inspection,
    cells: list[Cell] | None,
) -> Image.Image:
    img = base.convert("RGB").copy()
    W, H = img.size
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")
    scale = max(0.75, min(1.6, W / 1000))
    f_chip = load_font(int(21 * scale), bold=True)
    f_hud = load_font(int(19 * scale), bold=True)

    flagged_boxes = [f.box for f in insp.findings]

    # 1. verified cells (scripted path only) --------------------------------
    if cells:
        for c in cells:
            if c.box in flagged_boxes:
                continue
            d.rounded_rectangle(list(c.box), radius=9, outline=GREEN + (168,), width=2)

    # 2. defect boxes + non-overlapping callout chips -----------------------
    chip_h = int(34 * scale)
    gap = int(7 * scale)
    rows_used: list[list[tuple[float, float]]] = []   # occupied x-spans per row

    for f in sorted(insp.findings, key=lambda x: x.box[0]):
        color = RED if f.severity in ("critical", "major") else AMBER
        x0, y0, x1, y1 = f.box
        d.rectangle([x0 - 3, y0 - 3, x1 + 3, y1 + 3], fill=color + (34,))
        d.rounded_rectangle([x0, y0, x1, y1], radius=10, outline=color + (255,), width=5)
        _brackets(d, (x0 - 7, y0 - 7, x1 + 7, y1 + 7), color + (255,),
                  width=max(3, int(4 * scale)), arm=int(20 * scale))

        chip = f"{SHORT_CODE.get(f.code, f.code)}  {f.confidence * 100:.0f}%"
        tw = d.textlength(chip, font=f_chip) + 22
        cx0 = max(6.0, min(float(W) - tw - 6, x0 - 2))
        span = (cx0 - gap, cx0 + tw + gap)

        # stack chips upward until this one clears every chip already placed
        row = 0
        while row < len(rows_used) and any(
            span[0] < b and a < span[1] for a, b in rows_used[row]
        ):
            row += 1
        if row == len(rows_used):
            rows_used.append([])
        rows_used[row].append(span)

        cy0 = y0 - (chip_h + 12) * (row + 1)
        if cy0 < 4:                                   # no headroom -> go below
            cy0 = y1 + 12 + (chip_h + 8) * row
        cy0 = max(4, min(cy0, H - chip_h - 4))

        d.rounded_rectangle([cx0, cy0, cx0 + tw, cy0 + chip_h],
                            radius=int(9 * scale), fill=color + (242,))
        d.text((cx0 + 11, cy0 + int(6 * scale)), chip, font=f_chip, fill=(255, 255, 255))

    # 3. viewfinder brackets (under the HUD bar, hence drawn before it) -----
    ok = insp.verdict == "PASS"
    hud_col = GREEN if ok else RED
    _brackets(d, (14, 14, W - 15, H - 15), CYAN + (140,), width=3, arm=int(30 * scale))

    # 4. HUD ----------------------------------------------------------------
    mismatch = bool(insp.decoded and insp.reference and insp.decoded != insp.reference)
    pad = int(22 * scale)
    line_h = int(30 * scale)
    bar_h = line_h * (2 if mismatch else 1) + int(20 * scale)
    top = H - bar_h
    d.rectangle([0, top, W, H], fill=(9, 13, 22, 255))
    d.rectangle([0, top, 8, H], fill=hud_col + (255,))

    ty = top + int(10 * scale)
    left = f"{ENGINE_NAME}   ·   {insp.cells_scanned} cells   ·   {insp.latency_ms:.0f} ms"
    d.text((pad, ty), left, font=f_hud, fill=(226, 234, 246))
    right = "✓ ALL CELLS VERIFIED" if ok else f"✕ {len(insp.findings)} DEFECT(S)"
    rw = d.textlength(right, font=f_hud)
    d.text((W - rw - pad, ty), right, font=f_hud, fill=hud_col + (255,))

    # The punchline of the whole demo, burned into the frame itself.
    if mismatch:
        ty2 = ty + line_h
        d.text((pad, ty2), "BRAILLE READS", font=f_hud, fill=(148, 163, 184))
        ox = pad + d.textlength("BRAILLE READS", font=f_hud) + int(14 * scale)
        d.text((ox, ty2), insp.decoded, font=f_hud, fill=RED + (255,))
        ox += d.textlength(insp.decoded, font=f_hud) + int(12 * scale)
        d.text((ox, ty2), "≠", font=f_hud, fill=(148, 163, 184))
        ox += d.textlength("≠", font=f_hud) + int(12 * scale)
        d.text((ox, ty2), insp.reference, font=f_hud, fill=(226, 234, 246))

    # 5. verdict frame ------------------------------------------------------
    d.rounded_rectangle([2, 2, W - 3, H - 3], radius=8, outline=hud_col + (190,), width=5)

    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


# ══════════════════════════════════════════════════════════════════════════════
# 6. HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def from_png(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def threshold_from_sensitivity(s: int) -> float:
    """Sensitivity 1 (conservative) → 0.95 ; 10 (aggressive) → 0.55."""
    return round(0.95 - (s - 1) * (0.40 / 9), 3)


@st.cache_data(show_spinner=False)
def build_demo(kind: str) -> dict[str, Any]:
    """Render one of the two scripted labels. Cached — rendering is pure."""
    defects = GOOD_DEFECTS if kind == "good" else BAD_DEFECTS
    img, cells = render_label(defects=defects)
    return {
        "png": to_png(img),
        "cells": [asdict(c) for c in cells],
        "defects": [asdict(x) for x in defects],
        "reference": "PARACETAMOL",
    }


def _cells_from(raw: list[dict]) -> list[Cell]:
    return [Cell(index=c["index"], char=c["char"], dots=tuple(c["dots"]),
                 box=tuple(c["box"])) for c in raw]  # type: ignore[arg-type]


def _defects_from(raw: list[dict]) -> list[Defect]:
    return [Defect(**x) for x in raw]


def run_scripted(kind: str) -> None:
    data = build_demo(kind)
    cells = _cells_from(data["cells"])
    defects = _defects_from(data["defects"])
    threshold = threshold_from_sensitivity(st.session_state.sensitivity)
    insp = INSPECTOR.predict_scripted(cells, defects, data["reference"], threshold)
    base = from_png(data["png"])
    annotated = annotate(base, insp, cells)
    st.session_state.result = {
        "original": data["png"],
        "annotated": to_png(annotated),
        "insp": insp,
        "title": "PARACETAMOL 500 mg · carton face",
    }
    st.session_state.celebrate = insp.verdict == "PASS"


def run_upload(file) -> None:
    img = Image.open(file).convert("RGB")
    if img.width > 1400:
        img = img.resize((1400, int(img.height * 1400 / img.width)), Image.LANCZOS)
    threshold = threshold_from_sensitivity(st.session_state.sensitivity)
    insp = INSPECTOR.predict_image(img, threshold)
    annotated = annotate(img, insp, None)
    st.session_state.result = {
        "original": to_png(img),
        "annotated": to_png(annotated),
        "insp": insp,
        "title": getattr(file, "name", "uploaded image"),
    }
    st.session_state.celebrate = insp.verdict == "PASS"


def pipeline_animation(slot) -> None:
    """The visible 'AI is thinking' sequence — the money shot of the video."""
    stages = ["stage_load", "stage_norm", "stage_grid", "stage_dots", "stage_dec", "stage_cmp"]
    with slot.container():
        bar = st.progress(0)
        cap = st.empty()
        for i, key in enumerate(stages):
            cap.markdown(
                f"<div style='font-family:var(--font-ui);color:var(--accent);"
                f"font-weight:700;font-size:1.05rem'>▸ {t(key)}</div>",
                unsafe_allow_html=True,
            )
            bar.progress(int((i + 1) / len(stages) * 100))
            time.sleep(0.16)
        cap.empty()
    slot.empty()


# ══════════════════════════════════════════════════════════════════════════════
# 7. UI FRAGMENTS
# ══════════════════════════════════════════════════════════════════════════════

def hero() -> None:
    st.markdown(
        f"""
<div class="bg-hero"><div class="bg-hero-inner">
  <div class="bg-logo">
    <div class="bg-dots">⠃⠗⠇</div>
    <div>
      <div class="bg-title">{APP_NAME}</div>
      <div class="bg-tag">{t('tagline')}</div>
    </div>
  </div>
  <div class="bg-badges">
    <span class="chip accent">AI-JAM US 2026</span>
    <span class="chip">11th Int'l AI Invention Challenge</span>
    <span class="chip warn">{t('mock_chip')}</span>
  </div>
</div></div>
""",
        unsafe_allow_html=True,
    )


def section_head(kicker: str, head: str, sub: str = "") -> None:
    st.markdown(
        f"<div class='kicker'>{kicker}</div><div class='sec-head'>{head}</div>"
        + (f"<div class='sec-sub'>{sub}</div>" if sub else ""),
        unsafe_allow_html=True,
    )


def stat(v: str, l: str) -> None:
    st.markdown(f"<div class='stat'><div class='stat-v'>{v}</div><div class='stat-l'>{l}</div></div>",
                unsafe_allow_html=True)


def card(icon: str, title: str, body: str) -> None:
    st.markdown(
        f"<div class='card'><div class='card-ico'>{icon}</div>"
        f"<div class='card-t'>{title}</div><div class='card-b'>{body}</div></div>",
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, desc: str) -> None:
    st.markdown(
        f"<div class='kpi'><div class='kpi-l'>{label}</div>"
        f"<div class='kpi-v'>{value}</div><div class='kpi-d'>{desc}</div></div>",
        unsafe_allow_html=True,
    )


def verdict_badge(insp: Inspection) -> None:
    ok = insp.verdict == "PASS"
    st.markdown(
        f"""
<div class="verdict {'verdict-pass' if ok else 'verdict-fail'}">
  <div class="verdict-icon">{'✅' if ok else '🛑'}</div>
  <div>
    <div class="verdict-word">
      <span class="dotpulse {'ok' if ok else 'err'}"></span>
      {t('verdict_pass') if ok else t('verdict_fail')}
    </div>
    <div class="verdict-sub">{t('verdict_pass_sub') if ok else t('verdict_fail_sub')}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def finding_row(f: Finding) -> None:
    lang = st.session_state.lang
    st.markdown(
        f"""
<div class="finding {f.severity}">
  <div class="f-top">
    <span>{f.icon}</span>
    <span class="f-code">{f.code}</span>
    <span class="f-sev sev-{f.severity}">{f.severity}</span>
    <span class="f-conf">{f.confidence * 100:.1f}%</span>
  </div>
  <div class="f-msg">{f.message(lang)}</div>
  <div class="f-meta">{t('expected')}: {f.expected or '—'}   │   {t('observed')}: {f.observed or '—'}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_result(slot_id: str = "main") -> None:
    """Render the last inspection.

    `slot_id` keeps widget IDs unique: auto-pitch mode renders the result more
    than once inside a single script run, which would otherwise collide on the
    download button's auto-generated key.
    """
    res = st.session_state.get("result")
    if not res:
        st.markdown(
            f"<div class='panel' style='text-align:center;padding:56px 24px'>"
            f"<div style='font-size:3.4rem;line-height:1'>⠿</div>"
            f"<div class='sec-sub' style='margin:14px auto 0'>{t('empty_state')}</div></div>",
            unsafe_allow_html=True,
        )
        return

    insp: Inspection = res["insp"]
    verdict_badge(insp)

    if st.session_state.pop("celebrate", False):
        st.balloons()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f"<div class='col-cap'>{t('col_input')}</div>", unsafe_allow_html=True)
        show_image(from_png(res["original"]), caption=res.get("title"))
    with c2:
        st.markdown(f"<div class='col-cap'>{t('col_output')}</div>", unsafe_allow_html=True)
        show_image(from_png(res["annotated"]), caption=f"{ENGINE_NAME} · overlay")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    top_conf = max((f.confidence for f in insp.findings), default=0.0)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        kpi(t("m_latency"), f"{insp.latency_ms:.0f} ms", "on-device")
    with m2:
        kpi(t("m_cells"), f"{insp.cells_scanned}", "100% coverage")
    with m3:
        kpi(t("m_defects"), f"{len(insp.findings)}", insp.verdict)
    with m4:
        kpi(t("m_conf"), f"{top_conf * 100:.1f}%" if top_conf else "—", "detector score")

    if insp.source == "demo" and insp.reference:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        shown = insp.decoded or insp.reference
        colour = "var(--err)" if shown != insp.reference else "var(--ok)"
        st.markdown(
            f"""
<div class="panel">
  <div class="col-cap">{t('reference')} → {t('decoded')}</div>
  <div class="mono" style="font-size:1.5rem;font-weight:700;letter-spacing:.06em">
    <span style="color:var(--muted)">{insp.reference}</span>
    <span style="color:var(--muted);margin:0 14px">→</span>
    <span style="color:{colour}">{shown}</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    st.markdown(f"### {t('findings_head')}")
    if insp.findings:
        for f in sorted(insp.findings,
                        key=lambda x: {"critical": 0, "major": 1, "minor": 2}[x.severity]):
            finding_row(f)
    else:
        st.success(t("no_findings"), icon="✅")

    if insp.source == "upload":
        st.caption(t("upload_note"))

    st.download_button(
        "⬇︎  " + t("dl_report"),
        data=res["annotated"],
        file_name=f"brailleguard_{insp.verdict.lower()}.png",
        mime="image/png",
        key=f"dl_{slot_id}",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 8. AUTO-PITCH (hands-free 30 s run for recording)
# ══════════════════════════════════════════════════════════════════════════════

def auto_pitch() -> None:
    lang = st.session_state.lang
    # Order matters: the teleprompter and progress bar are created first so
    # they sit above the inspection result on screen while recording.
    timer = st.empty()
    bar = st.progress(0)
    slot = st.empty()
    t_start = time.perf_counter()
    fired_good = fired_bad = False

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= 30:
            break
        beat = next((b for b in PITCH_BEATS if b[0] <= elapsed < b[1]), PITCH_BEATS[-1])
        line = beat[3] if lang == "vi" else beat[2]
        timer.markdown(
            f"""
<div class="panel" style="border-color:var(--accent)">
  <div class="col-cap">🎬 AUTO-PITCH · {30 - int(elapsed):02d}s</div>
  <div style="font-size:1.5rem;font-weight:800;line-height:1.4">{line}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        bar.progress(min(100, int(elapsed / 30 * 100)))

        if elapsed >= 8 and not fired_good:
            fired_good = True
            run_scripted("good")
            with slot.container():
                render_result("auto_good")
        if elapsed >= 15 and not fired_bad:
            fired_bad = True
            run_scripted("bad")
            with slot.container():
                render_result("auto_bad")
        time.sleep(0.12)

    bar.progress(100)
    timer.markdown(
        f"<div class='panel' style='border-color:var(--ok)'>"
        f"<div class='col-cap' style='color:var(--ok)'>✓ 30s COMPLETE</div>"
        f"<div style='font-size:1.4rem;font-weight:800'>{PITCH_BEATS[-1][3 if lang == 'vi' else 2]}</div></div>",
        unsafe_allow_html=True,
    )
    slot.empty()


# ══════════════════════════════════════════════════════════════════════════════
# 9. TABS
# ══════════════════════════════════════════════════════════════════════════════

def tab_problem() -> None:
    section_head(t("problem_kicker"), t("problem_head"))
    st.markdown(f"<div class='sec-sub'>{t('problem_sub')}</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)

    a, b, c = st.columns(3, gap="large")
    with a:
        stat(t("stat1_v"), t("stat1_l"))
    with b:
        stat(t("stat2_v"), t("stat2_l"))
    with c:
        stat(t("stat3_v"), t("stat3_l"))

    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

    # the visual punchline
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown(
            """
<div class="panel">
  <div class="col-cap">THE SINGLE DOT</div>
  <div class="mono" style="font-size:1.9rem;font-weight:700;line-height:1.5">
    <span style="color:var(--ok)">⠕</span>&nbsp; O &nbsp;=&nbsp; dots 1·3·5<br/>
    <span style="color:var(--err)">⠑</span>&nbsp; E &nbsp;=&nbsp; dots 1·<span class="strike">3</span>·5
  </div>
  <div style="margin-top:18px;font-size:1.35rem;font-weight:800">
    PARACETAM<span style="color:var(--ok)">O</span>L
    <span style="color:var(--muted);margin:0 12px">→</span>
    PARACETAM<span style="color:var(--err)">E</span>L
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(f"#### {t('why_head')}")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            card("🎲", t("why1_t"), t("why1_b"))
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            card("📏", t("why3_t"), t("why3_b"))
        with c2:
            card("👁️", t("why2_t"), t("why2_b"))
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            card("🙍", t("why4_t"), t("why4_b"))

    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='note'><div class='note-t'>⚖︎ Regulation</div>"
        f"<div class='note-b'>{t('reg_note')}</div></div>",
        unsafe_allow_html=True,
    )


def tab_demo() -> None:
    section_head(t("demo_kicker"), t("demo_head"))
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns([1, 1, 0.8], gap="large")
    clicked_good = clicked_bad = clicked_auto = False
    with b1:
        clicked_good = wide_button(t("btn_pass"), type="primary", key="btn_good")
    with b2:
        clicked_bad = wide_button(t("btn_fail"), key="btn_bad")
    with b3:
        clicked_auto = wide_button(t("btn_auto"), key="btn_auto")

    with st.expander("📎  " + t("upload_head"), expanded=False):
        up = st.file_uploader(t("upload_label"), type=["png", "jpg", "jpeg", "webp", "bmp"],
                              key="uploader", label_visibility="visible")
        if up is not None:
            sig = hashlib.md5(up.getvalue()).hexdigest()
            if st.session_state.get("last_upload") != sig:
                st.session_state.last_upload = sig
                slot = st.empty()
                pipeline_animation(slot)
                run_upload(up)
                st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if clicked_auto:
        auto_pitch()
        st.rerun()

    if clicked_good or clicked_bad:
        slot = st.empty()
        pipeline_animation(slot)
        run_scripted("good" if clicked_good else "bad")
        if clicked_bad:
            st.toast("🛑 Defective label — batch held", icon="🛑")
        else:
            st.toast("✅ Label verified — batch cleared", icon="✅")

    render_result()


def tab_impact() -> None:
    section_head(t("impact_kicker"), t("impact_head"))
    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4, gap="large")
    with k1:
        kpi(t("k_speed"), "< 0.5 s", t("k_speed_d"))
    with k2:
        kpi(t("k_acc"), "≥ 98%", t("k_acc_d"))
    with k3:
        kpi(t("k_cost"), "$0", t("k_cost_d"))
    with k4:
        kpi(t("k_scale"), "100%", t("k_scale_d"))

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='note'><div class='note-t'>🔬 {t('feas_head')}</div>"
        f"<div class='note-b'>{t('feas_body')}</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown(f"#### {t('road_head')}")
        st.markdown(
            f"""
<div class="tl">
  <div class="tl-item"><div class="tl-t">{t('r1_t')}</div><div class="tl-b">{t('r1_b')}</div></div>
  <div class="tl-item"><div class="tl-t">{t('r2_t')}</div><div class="tl-b">{t('r2_b')}</div></div>
  <div class="tl-item"><div class="tl-t">{t('r3_t')}</div><div class="tl-b">{t('r3_b')}</div></div>
  <div class="tl-item"><div class="tl-t">{t('r4_t')}</div><div class="tl-b">{t('r4_b')}</div></div>
</div>
""",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(f"#### {t('who_head')}")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            card("🏭", t("w1_t"), t("w1_b"))
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            card("🏥", t("w2_t"), t("w2_b"))
        with c2:
            card("🧑‍🦯", t("w3_t"), t("w3_b"))
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            card("🏛️", t("w4_t"), t("w4_b"))

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='panel'><div class='card-b'>🌍 {t('sdg_note')}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
    with st.expander("🎬  " + t("script_head"), expanded=False):
        lang = st.session_state.lang
        rows = "".join(
            f"<tr><td class='mono' style='padding:8px 18px 8px 0;color:var(--accent);"
            f"font-weight:700;white-space:nowrap'>{s:02d}–{e:02d}s</td>"
            f"<td style='padding:8px 0'>{vi if lang == 'vi' else en}</td></tr>"
            for s, e, en, vi in PITCH_BEATS
        )
        st.markdown(
            f"<table style='width:100%;border-collapse:collapse;font-size:1.05rem'>{rows}</table>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 10. SIDEBAR + MAIN
# ══════════════════════════════════════════════════════════════════════════════

def sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f"<div class='bg-dots' style='font-size:2rem'>⠃⠗⠇</div>"
            f"<div style='font-weight:900;font-size:1.35rem;margin-top:2px'>{APP_NAME}</div>"
            f"<div style='color:var(--muted);font-size:.9rem'>v{APP_VERSION}</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        st.session_state.lang = st.radio(
            t("language"), options=["en", "vi"],
            format_func=lambda x: "🇺🇸  English" if x == "en" else "🇻🇳  Tiếng Việt",
            horizontal=True, key="lang_radio",
            index=0 if st.session_state.get("lang", "en") == "en" else 1,
        )
        st.session_state.theme = st.radio(
            t("theme"), options=["dark", "light"],
            format_func=lambda x: "🌙  Dark" if x == "dark" else "☀️  Light",
            horizontal=True, key="theme_radio",
            index=0 if st.session_state.get("theme", "dark") == "dark" else 1,
        )
        st.divider()

        st.session_state.sensitivity = st.slider(
            t("sensitivity"), 1, 10, st.session_state.get("sensitivity", 7),
            help=t("sensitivity_help"),
        )
        st.caption(
            f"→ confidence threshold "
            f"**{threshold_from_sensitivity(st.session_state.sensitivity):.2f}**"
        )
        st.divider()

        if wide_button("↺  " + t("reset"), key="reset_btn"):
            st.session_state.result = None
            st.session_state.last_upload = None
            st.rerun()

        st.divider()
        st.markdown(
            f"<div style='color:var(--muted);font-size:.86rem;line-height:1.6'>"
            f"<b>Engine</b><br/>{ENGINE_NAME}<br/><br/>"
            f"<b>Reference standard</b><br/>EN 15823 · ISO 17351<br/>Marburg Medium<br/><br/>"
            f"<b>Submission</b><br/>AI-JAM US 2026<br/>11th Int'l AI Invention Challenge"
            f"</div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    st.session_state.setdefault("lang", "en")
    st.session_state.setdefault("theme", "dark")
    st.session_state.setdefault("sensitivity", 7)
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("last_upload", None)

    # A widget's key already holds the user's latest choice at the start of the
    # run, while the widget call itself only returns it partway through. Syncing
    # here means the CSS and every t() lookup — including the sidebar's own
    # labels — use the current choice instead of lagging one interaction behind.
    for widget_key, state_key in (("lang_radio", "lang"), ("theme_radio", "theme")):
        if widget_key in st.session_state:
            st.session_state[state_key] = st.session_state[widget_key]

    inject_css()
    sidebar()
    hero()

    tabs = st.tabs([t("tab_problem"), t("tab_demo"), t("tab_impact")])
    with tabs[0]:
        tab_problem()
    with tabs[1]:
        tab_demo()
    with tabs[2]:
        tab_impact()

    st.markdown(
        f"<div class='footer'>{APP_NAME} v{APP_VERSION} · {t('mock_chip')} · "
        f"built for AI-JAM US 2026</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
