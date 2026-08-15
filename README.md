<div align="center">

# ⠃⠗⠇ BrailleGuard AI

**Every dot checked before it reaches a fingertip.**

AI-powered defect inspection for braille on medicine packaging.
Submission — **AI-JAM US 2026 · The 11th International AI Invention Challenge**

`Streamlit` · `Python 3.10+` · `Pillow` · `NumPy` · MVP with a simulated inference engine

</div>

---

## 1. The 30-second pitch

| Beat | Time | Line |
|---|---|---|
| **The Problem** | 0–8s | Braille on medicine boxes is checked by hand, by people who cannot read it. One flattened dot turns **PARACETAMOL** into **PARACETAMEL**. |
| **AI Solution** | 8–22s | BrailleGuard scans the label with a phone camera. Good label → **PASS**. Defective label → the AI boxes the exact cell, names the defect, holds the batch. |
| **Impact & Vision** | 22–30s | Under half a second. Any smartphone. 100% of the batch, not a sample. |

The same script is built into the app: **Impact & Vision → 30-second pitch script**.

---

## 2. Run it

```bash
cd braille-guard
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Opens on <http://localhost:8501>.

---

## 3. Recording the demo video

1. Launch the app, open the **② Live AI Demo** tab, and set the browser to full screen.
2. Two options:
   - **Manual** — click `RUN DEMO — GOOD LABEL`, then `RUN DEMO — DEFECTIVE LABEL`.
   - **Hands-free** — click `🎬 AUTO-PITCH (30s)`. It runs both inspections on a 30-second
     timer with the script line and a countdown on screen, so you only have to hit record.
3. The sidebar switches **Dark / Light** and **English / Tiếng Việt** live. Dark + English
   records best for international judges.

Everything on screen is sized for screen capture — headline type, the PASS/ERROR badge, and
the defect boxes all stay readable after video compression.

---

## 4. What the app actually does

| Tab | Contents |
|---|---|
| **① The Problem** | WHO vision-loss figures, why sampling-based QC misses embossing defects, the EU regulatory anchor, and the one-dot `O → E` punchline. |
| **② Live AI Demo** | Two scripted buttons, a drag & drop uploader, the inference progress pipeline, side-by-side original vs. annotated output, PASS/ERROR badge, per-defect report, downloadable inspection image. |
| **③ Impact & Vision** | Speed / accuracy / cost / coverage metrics, the feasibility note, a four-stage roadmap, who it serves, and the 30-second script. |

### Defects the engine reports

| Code | Severity | Meaning |
|---|---|---|
| `MISSING_DOT` | critical | A dot was not embossed — the cell decodes as a different letter. |
| `EXTRA_DOT` | critical | A spurious dot — same consequence, opposite cause. |
| `LOW_DOT_HEIGHT` | major | Dot height under the ~0.12 mm readable threshold (embossing pressure loss). |
| `CELL_PITCH` | minor | Cell spacing outside the 6.0 mm Marburg Medium tolerance. |
| `SURFACE_ANOMALY` | major | Contrast outlier found by the placeholder detector on an uploaded photo. |

Reference geometry follows **EN 15823 / ISO 17351** (Marburg Medium): 1.5 mm dot base,
2.5 mm dot spacing, 6.0 mm cell pitch.

---

## 5. Swapping in the real model

The whole AI layer is one class — `MockBrailleInspector` in `app.py`:

```python
class MockBrailleInspector:
    def predict_scripted(self, cells, defects, reference, threshold) -> Inspection: ...
    def predict_image(self, img, threshold) -> Inspection: ...
```

Keep the `Inspection` / `Finding` dataclasses and replace the body with your trained
detector — the entire UI, overlay renderer and report keep working untouched. Both mock
paths are deterministic on purpose: the same input always yields the same boxes, so a demo
never surprises you on stage.

- `predict_scripted` — used by the two demo buttons, where the injected defects are known.
- `predict_image` — used for uploads: a real (but naive) block-contrast anomaly detector.

> **Feasibility note.** MVP build: the algorithm is currently being optimised for lighting
> noise from real smartphone cameras. The accuracy figures shown in the app are engineering
> targets for the trained model, not measurements from this simulated build — stated plainly
> so the claim survives a judge's question.

---

## 6. Folder structure

```
braille-guard/
├── app.py               # the entire application (UI + braille engine + mock model)
├── requirements.txt     # streamlit, pillow, numpy
├── README.md            # this file
├── .gitignore           # Python / Streamlit
└── .streamlit/
    └── config.toml      # dark base theme, upload limit, clean toolbar for recording
```

No assets folder — every demo image is generated from code at runtime, so the repo stays
light and the demo can never break on a missing file.

---

## 7. Push it to GitHub

**A. This folder as its own new repository** (create an empty repo on GitHub first,
then replace `YOUR-REPO-URL`):

```bash
git init && git add . && git commit -m "BrailleGuard AI - AI-JAM US 2026 MVP"
git branch -M main
git remote add origin YOUR-REPO-URL
git push -u origin main
```

**B. Keeping it inside this profile repository** — run from the repository root:

```bash
git add braille-guard
git commit -m "Add BrailleGuard AI - AI-JAM US 2026 MVP"
git push -u origin main
```

---

## 8. Deploy a public link (for the submission form)

1. Push to GitHub (above).
2. Go to <https://share.streamlit.io> → **New app** → pick the repo.
3. Set **Main file path** to `braille-guard/app.py` (or `app.py` if it is its own repo).
4. Deploy. The public URL is what you paste into the AI-JAM submission.

---

## 9. Sources

- WHO, *World Report on Vision* (2019) — 2.2 billion people with vision impairment.
- Vision Loss Expert Group / *Lancet Global Health* (2021) — ~43 million people blind, 2020.
- EU Directive 2004/27/EC amending 2001/83/EC, Art. 56a — braille required on medicinal packaging.
- EN 15823 / ISO 17351 — braille on packaging for medicinal products (Marburg Medium).

<div align="center">

Built by **Nguyen Mau Dung** · AI-JAM US 2026

</div>
