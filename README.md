<div align="center">

# ⠃⠗⠇ BrailleGuard AI

**Every dot checked before it reaches a fingertip.**

Braille quality control for medicine packaging.

**AI-JAM US 2026 (Silicon Valley) · 9th of 840 entries from 41 countries · Silver Award, Healthcare category**

Built by a three-person team led by Nguyen Mau Dung.

`Streamlit` · `Python 3.10+` · `Pillow` · `NumPy` · MVP with a simulated inference engine

[![CI](https://github.com/dung037517-netizen/braille-ai-jam-2026/actions/workflows/ci.yml/badge.svg)](https://github.com/dung037517-netizen/braille-ai-jam-2026/actions/workflows/ci.yml)

</div>

---

## 1. The problem in one line

Braille on medicine boxes is checked by hand, by people who cannot read it. One flattened
dot turns **PARACETAMOL** into **PARACETAMEL** — and the person relying on the braille is
exactly the person who cannot check it against the print.

| Beat | Time | Line |
|---|---|---|
| **The Problem** | 0–8s | Braille on medicine boxes is checked by hand, by people who cannot read it. One flattened dot turns **PARACETAMOL** into **PARACETAMEL**. |
| **AI Solution** | 8–22s | BrailleGuard scans the label with a phone camera. Good label → **PASS**. Defective label → the AI boxes the exact cell, names the defect, holds the batch. |
| **Impact & Vision** | 22–30s | Under half a second. Any smartphone. 100% of the batch, not a sample. |

---

## 2. How it decides

Every inspection ends in one of three verdicts, and only one of them lets a batch through.

| Verdict | When | What happens |
|---|---|---|
| **PASS** | The photo is usable, the braille decodes to the approved text, and nothing is flagged. | Batch cleared. |
| **ERROR** | A defect is found, or the braille reads as a different word. | Batch held; the defective cells are boxed and named. |
| **HUMAN REVIEW** | The photo is too poor to judge, or a signal sits just under the decision threshold. | **Not passed.** The case goes to a person, with the reasons. |

The pipeline runs in this order:

1. **Check the photo before reading it.** Sharpness (variance of the Laplacian), brightness,
   contrast, glare or deep shadow, and size are measured first. A photo that fails any of
   them is never decoded and never passed — a blurry image is weak evidence, and the
   system does not clear a batch on weak evidence.
2. **Reconstruct the braille without the expected text.** The decoder reads the embossed
   dots into letters on its own; it is never shown what the label is supposed to say, so it
   cannot "see" what it expects.
3. **Compare with the approved text.** Only then is the decoded word compared with the
   reference. A missing dot that turns `O` into `E` is caught as a wrong word, not just a
   blemish.
4. **Do not drop near misses.** A candidate defect scoring within 0.15 below the decision
   threshold is too strong to ignore and too weak to call. It goes to a person instead of
   being discarded.

Image-quality limits (set on the generated demo labels — clean label sharpness ≈ 780,
after a 2 px blur ≈ 31 — and heuristics for this MVP, not calibrated limits):

| Check | Limit |
|---|---|
| Shortest side | ≥ 300 px |
| Sharpness (Laplacian variance at 640 px width) | ≥ 60 |
| Mean brightness | ≥ 50 of 255 |
| Contrast (grey-level standard deviation) | ≥ 14 |
| Pure-white or pure-black pixels | ≤ 25% |

---

## 3. Run it

```bash
git clone https://github.com/dung037517-netizen/braille-ai-jam-2026.git
cd braille-ai-jam-2026
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Opens on <http://localhost:8501>.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Eleven tests check the decisions that matter: a clean label passes and reads
`PARACETAMOL`; a missing dot reads `PARACETAMEL` and fails; the decoded text is the same
whatever reference it is compared with; a blurry, dark, washed-out, flat or tiny photo is
sent to review and never passed; and a score just under the threshold is reviewed at a
strict setting but called a defect at the default one. CI runs them on every push to
`main` and every pull request.

---

## 4. What the app does

| Tab | Contents |
|---|---|
| **① The Problem** | WHO vision-loss figures, why sampling-based QC misses embossing defects, the EU regulatory anchor, and the one-dot `O → E` punchline. |
| **② Live AI Demo** | Three scripted demos — a good label, a defective label, and a blurry photo of the good label — a drag-and-drop uploader, the inference pipeline, original and annotated images side by side, the verdict, the reasons for any review, a per-defect report, and a downloadable inspection image. |
| **③ Impact & Vision** | Speed / accuracy / cost / coverage metrics, the feasibility note, a four-stage roadmap, who it serves, and the 30-second script. |

The sidebar switches **Dark / Light** and **English / Tiếng Việt** live, and sets the
detection sensitivity (sensitivity 7 → confidence threshold 0.68).

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

## 5. Recording the demo video

1. Launch the app, open the **② Live AI Demo** tab, and set the browser to full screen.
2. Either click **GOOD LABEL**, **DEFECTIVE LABEL** and **BLURRY PHOTO** in turn, or click
   **AUTO-PITCH (30s)**, which runs the good and defective inspections on a 30-second timer
   with the script line and a countdown on screen.
3. Dark + English records best for international judges.

Everything on screen is sized for screen capture — headline type, the verdict badge and the
defect boxes stay readable after video compression.

---

## 6. Swapping in the real model

The whole AI layer is one class — `MockBrailleInspector` in `app.py`:

```python
class MockBrailleInspector:
    def predict_scripted(self, cells, defects, reference, threshold, image=None) -> Inspection: ...
    def predict_image(self, img, threshold) -> Inspection: ...
```

Keep the `Inspection` / `Finding` dataclasses and replace the bodies with a trained
detector — the UI, overlay renderer and report keep working untouched. The quality gate
(`assess_quality`) and the verdict rule (`decide`) sit outside the model on purpose: they
stay in front of whatever detector is swapped in.

- `predict_scripted` — the scripted demos, where the injected defects are known.
- `predict_image` — uploads: a real (but naive) block-contrast anomaly detector.

Both paths are deterministic: the same input always yields the same verdict.

> **Feasibility note.** MVP build: the algorithm is currently being optimized for lighting
> noise from real smartphone cameras. The accuracy figures shown in the app are engineering
> targets for the trained model, not measurements from this simulated build — stated plainly
> so the claim survives a judge's question.

---

## 7. Folder structure

```
braille-ai-jam-2026/
├── app.py                 # the application: UI, braille engine, quality gate, mock model
├── requirements.txt       # streamlit, pillow, numpy
├── requirements-dev.txt   # + pytest
├── tests/
│   └── test_inspector.py  # the decision rules, tested without a server
├── .streamlit/
│   └── config.toml        # dark theme, upload limit, clean toolbar for recording
├── .github/workflows/
│   └── ci.yml             # compile and test on every push and pull request
└── .gitignore
```

No assets folder — every demo image is generated from code at runtime, so the repository
stays light and the demo can never break on a missing file.

---

## 8. Deploy a public link

1. Go to <https://share.streamlit.io> → **New app** → pick this repository.
2. Set **Main file path** to `app.py` and deploy.

---

## 9. Sources

- WHO, *World Report on Vision* (2019) — 2.2 billion people with vision impairment.
- Vision Loss Expert Group / *Lancet Global Health* (2021) — ~43 million people blind, 2020.
- EU Directive 2004/27/EC amending 2001/83/EC, Art. 56a — braille required on medicinal packaging.
- EN 15823 / ISO 17351 — braille on packaging for medicinal products (Marburg Medium).

<div align="center">

A three-person team led by **Nguyen Mau Dung** · AI-JAM US 2026, Silicon Valley

</div>
