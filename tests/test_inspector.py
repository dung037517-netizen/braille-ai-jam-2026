"""Checks on the inspection logic, run without the Streamlit server.

    pip install -r requirements-dev.txt
    pytest
"""

from __future__ import annotations

import os
import sys

import pytest
from PIL import ImageEnhance, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app  # noqa: E402  (the app module is importable without a running server)

THRESHOLD = app.threshold_from_sensitivity(7)
REFERENCE = "PARACETAMOL"


def label(defects=None):
    return app.render_label(defects=defects or [])


def test_a_clean_label_passes_and_reads_the_approved_text():
    img, cells = label()
    assert app.assess_quality(img).ok
    insp = app.INSPECTOR.predict_scripted(cells, [], REFERENCE, THRESHOLD, image=img)
    assert insp.verdict == "PASS"
    assert insp.decoded == REFERENCE
    assert insp.review == []


def test_a_missing_dot_changes_the_word_and_fails_the_label():
    img, cells = label(app.BAD_DEFECTS)
    insp = app.INSPECTOR.predict_scripted(cells, app.BAD_DEFECTS, REFERENCE, THRESHOLD, image=img)
    assert insp.verdict == "ERROR"
    assert insp.decoded == "PARACETAMEL"
    assert {f.code for f in insp.findings} >= {"MISSING_DOT"}


def test_the_decoder_never_sees_the_expected_text():
    _, cells = label(app.BAD_DEFECTS)
    against_right = app.INSPECTOR.predict_scripted(cells, app.BAD_DEFECTS, "PARACETAMOL", THRESHOLD)
    against_wrong = app.INSPECTOR.predict_scripted(cells, app.BAD_DEFECTS, "PARACETAMEL", THRESHOLD)
    # The reading depends only on the dots; the reference is used afterwards.
    assert against_right.decoded == against_wrong.decoded == "PARACETAMEL"


def test_a_blurry_photo_is_never_passed_automatically():
    img, cells = label()
    poor = app.degrade_capture(img)
    quality = app.assess_quality(poor)
    assert not quality.ok
    assert quality.sharpness < app.QUALITY_LIMITS["sharpness"]
    insp = app.INSPECTOR.predict_scripted(cells, [], REFERENCE, THRESHOLD, image=poor)
    assert insp.verdict == "REVIEW"
    assert insp.decoded == ""          # nothing is read from an unusable image
    assert insp.review and all(en and vi for en, vi in insp.review)


@pytest.mark.parametrize(
    "name, spoil",
    [
        ("out of focus", lambda im: im.filter(ImageFilter.GaussianBlur(4))),
        ("too dark", lambda im: ImageEnhance.Brightness(im).enhance(0.15)),
        ("glare", lambda im: ImageEnhance.Brightness(im).enhance(2.5)),
        ("flat", lambda im: ImageEnhance.Contrast(im).enhance(0.2)),
        ("too small", lambda im: im.resize((240, int(im.height * 240 / im.width)))),
    ],
)
def test_poor_uploads_go_to_a_person(name, spoil):
    img, _ = label()
    insp = app.INSPECTOR.predict_image(spoil(img), THRESHOLD)
    assert insp.verdict == "REVIEW", name
    assert insp.review, name


def test_a_signal_just_under_the_threshold_is_not_cleared():
    _, cells = label()
    weak = [d for d in app.BAD_DEFECTS if d.kind == "LOW_DOT_HEIGHT"]
    # At a strict threshold the dot-height score falls just short: not a
    # defect, but not a pass either.
    strict = app.INSPECTOR.predict_scripted(cells, weak, REFERENCE, 0.95)
    assert strict.verdict == "REVIEW"
    assert strict.findings == []
    # At the default threshold the same score is a defect.
    default = app.INSPECTOR.predict_scripted(cells, weak, REFERENCE, THRESHOLD)
    assert default.verdict == "ERROR"


def test_verdict_rules():
    assert app.decide([], False, []) == "PASS"
    assert app.decide([], True, []) == "ERROR"
    assert app.decide([], False, [("why", "vì sao")]) == "REVIEW"
    finding = app.Finding("MISSING_DOT", "critical", "", (0, 0, 1, 1), 0.99, "", "")
    assert app.decide([finding], False, [("why", "vì sao")]) == "ERROR"
