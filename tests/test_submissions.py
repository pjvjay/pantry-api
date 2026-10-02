"""Origin submissions — the write path, at the service level (no HTTP).

Every assertion names the right answer read independently from the DB: a
submission's row, the evidence row approval copies, the resolved summary,
and — the test the whole path exists for — which product the planner
ships before and after a reading is approved.
"""
from __future__ import annotations

import getpass
import json
import os

import pytest
from sqlalchemy.orm import Session

_TMP_DB = None
GARLIC = 13                      # seeds/products.json: "Fresh Garlic"
US = ["United States"]


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    global _TMP_DB
    _TMP_DB = tmp_path_factory.mktemp("db") / "test.db"
    os.environ["DB_URL"] = os.environ.get("PANTRY_TEST_DB_URL") or f"sqlite:///{_TMP_DB}"
    os.environ["DEMO_MODE"] = "1"          # the plan test runs the pipeline keylessly
    os.environ.pop("ANTHROPIC_API_KEY", None)
    from pantry_planner import config, db

    config.settings.cache_clear()
    db.seed_from_json()
    yield
    os.environ.pop("DEMO_MODE", None)
    config.settings.cache_clear()


@pytest.fixture(autouse=True)
def clean_tables():
    from pantry_planner.db import (
        OriginSubmissionRow,
        ProductOriginEvidenceRow,
        ProductOriginRow,
        engine,
    )

    with Session(engine()) as s:
        s.query(OriginSubmissionRow).delete()
        s.query(ProductOriginEvidenceRow).delete()
        s.query(ProductOriginRow).delete()
        s.commit()
    yield


def _submit(submitted_by="agent-a", **kw):
    from pantry_planner.ingest import submit_origin

    rec = dict(product_id=GARLIC, claim_type="product-of", country="usa",
               verbatim="Product of USA", confidence="high")
    rec.update(kw)
    return submit_origin(rec, submitted_by=submitted_by)


def _rows(model, **filters):
    from pantry_planner.db import engine

    with Session(engine()) as s:
        rows = s.query(model).filter_by(**filters).order_by(model.id).all()
        s.expunge_all()
        return rows


def _submissions(**filters):
    from pantry_planner.db import OriginSubmissionRow

    return _rows(OriginSubmissionRow, **filters)


def _evidence(**filters):
    from pantry_planner.db import ProductOriginEvidenceRow

    return _rows(ProductOriginEvidenceRow, **filters)


# ─── submit ──────────────────────────────────────────────────

def test_submit_queues_a_pending_row_with_the_country_canonicalised():
    out = _submit()
    rows = _submissions()
    assert len(rows) == 1
    row = rows[0]
    assert out["id"] == row.id
    assert out["duplicate"] is False
    assert row.status == "pending"
    assert row.country == "United States"            # "usa" canonicalised
    assert row.ingredient_origin == "United States"  # full claim: both fields
    assert row.manufactured_in == "United States"
    assert row.verbatim == "Product of USA"
    assert row.submitted_by == "agent-a"
    assert row.submitted_at.startswith("20")
    assert row.evidence_id is None
    assert out["product_name"] == "Fresh Garlic"
    assert _evidence() == []                         # pending is not evidence


def test_processing_claim_fills_only_manufactured_in():
    _submit(claim_type="made-in", country="Canada", verbatim="Made in Canada")
    row = _submissions()[0]
    assert row.manufactured_in == "Canada"
    assert row.ingredient_origin == ""


def test_duplicate_pending_returns_the_same_row_and_inserts_nothing():
    first = _submit()
    again = _submit(submitted_by="agent-b", confidence="low")   # same reading
    assert again["id"] == first["id"]
    assert again["duplicate"] is True
    assert again["status"] == "pending"
    assert again["submitted_by"] == "agent-a"
    assert len(_submissions()) == 1


def test_validation_errors_name_the_field():
    from pantry_planner.ingest import submit_origin

    with pytest.raises(ValueError, match="claim_type.*product-of"):
        _submit(claim_type="origin")
    with pytest.raises(ValueError, match=r"country.*Amerca.*United States"):
        _submit(country="Amerca")
    with pytest.raises(ValueError, match="verbatim"):
        _submit(verbatim="US")
    with pytest.raises(ValueError, match="confidence.*high"):
        _submit(confidence="certain")
    with pytest.raises(ValueError, match="product_id.*99999"):
        _submit(product_id=99999)
    with pytest.raises(ValueError, match="note.*1000"):
        _submit(note="n" * 1001)
    with pytest.raises(ValueError, match="product_id"):
        submit_origin({"claim_type": "product-of"}, submitted_by="x")
    assert _submissions() == []


# ─── review ──────────────────────────────────────────────────

def test_approve_copies_one_evidence_row_and_resolves_the_product():
    from pantry_planner import db
    from pantry_planner.ingest import review_submission

    sid = _submit(note="clear photo")["id"]
    out = review_submission(sid, "approve", reviewed_by="reviewer-r", note="checked")

    ev = _evidence(product_id=GARLIC)
    assert len(ev) == 1
    assert ev[0].source == "agent-label"
    assert ev[0].verbatim == "Product of USA"
    assert ev[0].claim_type == "product-of"
    assert ev[0].ingredient_origin == ev[0].manufactured_in == "United States"
    assert ev[0].source_ref == f"submission:{sid}"
    assert ev[0].note == f"clear photo [submission {sid} by agent-a; approved by reviewer-r]"

    summary = db.load_resolved_origins([GARLIC])[GARLIC]
    assert summary.status == "resolved"
    assert summary.country == "United States"

    row = _submissions()[0]
    assert row.status == "approved"
    assert row.reviewed_by == "reviewer-r"
    assert row.review_note == "checked"
    assert row.reviewed_at.startswith("20")
    assert row.evidence_id == ev[0].id
    assert out["evidence_id"] == ev[0].id and out["status"] == "approved"

    with pytest.raises(ValueError, match="approved"):
        review_submission(sid, "approve", reviewed_by="reviewer-r")
    with pytest.raises(ValueError, match="approved"):
        review_submission(sid, "reject", reviewed_by="reviewer-r", note="changed my mind")
    assert len(_evidence(product_id=GARLIC)) == 1


def test_resubmitting_an_approved_reading_is_a_duplicate():
    from pantry_planner.ingest import review_submission

    sid = _submit()["id"]
    review_submission(sid, "approve", reviewed_by="r")
    again = _submit()
    assert again["id"] == sid
    assert again["duplicate"] is True
    assert again["status"] == "approved"
    assert len(_submissions()) == 1


def test_reject_requires_a_note_and_a_rejected_twin_is_returned_not_requeued():
    from pantry_planner.ingest import review_submission

    sid = _submit()["id"]
    with pytest.raises(ValueError, match="note"):
        review_submission(sid, "reject", reviewed_by="r")
    with pytest.raises(ValueError, match="note"):
        review_submission(sid, "reject", reviewed_by="r", note="no")
    assert _submissions()[0].status == "pending"

    out = review_submission(sid, "reject", reviewed_by="r", note="photo shows the importer")
    assert out["status"] == "rejected"
    assert out["review_note"] == "photo shows the importer"
    assert out["reviewed_by"] == "r"
    assert _evidence() == []

    again = _submit(submitted_by="agent-b")
    assert again["id"] == sid
    assert again["status"] == "rejected"
    assert again["review_note"] == "photo shows the importer"
    assert again["duplicate"] is True
    assert len(_submissions()) == 1

    with pytest.raises(ValueError, match="rejected"):
        review_submission(sid, "approve", reviewed_by="r")
    with pytest.raises(ValueError, match="decision"):
        review_submission(sid, "maybe", reviewed_by="r")
    with pytest.raises(ValueError, match="Unknown submission id 404"):
        review_submission(404, "approve", reviewed_by="r")


# ─── THE RIGHT-ANSWER TEST ───────────────────────────────────

def test_pending_changes_no_plan_and_approval_gates_it():
    """With a US exclusion and no evidence the demo planner ships Fresh
    Garlic (13). A pending "Product of USA" reading for it must leave that
    plan untouched; approving it must make the same plan gate on Garlic.
    That is the behaviour the whole write path exists to produce."""
    from pantry_planner import flow
    from pantry_planner.ingest import review_submission
    from pantry_planner.nlsearch import PlanAborted

    def garlic_line():
        plan = flow.run("tomato_penne", exclude=US)
        return next(li for li in plan.line_items if li.ingredient_name == "Garlic")

    assert garlic_line().product_id == GARLIC, "precondition: unfiltered plan ships Fresh Garlic"

    sid = _submit()["id"]
    assert garlic_line().product_id == GARLIC, "a pending submission is not evidence"

    review_submission(sid, "approve", reviewed_by="reviewer-r")
    with pytest.raises(PlanAborted) as exc:
        flow.run("tomato_penne", exclude=US)
    alert = exc.value.execution.aborted
    assert alert.code.value == "excluded_by_origin"
    assert [d["name"] for d in alert.details] == ["Garlic"]


# ─── list + CLI ──────────────────────────────────────────────

def test_list_submissions_pages_oldest_first():
    from pantry_planner.ingest import list_submissions, review_submission

    ids = [_submit(verbatim=f"Product of USA #{i}")["id"] for i in range(3)]
    review_submission(ids[1], "reject", reviewed_by="r", note="blurry")

    page, total = list_submissions("pending", limit=1, offset=0)
    assert total == 2 and [p["id"] for p in page] == [ids[0]]
    page, total = list_submissions("pending", limit=5, offset=1)
    assert total == 2 and [p["id"] for p in page] == [ids[2]]
    page, total = list_submissions(None, limit=10, offset=0)
    assert total == 3 and [p["id"] for p in page] == ids
    assert all(p["product_name"] == "Fresh Garlic" for p in page)
    page, total = list_submissions("rejected", limit=10, offset=0)
    assert (total, [p["id"] for p in page]) == (1, [ids[1]])
    with pytest.raises(ValueError, match="status"):
        list_submissions("bogus", limit=10, offset=0)


def test_cli_lists_and_reviews(capsys):
    from pantry_planner.ingest import main

    a = _submit()["id"]
    b = _submit(verbatim="Product of U.S.A.")["id"]

    assert main(["submissions"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["total"] == 2 and [i["id"] for i in listed["items"]] == [a, b]

    assert main(["review", str(a), "reject", "--note", "blurry", "--by", "cli-user"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "rejected" and out["reviewed_by"] == "cli-user"
    assert _submissions(id=a)[0].review_note == "blurry"

    assert main(["review", str(b), "approve"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "approved" and out["reviewed_by"] == getpass.getuser()
    assert len(_evidence(product_id=GARLIC)) == 1

    assert main(["review", str(b), "approve"]) == 2      # already approved
    assert "approved" in capsys.readouterr().err
    assert main(["submissions", "bogus"]) == 2
    assert main(["review"]) == 2
