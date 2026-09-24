"""Load an expert-completed CSV and write provenance-bound human scores."""

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from regagent.application.agent_generation.evaluation import AgentEvaluationRecord
from regagent.application.agent_generation.human_review import parse_judgments, score_judgments


def score_agent_review(
    run_dir: Path, output_path: Path, *, allow_draft: bool = False
) -> dict[str, Any]:
    if output_path.exists():
        raise ValueError("Score output already exists; choose a new filename")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] not in {"complete_pilot_ungraded", "complete_ungraded"}:
        raise ValueError("Agent run is incomplete or its corpus changed")
    if manifest["retrieval_labels"] == "pilot_unreviewed" and not allow_draft:
        raise ValueError("Questions have unreviewed labels; use --allow-draft for pilot scoring")
    results_raw = (run_dir / "results.jsonl").read_bytes()
    if hashlib.sha256(results_raw).hexdigest() != manifest["results_sha256"]:
        raise ValueError("Agent results do not match the run manifest")
    records = [
        AgentEvaluationRecord.model_validate_json(line)
        for line in results_raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != manifest["case_count"]:
        raise ValueError("Agent result count does not match the run manifest")
    review_path = run_dir / "human_review.csv"
    review_raw = review_path.read_bytes()
    with review_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    judgments = parse_judgments(records, rows)
    scores = score_judgments(records, judgments)
    report = {
        "status": (
            "pilot_reviewed_answers_unreviewed_questions"
            if manifest["retrieval_labels"] == "pilot_unreviewed"
            else "reviewed"
        ),
        "experiment_id": manifest["experiment_id"],
        "created_at": datetime.now(UTC).isoformat(),
        "results_sha256": manifest["results_sha256"],
        "review_sha256": hashlib.sha256(review_raw).hexdigest(),
        "case_count": len(records),
        "reviewers": sorted({judgment.reviewer for judgment in judgments}),
        "rubric": (
            "Success: a correct, cited, correct-version answer to an answerable question; "
            "or refusal on an unanswerable question. Failures count as unsuccessful."
        ),
        "summary": scores,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
