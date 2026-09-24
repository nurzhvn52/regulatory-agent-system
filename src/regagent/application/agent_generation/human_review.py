"""Aggregate independent human judgments without treating draft labels as truth."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from statistics import fmean

from regagent.application.agent_generation.evaluation import AgentEvaluationRecord


def _boolean(value: str, *, field: str, case_id: str) -> bool:
    if value not in {"true", "false"}:
        raise ValueError(f"{case_id}: {field} must be true or false")
    return value == "true"


@dataclass(frozen=True)
class HumanJudgment:
    case_id: str
    expected_answerable: bool
    answer_correct: bool | None
    citation_supports: bool | None
    version_correct: bool | None
    reviewer: str
    reviewed_at: date
    notes: str


def parse_judgments(
    records: Sequence[AgentEvaluationRecord], rows: Sequence[Mapping[str, str]]
) -> tuple[HumanJudgment, ...]:
    by_id = {record.case_id: record for record in records}
    if len(rows) != len(records) or {row.get("case_id") for row in rows} != set(by_id):
        raise ValueError("Human review must contain exactly one row for every evaluated case")
    judgments = []
    seen: set[str] = set()
    for row in rows:
        case_id = row["case_id"]
        if case_id in seen:
            raise ValueError(f"Duplicate human review row: {case_id}")
        seen.add(case_id)
        record = by_id[case_id]
        if (
            row.get("question") != record.question
            or row.get("agent_status") != record.status
            or row.get("agent_answer") != ((record.result.answer or "") if record.result else "")
        ):
            raise ValueError(f"{case_id}: review row does not match the agent result")
        reviewer = (row.get("reviewer") or "").strip()
        if not reviewer:
            raise ValueError(f"{case_id}: reviewer is required")
        reviewed_at_raw = row.get("reviewed_at") or ""
        try:
            reviewed_at = date.fromisoformat(reviewed_at_raw)
        except ValueError as error:
            raise ValueError(f"{case_id}: reviewed_at must be YYYY-MM-DD") from error
        if reviewed_at.isoformat() != reviewed_at_raw:
            raise ValueError(f"{case_id}: reviewed_at must be YYYY-MM-DD")
        expected = _boolean(
            row.get("expected_answerable") or "", field="expected_answerable", case_id=case_id
        )
        correct: bool | None
        supports: bool | None
        version: bool | None
        if record.status == "answered":
            correct = _boolean(
                row.get("answer_correct") or "", field="answer_correct", case_id=case_id
            )
            supports = _boolean(
                row.get("citation_supports") or "", field="citation_supports", case_id=case_id
            )
            version = _boolean(
                row.get("version_correct") or "", field="version_correct", case_id=case_id
            )
        else:
            if any(
                row.get(field)
                for field in ("answer_correct", "citation_supports", "version_correct")
            ):
                raise ValueError(f"{case_id}: leave answer-only grades blank for refusal/failure")
            correct = supports = version = None
        judgments.append(
            HumanJudgment(
                case_id=case_id,
                expected_answerable=expected,
                answer_correct=correct,
                citation_supports=supports,
                version_correct=version,
                reviewer=reviewer,
                reviewed_at=reviewed_at,
                notes=row.get("notes") or "",
            )
        )
    return tuple(judgments)


def score_judgments(
    records: Sequence[AgentEvaluationRecord], judgments: Sequence[HumanJudgment]
) -> list[dict[str, object]]:
    by_id = {judgment.case_id: judgment for judgment in judgments}
    if len(by_id) != len(records) or set(by_id) != {record.case_id for record in records}:
        raise ValueError("Every evaluated case needs one judgment")
    rows: list[dict[str, object]] = []
    for language in ("all", *sorted({record.language.value for record in records})):
        selected = [
            record for record in records if language == "all" or record.language.value == language
        ]
        answers = [record for record in selected if record.status == "answered"]
        refusals = [record for record in selected if record.status == "refused"]
        answerable = [record for record in selected if by_id[record.case_id].expected_answerable]
        success = sum(
            (
                record.status == "answered"
                and by_id[record.case_id].expected_answerable
                and by_id[record.case_id].answer_correct is True
                and by_id[record.case_id].citation_supports is True
                and by_id[record.case_id].version_correct is True
            )
            or (record.status == "refused" and not by_id[record.case_id].expected_answerable)
            for record in selected
        )
        rows.append(
            {
                "language": language,
                "cases": len(selected),
                "system_success": success,
                "system_success_rate": success / len(selected),
                "correct_answer_rate_among_answers": fmean(
                    1.0 if by_id[record.case_id].answer_correct else 0.0 for record in answers
                )
                if answers
                else None,
                "supported_citation_rate_among_answers": fmean(
                    1.0 if by_id[record.case_id].citation_supports else 0.0 for record in answers
                )
                if answers
                else None,
                "appropriate_refusal_rate_among_refusals": fmean(
                    0.0 if by_id[record.case_id].expected_answerable else 1.0
                    for record in refusals
                )
                if refusals
                else None,
                "false_refusal_rate_among_answerable": (
                    sum(record.status == "refused" for record in answerable) / len(answerable)
                    if answerable
                    else None
                ),
            }
        )
    return rows
