# Data policy

- `samples/` contains only small, redistributable fixtures used by tests and examples.
- `raw/` contains downloaded or user-provided source documents and is ignored by Git.
- `evaluation/drafts/` contains version-controlled candidate questions explicitly marked
  `draft`; they require human review and cannot run without `--allow-draft`.
- `evaluation/reviewed/` is reserved for reviewed benchmark cases with reviewer/date.
- `evaluation/generated/` contains disposable generated candidates and remains ignored.
- Evaluation reports and source-text review packets are local artifacts under `artifacts/`.
- Every source document must retain its official URL, retrieval date, publisher, version,
  and content hash.
- Confidential corporate documents must not be added until a separate access-control and
  retention policy has been approved.
