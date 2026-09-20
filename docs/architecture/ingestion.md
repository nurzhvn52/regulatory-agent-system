# Ingestion pipeline

```text
file -> format parser -> conservative normaliser -> legal structure parser
     -> fixed-window chunker -------------------------------------------|
     -> legal-structure chunker ----------------------------------------|
                                                                      v
                                  versioned PostgreSQL document store
```

## Invariants

- Every source retains an official URL, publisher, retrieval date, and SHA-256 hash.
- A document can contain multiple immutable versions.
- One source version can retain multiple processing pipelines, identified by a SHA-256
  signature of parser, normaliser, and legal-structure rule versions.
- Re-ingesting unchanged content does not create duplicate versions or chunks.
- Section identifiers are deterministic for identical normalised content.
- Every chunk records all source section identifiers, its strategy, and its configuration.
- Fixed-window chunks may cross article boundaries and form the experimental baseline.
- Legal-structure chunks never cross an article boundary.
- Normalisation does not paraphrase or repair legal text.

## Supported structure

The first rule set recognises common Russian and Kazakh forms of parts, chapters,
articles, numbered paragraphs, and subparagraphs. Unknown blocks are retained as body
text under the nearest detected parent; they are never discarded.
