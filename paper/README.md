# Paper build

Published preprint: [`10.5281/zenodo.22082268`](https://doi.org/10.5281/zenodo.22082268).

`main.tex` is the source for the completed four-model intervention paper. Its numeric
tables and plots are generated from the hash-pinned, machine-readable report at
`results/FOUR_MODEL_EXPLORATORY_FRONTIER.json` and the hash-verified derived
scale report at `results/INTERVENTION_SCALE_COMPARISON.json`; they are not typed
manually.

The panel reports verified endpoint estimates. Its held-out layer screen used
five seeded random controls per layer, followed by an independent behavioral
evaluation of the selected intervention. A larger control bank is a future
robustness test for layer selection and the repeated size-associated pattern.

## Build the data-backed assets

From the repository root:

```bash
uv run python scripts/build_exploratory_paper_assets.py
uv run python scripts/build_exploratory_paper_assets.py --check
```

The generator checks the report SHA-256 and structured evidence scope before
writing:

```text
paper/generated/model_diagnostics.tex
paper/generated/primary_results.tex
paper/generated/full_frontier_results.tex
paper/generated/intervention_scale_table.tex
paper/generated/dose_response_plot.tex
paper/generated/frontier_plot.tex
```

## Compile the PDF

The release build uses [Tectonic](https://tectonic-typesetting.github.io/):

```bash
cd paper
tectonic main.tex --outdir build --keep-logs
```

Expected source and build artifacts:

```text
paper/main.tex
paper/references.bib
paper/build/main.pdf
```

For release, copy the compiled PDF to
`output/pdf/selective-resistance-under-pressure.pdf` and render every page to
PNG for visual inspection. Check for clipped tables, overlapping labels,
unresolved citations, blank pages, and unreadable plot legends before
publishing.

## Current study and optional confirmatory path

- `build_exploratory_paper_assets.py` consumes the completed, verified current
  study summary and preserves its exact protocol scope.
- `build_paper_assets.py` is available for a future frozen confirmatory run. It
  requires complete semantic artifacts with one shared launch identity.

The separation makes it possible to reproduce the current findings while also
supporting a stricter tagged replication later.

## Package the accompanying artifacts

The selected fit results, directions, behavioral responses, paired GSM8K
outputs, per-token KL primitives, and required notices form a 32.9 MB
uncompressed canonical bundle kept out of Git. Build it with:

```bash
uv run python scripts/package_exploratory_artifacts.py \
  --archive dist/selective-sycophancy-exploratory-artifacts.zip
```

The tracked `results/EXPLORATORY_ARTIFACT_MANIFEST.json` records every archive
member's relative path, byte count, and SHA-256 except for the embedded copy of
the manifest itself, whose self-hash would be circular. Packaged fit summaries
replace absolute checkout prefixes with `<REPOSITORY>` and retain both original
and packaged hashes. The archive includes the WikiText-2 CC BY-SA notice and
the GSM8K MIT notice required by redistributed or reconstructed dataset text,
plus the project's AGPL license and upstream attribution.
