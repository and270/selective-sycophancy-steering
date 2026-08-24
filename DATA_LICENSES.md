# Data licenses and redistribution policy

Verified on 2026-08-05 for the pinned revisions listed in the study contracts.

## SycophancyEval source

- repository: https://github.com/meg-tong/sycophancy-eval
- pinned commit: `9a1694221e3639887138f61deae344335eca6752`
- source file: `datasets/answer.jsonl`
- license status at verification: no explicit repository license file was found

Public availability is not treated as redistribution permission. This repository therefore commits only source identities, transformation code, split counts, and cryptographic hashes. Users regenerate the 1,816 derived pairs and final JSONL splits locally from the pinned upstream checkout.

The generated question records are gitignored. Current result artifacts retain
model-response maps and item IDs but do not add the materialized source-question
JSONL files to the public repository.

## WikiText-2

- Hugging Face dataset: `Salesforce/wikitext`
- pinned revision: `b08601e04326c79dfdd32d625aee71d232d685c3`
- Hub license tag at verification: `cc-by-sa-3.0`

The parquet is downloaded separately and not committed. The KL
artifacts do embed the 64 selected context strings alongside their row
identities, hashes, and per-context metrics. Consequently, any release bundle
containing those artifacts must carry the following attribution and license
notice:

- **Dataset:** WikiText-2, from the WikiText dataset
- **Authors:** Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher
- **Pinned source:** `Salesforce/wikitext` revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`
- **Paper:** *Pointer Sentinel Mixture Models*, ICLR 2017,
  https://openreview.net/forum?id=Byj72udxe
- **License:** Creative Commons Attribution-ShareAlike 3.0 Unported,
  https://creativecommons.org/licenses/by-sa/3.0/

The bundle selects and reproduces the context strings verbatim; it does not
edit their wording. See `licenses/WIKITEXT-2-CC-BY-SA-3.0.md`, which is included
in the portable study archive together with this file.

## GSM8K

- Hugging Face dataset: `openai/gsm8k`
- pinned revision: `740312add88f781978c0658806c59bc2815b9866`
- Hub license tag at verification: `mit`

The source parquet is downloaded separately and not committed. GSM8K
artifacts retain prompt hashes, source indices, reference answers, raw model
generations, and scoring details needed to inspect the paired sample. Some
generations restate substantial parts of a problem, so portable bundles include
the upstream MIT notice at `licenses/GSM8K-MIT.txt`.
