# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Codeconscious  
**Team Members:** Nihal (Leader), Hari Niveditha, Anvitha Rai, Rakshith Kumar  
**Submission Date:** 25 September 2026

**Challenge:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Metric:** macro-averaged F<sub>0.5</sub> over Source 1 entities  
**Deliverable:** `output/matching_results.tsv` + `output/candidate_pairs.tsv`

---

## 1. Executive Summary

*Blocking + classifier, with the blocking index rebuilt from scratch for scale.*

A sparse TF-IDF cosine between the 1,732,544 Source 1 rows and 4,887,273 Source
2 rows needs ~39 GB of dense `float32` **per chunk** on a machine with 7.6 GB —
so candidate generation was re-engineered into a **sorted packed-key index**
(`crc32(key) << 32 | row_index`, one `uint64` array, lookup = two
`searchsorted` calls), with an adaptive document-frequency cap that drops
ubiquitous tokens instead of blowing up fan-out. Records live in **column
byte blobs** rather than Python objects, Source 1 is streamed, and scoring runs
across processes that share one memory-mapped copy of the data. The result is
a fully supervised pipeline — logistic regression on 37 pairwise features,
threshold swept for macro F<sub>0.5</sub> — that runs end to end over the real
2.5 GB dataset rather than a sample of it.

**Key innovations**

1. A blocking index that costs 8 bytes per posting instead of several GB of
   `dict[str, list[int]]` or a TF-IDF matrix, with `crc32` (not `hash()`) so
   outputs are byte-reproducible across processes and runs.
2. Column-blob + mmap storage that puts ~10M-row target files and their index
   into N worker processes at **one** copy of the data through the page cache.
3. Cost-profiled feature engineering: address scorers replaced by the cheap
   equivalents that reproduce them, because measured on real addresses
   `WRatio` costs ~60 µs vs ~2 µs for `ratio` — paid 116M times.

---

## 2. Methodology

### 2.1 Problem Analysis

*Key insights discovered during EDA.*

**Scale is the first design constraint.** All seven TSVs were profiled before
writing code:

| | train | test |
|---|---:|---:|
| Source 1 | 2,206,821 | 1,732,544 |
| Source 2 | 5,034,616 | 4,887,273 |
| Source 3 | 5,285,603 | 5,082,316 |
| ground-truth links | 7,638,365 (max 11 per S1 row) | — |
| raw size | ~1.32 GB | ~1.19 GB |

All files are clean UTF-8 with exactly four columns and no malformed rows;
every ground-truth ID resolved against the training targets (100%). Holding
these rows as Python `str` objects costs 6–8 GB against a 7.6 GB machine, so
nothing in the pipeline materialises them that way.

**Noise patterns actually present** (each mapped to a handling in §2.3):

| Pattern | Where |
|---|---|
| Abbreviations, legal-suffix drift (`Corp`/`Corporation`, `Pvt`/`Private`, `Ltd`/`Limited`) | both |
| DBA / trade names differing from registered names | both |
| `&` vs `and`, symbol-vs-word variation | both |
| Word-order transpositions (`Coffee Shop` ↔ `Shop Coffee`) | names |
| Typos and transliteration (`Café` ↔ `cafe`) | both |
| `Rd`/`Road`, `St`/`Street` style expansion | addresses |
| Missing components — no PIN, no state, landmark-only (`Near SBI ATM`) | addresses |
| Municipal numbering formats differing between countries | addresses |

**Country is genuinely open.** Training contains only `US` and `India`; the
test split adds **`France`**, never seen during fitting. Country therefore
enters the model as a single binary *agreement* feature with no fixed
vocabulary — see §2.4.

**Class imbalance is extreme.** Blocking deliberately over-generates: roughly
52 candidate pairs per Source 1 row against ~3.5 true links per row, i.e. the
negative class dominates by more than 10:1 *after* blocking.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier  
**Core Innovation:** A sorted packed-key blocking index with an adaptive
document-frequency cap, plus column-blob/mmap storage that makes a 10M-record
supervised pipeline fit — and run in parallel — on 7.6 GB.

```
TSV inputs
  → 1. Normalisation        (normalize.py)         §2.3
  → 2. Candidate generation (blocking.py)          §3  → candidate_pairs.tsv
  → 3. Feature engineering  (features.py)          §4
  → 4. Matching model       (model.py)             §4
  → 5. Output assembly      (pipeline.py)          §4/§5 → matching_results.tsv
```

Each stage is deterministic and seeded; reruns produce byte-identical output,
and output was verified identical between `--workers 1` and `--workers 4`.

### 2.3 Normalisation & Noise Handling

Directly addresses every noise pattern in §2.1.

**Names**

| Noise pattern | Handling |
|---|---|
| Abbreviations (`Corp`/`Corporation`, `Pvt`/`Private`, `Ltd`/`Limited`) | Context-specific abbreviation table (name table ≠ address table) |
| Legal suffix inconsistencies | Suffix stripping to a *core* name; the full form is kept for scoring |
| DBA / trade names | Content-token view + rare-token share |
| `&` vs `and` | Symbol → word expansion before tokenising |
| Word-order transpositions | Sorted-token view; `token_sort_ratio` |
| Typos | Character shingles for blocking + RapidFuzz partial/ratio for scoring |

**Addresses**

| Noise pattern | Handling |
|---|---|
| `Rd`/`Road`, `St`/`Street` | Address-specific abbreviation table |
| Transliteration variants | Unicode NFKD accent folding (`Café` → `cafe`) |
| Missing components (no PIN, no state) | Overlap scored, never required — absence is not penalised |
| Landmark references (`Near SBI ATM`) | Landmark vocabulary expanded; treated as an ordinary token |
| Municipal numbering formats | Digit-set Jaccard + postcode extraction |
| Component reordering | Sorted-token view + token Jaccard over content tokens |

**Implementation note.** Name and address use **separate** abbreviation
tables. A single shared table causes collisions — `Co` must expand to
`company` in a name but never to `Colorado`, and `TN`/`OR`/`GA` are US states
in an address yet nothing in a name. These collisions were found and fixed
during testing.

### 2.4 Handling Country as an Open Set (France)

Training covers `US` and `India`; the test set adds **`France`**, which never
appears in training. The brief explicitly forbids hard-coding, filtering, or
one-hot encoding to `{US, India}`. Country enters the model **only** as a
binary `country_match` feature — *do these two records agree?* — which has no
fixed vocabulary and transfers to any unseen label with zero retraining.

Two further guarantees:

* **Rows are driven by `test_source1.tsv`**, never by a country filter, so
  every French entity is emitted.
* **Region abbreviations are expanded, not filtered.** Address tokens such as
  `CA`/`TN`/`MH` are expanded only in the *address* context; no country value
  is ever rejected or mapped to a fixed list.

---

## 3. Candidate Generation (Blocking)

*Blocking determines the recall ceiling — a pair that never reaches
`candidate_pairs.tsv` can never be recovered downstream. It is also the most
memory-hungry stage, which shaped its design as much as recall did.*

### 3.1 Why not a TF-IDF similarity matrix

The obvious fuzzy-recall channel is a sparse TF-IDF cosine between Source 1
(1,732,544 rows) and Source 2 (4,887,273 rows). On this dataset that is not
merely slow — it is impossible here: one 2,000-row chunk against a single
source needs ~39 GB of dense `float32`, and the target machine has **7.6 GB
total**. Brute-force nearest neighbours is out for the same reason.

### 3.2 The replacement: a sorted packed-key index

Every blocking key is hashed once and packed with its row index into a single
`uint64`:

```
packed = crc32(key) << 32 | row_index
```

Sorting that one array groups all records sharing a key into a contiguous run,
so candidate lookup is **two `np.searchsorted` calls per key**. There is no
`dict[str, list[int]]` (several GB for ~134M postings) and no Python object
per posting — the whole index costs 8 bytes per key.

Two properties make it work:

* **DF cap.** A key shared by more than `df_cap` (default 60) rows is skipped
  at query time. This bounds fan-out *adaptively*: ubiquitous tokens like
  `street` drop out on their own while rare ones survive, with no extra pass
  over the data.
* **`crc32`, not `hash()`.** `hash()` is randomised per process, so blocking
  would differ between runs and outputs would not be reproducible. A crc32
  collision can only *add* a candidate (it unions two key groups), never remove
  one — so recall is unaffected and the classifier filters the extras.

### 3.3 Keys used

- **Blocking keys used:** a mix of exact structure and typo-tolerant structure
  — no phonetic encoding, no TF-IDF:

| Key | Kind | Example / rationale |
|---|---|---|
| `N:` sorted core name | exact | word-order invariant: `"blue ribbon bakery"` |
| `F:` first + last token | exact | survives a changed middle: `acme\|bakery` |
| `I:` initialism | exact | `J P Morgan` → `jpmc` |
| `L:` longest token | exact | the most distinctive single word |
| `G:` sorted-name prefix (5) | fuzzy | tolerates typos inside a long name |
| `C:` sampled char 4-grams | fuzzy | order-invariant shingles of the sorted name |
| `P:` postcode / PIN | exact | highly selective within a country |
| `A:` house number + street token | exact | `123\|main` |

Character 4-grams are sampled with stride 3 (max 6 per record) to keep the
index small: a single typo destroys only its own shingle, while the remaining
sampled shingles still connect the pair. Query keys are consumed **rarest
group first**, because the rarest shared key is the most selective evidence
available and least likely to flood the candidate budget with noise.

- **Candidate pairs generated:** 66,754,930 (38.5 per Source 1
  row, after `--max-candidates 80` and `df_cap=60`). Written in full to
  `output/candidate_pairs.tsv` — one row per Source 1 entity, empty list only
  when blocking found nothing.

- **How you ensured true matches were not lost:**
  * **Nine independent key families are unioned**, so a pair only has to agree
    on *one* of them — exact name, first/last token, initialism, longest token,
    name prefix, character shingles, postcode, house-number|street. Typos
    invalidate an exact key but not the shingle key; reordering invalidates
    nothing because keys are built over a sorted token view.
  * **`df_cap` is applied only at query time**, so rare keys are never
    discarded by a build-time pass that might drop a key which is rare for the
    corpus but decisive for this row.
  * **`--max-candidates` is generous (80)** and the budget is spent from the
    rarest key outward, so the rows that survive are the most specific matches
    rather than the most common.
  * Recall is measured against ground truth during development: a 20% train
    slice held out from *fitting* (but not from blocking) is scored with the
    same keys, so blocking recall and model F<sub>0.5</sub> are tracked
    separately.
  * Because `crc32` collisions only ever *add* pairs, the deterministic-hash
    decision can never lose a true match.

---

## 4. Matching Model

**Features used:**

- **Name features (16):** RapidFuzz `ratio`, `token_sort_ratio`,
  `token_set_ratio`, `partial_ratio`, `WRatio`; token Jaccard (all tokens and
  content tokens); initialism match; length ratio and absolute length
  difference; rare-token share (IDF-weighted); suffix-only difference;
  interaction terms.
- **Address features (9):** `token_sort`, `token_set`, `partial`, `WRatio`,
  `ratio`, token Jaccard, sorted-token ratio, length difference, content
  overlap.
- **Other (12):** postcode agreement, phone agreement, digit-multiset Jaccard,
  country match (open-set binary, §2.4), and cross-field interactions
  (name×address agreement, length-difference products).

37 features total. Every one is a bounded ratio in `[0, 1]`, so no scaler is
fitted and nothing has to be re-derived at inference time.

> **Why addresses use cheaper scorers than names.** Measured on the real data,
> `fuzz.WRatio` costs ~60 µs on address strings and token-set/partial ~22–31 µs
> each, versus ~2 µs for `fuzz.ratio`. Because addresses are long, that premium
> is paid ~116M times over a full run. Since `sorted_tokens` is already
> word-sorted, a plain ratio *is* `token_sort_ratio`, and Jaccard over content
> tokens reproduces `token_set_ratio`'s order-independence — so the address
> columns keep the same semantics at a fraction of the cost. Names (short, and
> where the real discrimination lives) keep the expensive scorers.

**Model type:** class-balanced **logistic regression** (scikit-learn,
BSD-3-Clause), fitted mini-batch with `SGDClassifier` (averaged weights).

- Fitted on blocking candidates labelled against `train_ground_truth.tsv`
  (7,638,365 ground-truth links); negatives downsampled 4:1 per positive to
  fight the imbalance of blocking output.
- Mini-batches rather than one design matrix: holding all labels plus negatives
  as features would need several GB. Batches are discarded after each
  `partial_fit`.
- Row caps `--train-rows 300,000` / `--calib-rows 50,000` bound each pass — a
  37-feature linear model saturates well before it has seen every Source 1 row,
  and cost is linear in rows.
- Falls back to a calibrated weighted heuristic if labels are absent or
  training OOMs, so the pipeline always produces a valid submission.

**Threshold selection method:** F<sub>0.5</sub> optimization on a validation
set. A held-out slice of training rows is kept out of *fitting* entirely and
used only for calibration, so the threshold is never tuned on data the model
saw. The threshold is swept over a grid and the macro-F<sub>0.5</sub>-optimal
value is chosen.

**Licensing / parameter limits:** the final model is MIT/Apache-2.0 compatible
(NumPy BSD-3, SciPy BSD-3, scikit-learn BSD-3, RapidFuzz MIT — all permissive)
and has **37 parameters**, far under the 8B ceiling. No external data lookup of
any kind is performed — see Appendix B.

---

## 5. Results & Error Analysis

- **F<sub>0.5</sub> Score (macro):** **0.7197** on the held-out training
  slice (30,000 Source 1 rows, never used for fitting), at threshold
  **0.80**. The test split has no ground truth, so this is the
  pipeline's own calibrated estimate.

- **Blocking quality (test split):** 66,754,930 candidate pairs over
  1,732,544 Source 1 entities (38.5 per row); 1,456,440 entities
  emitted a non-empty match list and 276,104 emitted an empty one.

**Common false positives (wrong merges):**

1. **Chain/branch siblings.** Businesses sharing a brand, a street, and a
   postcode but differing in unit number or branch suffix — nearly identical
   under every feature except the rare-token share.
2. **Generic names.** `SP SZ`, `Sri Balaji Traders` vs `Sri Balaji Enterprises`
   where the distinguishing token is also a common legal suffix.
3. **Landmark-colliding Indian addresses.** Two distinct addresses that both
   normalise to a shared landmark plus postcode, with no house number to
   separate them.
4. **Empty-address collisions.** Pairs whose addresses are both missing leave
   the name features unopposed; the empty-vs-empty Jaccard is scored as
   agreement rather than as absence of evidence.

**Common false negatives (missed matches):**

1. **Heavy transliteration + abbreviation on long names**, where the
   distinctive token is consumed by suffix stripping on one side only.
2. **French addresses with accented street names** where NFKD folding and
   hyphenation differ, plus unfamiliar region abbreviations that no training
   row ever exercised.
3. **Ground-truth links that blocking never proposed** — irrecoverable by
   construction, which is why §3.3 is written as a recall argument first.
4. **DBA-only matches**, where the recorded name shares almost nothing with
   the linked source's registered name.

---

## 6. Conclusion

Rebuilding blocking as a sorted packed-key index turned a problem that needed
~39 GB per chunk into one that needs 8 bytes per posting, which is what allowed
the *whole* 2.5 GB dataset to be used rather than a sample — and let scoring
run across processes sharing a single mmap of the data. The classifier itself
is deliberately modest (37 features, logistic regression, threshold swept for
F<sub>0.5</sub>) because on this problem the recall ceiling is set by blocking
and the precision ceiling by normalisation; a heavier model would have bought
little next to getting candidate generation and noise handling right. The main
lesson is that measurement beat intuition at every turn: profiling replaced an
assumed 60 µs WRatio with a 2 µs equivalent, revealed a doubling in feature
derivation, and caught a ~7 GB transient allocation before it could fail on a
7.6 GB machine.

---

## Appendix

### A. Code Artefacts

Complete, runnable code ships under `code/business_entity_resolution/` — all
source in `src/`, plus `README.md` and `requirements.txt`.

| File | Responsibility |
|---|---|
| `run_pipeline.py` | CLI entry point: `--mode {test,train-eval,validate}` |
| `pipeline.py` | Orchestration, worker pool, per-chunk output stitching |
| `blocking.py` | Packed sorted-key index, DF-capped lookup, spill/reload |
| `store.py` | Column-blob `RecordStore`, mmap spill/reload, record keys |
| `normalize.py` | Name/address normalisation, abbreviation tables, keys |
| `features.py` | 37 pairwise features |
| `model.py` | Scorer, F<sub>0.5</sub> metrics, threshold sweep |
| `config.py` | All tunables |
| `validate_submission.py` | Local reimplementation of the challenge rules |

**Reproduce end to end** (working directory = the `student_resource/` folder
holding `dataset/` and `utils/`):

```bash
pip install -r code/business_entity_resolution/requirements.txt

# 1. generate the submission files
#    (--workers defaults to every logical core; the processes share one
#     memory-mapped copy of the target data, so more workers != more RAM)
python code/business_entity_resolution/src/run_pipeline.py \
  --data-dir . --mode test --output-dir output

# 2. validate against the challenge rules
python code/business_entity_resolution/src/run_pipeline.py \
  --mode validate --data-dir . --output-dir output

# 3. measure your own score on a held-out train slice
python code/business_entity_resolution/src/run_pipeline.py \
  --data-dir . --mode train-eval --output-dir output
```

Outputs are TSV with the exact required headers, tab-separated, no quoting, LF
endings. The five output rules (one row per Source 1 entity, empty list for
singletons, no duplicate IDs in a list, S2-/S3- IDs only, matches ⊆ candidates)
hold **by construction**: rows are assembled from `test_source1.tsv` itself,
lists are deduplicated preserving order, and predictions are drawn from the
candidate set. `validate_submission.py` re-checks all of them and mirrors the
official `utils/validate_submission.py` semantics — headers compared
case/whitespace-insensitively, `candidate_pairs.tsv` optional, matched ⊆
candidates a warning rather than an error, and ID-existence checking off by
default unless `--check-ids` is passed (it needs both target files resident as
sets, several GB).

### B. Additional Results & Compliance

**Scale / determinism.**

| | train | test |
|---|---:|---:|
| Source 1 | 2,206,821 | 1,732,544 |
| Source 2 | 5,034,616 | 4,887,273 |
| Source 3 | 5,285,603 | 5,082,316 |

* **Column blobs, not objects.** Each field is one contiguous byte buffer plus
  `int32` offsets; `country` is dict-encoded to `uint8` (four labels × 11.7M
  rows ≈ 11 MB instead of ~700 MB of Python strings).
* **Spilled to disk, shared by workers.** Windows uses `spawn`, so workers
  cannot inherit memory; blobs and the key index are written once and
  memory-mapped by every worker — N processes cost one copy of the data, not N.
* **Verified byte-identical** output at `--workers 1` and `--workers 4`.

**Fair-play compliance.** ⚠️ External data lookup is strictly prohibited — no
commercial ER APIs, no government registration lookups, no geocoding APIs, no
internet augmentation. This pipeline performs **only local string and matrix
computation** on the supplied TSV files. It opens no network connections,
resolves nothing against outside data, and enriches records with nothing beyond
what the files already contain.

**Limitations and possible improvements.**

* **Blocking volume.** `--max-candidates` defaults to 80 (38.5
  actually emitted per row). If validation F<sub>0.5</sub> appears
  recall-capped, raise it first — and then `--df-cap`, which lets noisier keys
  through — since no classifier recovers a pair that never became a candidate.
* **Throughput.** Scoring is Python-bound (RapidFuzz plus feature assembly),
  so the GIL makes threads useless and the pipeline uses processes instead.
  A full run is measured in tens of minutes; the row caps on training and
  calibration exist for exactly this reason.
* **No transitive closure.** Pairs are scored independently. Since Source 1 is
  already deduplicated, clique/connected-component reasoning across S2/S3 is
  not required, but it could recover matches with weak pairwise evidence.
* **Linear model.** A gradient-boosted trees model would likely add a point or
  two of F<sub>0.5</sub> on the same features; logistic regression was chosen
  for licence clarity, small-sample robustness, and interpretability.
* **Landmark-heavy Indian addresses** with no street number remain the hardest
  case; the rare-token share feature helps but does not fully solve them.

---

**Note:** Teams can modify sections according to their approach while
maintaining clarity and technical depth.
