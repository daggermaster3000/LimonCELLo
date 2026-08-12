# Data app (`data_app.py`)

[← Home](Home.md)

A Streamlit dashboard to **explore and compare finished runs**: distributions,
scatter, QC, validation against ground truth, per-cilium screening, and a PDF
report.

```powershell
streamlit run data_app.py
```

![Data app](images/dataapp-overview.png)

---

## Adding runs

In the sidebar, paste your **Output folder or run directory**, then:

- **Available runs** multiselect → **Add** the ones you want, or
- **Add ALL found runs** (handy for a whole super-batch — dozens of coverslips), or
- **Add latest run** (newest by modification time).

Each loaded run can be **relabelled**; the label is saved per run and survives
restarts.

### Run discovery

The app finds a "run" by content — **any folder containing
`csv/all_cilia_features.xlsx`** — walking the output tree recursively. This means:

- single-batch `lc-analysis-*` folders,
- super-batch mirror folders (`d11/cv2/…`, `d38/40x FOVs/…`),
- nested re-runs.

When a folder holds both a **stale** flat run and a **fresh** nested
`lc-analysis-*` run, the newest (innermost) one **supersedes** the old one
automatically.

### Speed

The first load of a run reads its `.xlsx` (slow) and writes a
`csv/all_data.parquet` sidecar; every later load reads the parquet
(**~15× faster**). Loads run in parallel and are cached, so re-runs of the app are
near-instant. (Read-only shares simply fall back to the workbook.)

---

## Day / coverslip grouping

Because super-batch keeps each run's path as its label (`d11/cv2`), the app derives
two grouping columns:

- **day** = first path component (`d11`)
- **coverslip** = second component (`cv2`)

So you can **compare between days** while still **splitting or colouring by
coverslip** within a day. In the Distribution tab, **Compare by** defaults to
**day**; use **Color dots by → coverslip** to keep coverslips distinguishable.

![Compare by day, colour by coverslip](images/dataapp-day-coverslip.png)

---

## Tabs

| Tab | What it shows |
|-----|---------------|
| **Overview** | Per-sample overlays, ROIs, MIPs. |
| **Distribution** | Histogram / KDE / box / violin of any metric, grouped across runs, days or coverslips. Per-group normalisation available. |
| **Scatter** | Any metric vs any metric, coloured by run/day/coverslip/class. |
| **QC** | Per-sample and global QC sheets; deviant-sample flagging. |
| **Validation** | Compare against **hand counts** (correlation, Bland–Altman, threshold optimiser) or **per-cilia boxes** (recall, confusion matrix, manual-vs-pipeline class). |
| **Screening** | Review every cilium ROI, keep/reject; decisions saved to `human_validation.csv`. AI validation assists here. |
| **Report** | Build a PDF summarising the loaded runs. |

---

## Screening & AI pre-validation

The Screening tab shows each cilium's ROI so you can **keep/reject** by hand. If a
run was **AI-validated in batch** (`human_validation.csv` with an `ai_score`
column), those decisions and scores load automatically — the cilia show as already
screened by the AI, with scores in the gallery, and are excluded from the
*human-only* screening rate. Re-screening one by hand overrides the AI. See
[AI cilia validator](07-AI-Cilia-Validator.md).

---

## Excluding rejected cilia

The sidebar **Exclude human-rejected cilia from all tabs** toggle removes rejected
cilia from every plot and table, so your analysis reflects the curated set.

---

**Next:** [AI cilia validator →](07-AI-Cilia-Validator.md)
