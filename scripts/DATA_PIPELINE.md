# JUMP Core Data Pipeline Documentation

This document outlines the data pipeline for generating metadata files used in the JUMP Core project, tracing from raw external sources to final filtered datasets.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXTERNAL DATA SOURCES                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  - JUMP Cell Painting GitHub (crispr.csv.gz, orf.csv.gz)                    │
│  - broad_babel package (well, plate, compound tables)                        │
│  - RefChemDB (compound-gene annotations)                                     │
│  - MOTIVE annotations (compound-compound, compound-gene)                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1: src/standardize_annotations.py                                      │
│  → /work/datasets/jump_core/annotations/inchikey_to_jcp2022_mapping_*.csv   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 2: src/download_images.py                                              │
│  → /work/datasets/jump_core/metadata.parquet                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 3: analysis/.../well_downloading/analyze_metadata.py                   │
│  → metadata/metadata_filtered.parquet                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 4: analysis/.../well_downloading/prepare_negative_controls.py          │
│  → metadata/metadata_negative_controls.parquet                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 5: scripts/compare_metadata_profiles.py                                │
│  → metadata/metadata_dataset.parquet                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 6: scripts/compare_compound_overlap.py                                 │
│  → metadata/metadata_dataset_filtered_4reps.parquet                          │
└─────────────────────────────────────────────────────────────────────────────┘

                    PARALLEL: RefChemDB Annotation Pipeline

┌─────────────────────────────────────────────────────────────────────────────┐
│  analysis/.../annotation_filtering/04_refchemdb_match.ipynb                  │
│  → metadata/refchemdb_conf_jump_matched.parquet                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: InChIKey to JCP2022 Mapping

**Script:** `src/standardize_annotations.py`

**Inputs:**
- `/work/datasets/jump_core/annotations/jump_metadata.duckdb` - JUMP metadata database
- `/work/datasets/jump_core/annotations/annotations_compound_compound.parquet` - MOTIVE compound-compound interactions
- `/work/datasets/jump_core/annotations/annotations_compound_gene.parquet` - MOTIVE compound-gene interactions

**Outputs:**
- `/work/datasets/jump_core/annotations/inchikey_to_jcp2022_mapping_compound_compound.csv`
- `/work/datasets/jump_core/annotations/inchikey_to_jcp2022_mapping_compound_gene.csv`
- `/work/datasets/jump_core/annotations/inchikey_to_jcp2022_mapping_combined.csv`

**Description:**
Translates InChIKey chemical identifiers from MOTIVE annotation databases to JUMP JCP2022 IDs using the JUMP metadata database. Matches are performed on the InChIKey connectivity layer (first 14 characters).

---

## Step 2: JUMP Metadata Generation

**Script:** `src/download_images.py`

**Inputs:**
- `https://github.com/jump-cellpainting/datasets/.../crispr.csv.gz` - CRISPR JCP IDs
- `https://github.com/jump-cellpainting/datasets/.../orf.csv.gz` - ORF JCP IDs
- `/work/datasets/jump_core/annotations/inchikey_to_jcp2022_mapping_combined.csv` - Compound JCP IDs
- `broad_babel` package - Well and plate location metadata
- `jump_portrait` package - Image location metadata

**Outputs:**
- `/work/datasets/jump_core/metadata.parquet`

**Description:**
Combines perturbation lists (CRISPR, ORF, compounds) with location metadata from `broad_babel` and `jump_portrait` packages. Creates a unified metadata file mapping JCP2022 IDs to source/batch/plate/well locations.

**Columns:** `Metadata_Source`, `Metadata_Batch`, `Metadata_Plate`, `Metadata_Well`, `Metadata_JCP2022`

---

## Step 3: Metadata Filtering (Fill Rate)

**Script:** `analysis/annotated_data_selection/well_downloading/analyze_metadata.py`

**Inputs:**
- `/work/datasets/jump_core/metadata.parquet`
- `broad_babel.get_table("compound")` - Compound classification
- `broad_babel.get_table("crispr")` - CRISPR classification
- `broad_babel.get_table("orf")` - ORF classification
- `broad_babel.get_table("plate")` - Plate metadata (for TARGET2 filtering)

**Outputs:**
- `/home/jfredinh/projects/JUMP_core/metadata/metadata_filtered.parquet`
- `/home/jfredinh/projects/JUMP_core/metadata/metadata_filtered_with_target2.parquet`

**Description:**
Filters metadata based on:
1. Excludes source_9 (1536-well plates)
2. Optionally excludes TARGET2 plates
3. Applies 25% minimum plate fill rate threshold
4. Classifies perturbations by type (COMPOUND, CRISPR, ORF, UNKNOWN)

**Added Columns:** `Perturbation_Type`

---

## Step 4: Negative Control Preparation

**Script:** `analysis/annotated_data_selection/well_downloading/prepare_negative_controls.py`

**Inputs:**
- `/home/jfredinh/projects/JUMP_core/metadata/metadata_filtered.parquet`
- `broad_babel.get_table("well")` - Full well metadata
- `broad_babel.get_table("plate")` - Plate metadata

**Outputs:**
- `/home/jfredinh/projects/JUMP_core/metadata/metadata_negative_controls.parquet`

**Description:**
Selects negative control wells from plates in the filtered metadata:
- COMPOUND: JCP2022_033924 (DMSO) - 50% sample
- CRISPR: JCP2022_800001 (Non-targeting) - 50% sample
- ORF: JCP2022_805264, JCP2022_915128 (LacZ/untreated) - 100%

---

## Step 5: Profile-Metadata Matching

**Script:** `scripts/compare_metadata_profiles.py`

**Inputs:**
- `/home/jfredinh/projects/JUMP_core/metadata/metadata_filtered.parquet`
- `/home/jfredinh/projects/JUMP_core/metadata/metadata_negative_controls.parquet`
- `/work/datasets/jump_core_annotated/raw_jump_CP_profiles/profiles.parquet`

**Outputs:**
- `/home/jfredinh/projects/JUMP_core/metadata/metadata_dataset.parquet`

**Description:**
Joins filtered metadata with actual profile data to:
1. Keep only wells that exist in both metadata and profiles
2. Add JCP2022 IDs from profile data
3. Add perturbation type classification

**Columns:** `Metadata_Source`, `Metadata_Batch`, `Metadata_Plate`, `Metadata_Well`, `Metadata_JCP2022`, `Metadata_broad_sample`, `Metadata_Symbol`, `Metadata_pert_type`, `Metadata_Perturbation_Type`

---

## Step 6: Replicate Filtering

**Script:** `scripts/compare_compound_overlap.py`

**Inputs:**
- `/home/jfredinh/projects/JUMP_core/metadata/metadata_dataset.parquet`
- `/home/jfredinh/projects/JUMP_core/metadata/refchemdb_conf_jump_matched.parquet`

**Outputs:**
- `/home/jfredinh/projects/JUMP_core/metadata/metadata_dataset_filtered_4reps.parquet`
- `/home/jfredinh/projects/JUMP_core/metadata/dataset_overlaps/jcpids_source_2_6_8_4reps.parquet`
- `/home/jfredinh/projects/JUMP_core/metadata/dataset_overlaps/jcpids_source_7_4reps.parquet`
- `/home/jfredinh/projects/JUMP_core/metadata/dataset_overlaps/targets_source_*.parquet`

**Description:**
Filters compounds to those with ≥4 replicates:
- Source 2/6/8: Grouped together for replicate counting
- Source 7: Counted separately
- Source 4 (ORF) and Source 13 (CRISPR): Kept as-is

**Added Columns:** `Metadata_Group` (group_high, group_low, group_orf, group_crispr, group_other)

---

## RefChemDB Annotation Pipeline

**Notebook:** `analysis/annotated_data_selection/annotation_filtering/04_refchemdb_match.ipynb`

**Inputs:**
- `analysis/.../outputs/refchemdb/ref_chem_overlap.csv` - RefChemDB with JUMP overlap
- `analysis/.../outputs/metadata/perturbation_metadata.parquet` - JUMP perturbation metadata

**Outputs:**
- `/home/jfredinh/projects/JUMP_core/metadata/refchemdb_conf_jump_matched.parquet`
- (Also copied to: `analysis/.../annotation_filtering/refchemdb_conf_jump_matched.parquet`)

**Description:**
Matches RefChemDB compound-gene annotations with JUMP perturbations:
1. Filters to gene targets only (target_type == "gene")
2. Filters to support > 1 (confident interactions)
3. Adds tier classifications:
   - **CrossModalityTier**: For compound→gene retrieval evaluation
   - **WithinModalityTier**: For compound→compound retrieval evaluation
4. Matches compound mode (Positive/Negative) with perturbation modality (ORF/CRISPR)

**Tier Definitions:**

| Tier | CrossModalityTier Criteria | WithinModalityTier Criteria |
|------|---------------------------|----------------------------|
| Tier0 | 1 target, 1 compound, directional, support≥5 | Same + ≥2 compounds with same criteria |
| Tier1 | 1 target interaction (support≥5), directional | Directional, single mode per target |
| Tier2 | <3 target interactions (support≥5), directional | Directional only |
| Tier3 | All other confident interactions | Any duplicated target |

---

## Key Finding: Unannotated Compounds

**Analysis:** `dataset_curration/compare_jcpids.py`

### Why These Compounds Were Selected

Compound selection is based on **MOTIVE annotations**, not RefChemDB:

| Source | Unique JCPIDs |
|--------|---------------|
| MOTIVE (compound selection) | 4,979 |
| RefChemDB | 5,088 |
| Common | 3,172 |
| Only in MOTIVE | 1,807 |
| Only in RefChemDB | 1,916 |

**MOTIVE** aggregates compound-gene and compound-compound interactions from multiple databases:
- biokg, primekg, pharmebinet, openbiolink, opentargets, hetionet, dgidb, drugrep

**RefChemDB** is a separate curated database with different coverage.

### The 1,191 Unannotated Compounds

When comparing `metadata_dataset_filtered_4reps.parquet` with `ref_chem_overlap.csv`:

| Dataset | Unique Compound JCPIDs |
|---------|------------------------|
| ref_chem_overlap.csv (all) | 5,088 |
| ref_chem_overlap.csv (support > 1) | 2,154 |
| refchemdb_conf_jump_matched.parquet | 2,064 |
| metadata_dataset_filtered_4reps (compounds) | 3,833 |

**1,191 compounds have NO RefChemDB annotation** but are included because they're in MOTIVE.

**MOTIVE Source Breakdown of Unannotated Compounds:**
| MOTIVE Source | Count |
|---------------|-------|
| From Compound-Gene | 1,110 (93%) |
| From Compound-Compound | 593 |
| Only in CC | 81 |
| Only in CG | 598 |
| In both | 512 |

These compounds have annotations in MOTIVE (from databases like DrugBank, OpenTargets, etc.) but **not in RefChemDB**. This is because:
1. MOTIVE and RefChemDB are different annotation sources with different coverage
2. Compound selection was based on MOTIVE, not RefChemDB
3. RefChemDB requires `support > 1` for confidence, which may exclude some interactions

**JUMP Source Distribution of Unannotated Compounds:**
| Source | Rows |
|--------|------|
| source_7 | 5,445 |
| source_6 | 912 |
| source_2 | 911 |
| source_8 | 497 |

### Implication

These compounds **do have known targets** in MOTIVE, just not in RefChemDB. If you need target annotations for these compounds, you could:
1. Use MOTIVE compound-gene annotations directly
2. Cross-reference with the original MOTIVE databases (DrugBank, OpenTargets, etc.)
3. Exclude them from RefChemDB-specific analyses

---

## Missing RefChemDB Compounds

Of the **2,154 RefChemDB compounds with support > 1**, only **1,637 (76.0%)** are in the final filtered metadata.

**517 RefChemDB compounds are missing.** Here's why:

| Stage | Filter | Missing | % of Total |
|-------|--------|---------|------------|
| 1 | Not in MOTIVE selection | 280 | 54.2% |
| 2 | In MOTIVE but no profile match | 107 | 20.7% |
| 3 | Filtered out (< 4 replicates) | 130 | 25.1% |
| **Total** | | **517** | **100%** |

### Breakdown by Stage

**Stage 1: Not in MOTIVE selection (280 compounds, 54.2%)**
- These RefChemDB compounds were never selected for download
- They exist in RefChemDB but not in the MOTIVE compound-compound or compound-gene annotations
- Since compound selection was based on MOTIVE, these were excluded from the start

**Stage 2: No profile match (107 compounds, 20.7%)**
- These compounds are in MOTIVE and were selected for download
- However, no matching wells exist in the JUMP profile data
- Possible reasons: wells failed QC, plates excluded, or images never acquired

**Stage 3: Insufficient replicates (130 compounds, 25.1%)**
- These compounds are in MOTIVE and have profile data
- However, they have fewer than 4 replicates in the relevant sources
- They were filtered out by the 4-replicate minimum threshold

### Coverage Summary

| Metric | Value |
|--------|-------|
| RefChemDB compounds (support > 1) | 2,154 |
| In final filtered metadata | 1,637 |
| **Coverage** | **76.0%** |

To improve RefChemDB coverage, you could:
1. Add RefChemDB-only compounds to the MOTIVE selection (+280)
2. Investigate missing profile data for MOTIVE compounds (+107)
3. Relax the 4-replicate threshold (+130, but lower statistical power)

---

## File Locations Summary

| File | Location | Description |
|------|----------|-------------|
| metadata.parquet | /work/datasets/jump_core/ | Raw JUMP metadata |
| metadata_filtered.parquet | metadata/ | 25% fill rate filtered |
| metadata_negative_controls.parquet | metadata/ | Negative control wells |
| metadata_dataset.parquet | metadata/ | Profile-matched metadata |
| metadata_dataset_filtered_4reps.parquet | metadata/ | **Final filtered dataset** |
| refchemdb_conf_jump_matched.parquet | metadata/ | RefChemDB annotations |
| refchemdb_targets_by_jcp.parquet | metadata/dataset_overlaps/ | Aggregated targets |

---

## Running the Pipeline

```bash
# Step 1: Generate InChIKey mappings
python src/standardize_annotations.py

# Step 2: Generate JUMP metadata (or use --metadata flag with existing file)
python src/download_images.py --metadata /path/to/existing/metadata.parquet

# Step 3: Filter by fill rate
python analysis/annotated_data_selection/well_downloading/analyze_metadata.py

# Step 4: Prepare negative controls
python analysis/annotated_data_selection/well_downloading/prepare_negative_controls.py

# Step 5: Match with profiles
python scripts/compare_metadata_profiles.py

# Step 6: Filter by replicates and generate overlap analysis
python scripts/compare_compound_overlap.py
```

---

## Dependencies

- `polars` - Data processing
- `pandas` - Data processing
- `broad_babel` - JUMP metadata access
- `jump_portrait` - Image location metadata
- `duckdb` - Database queries
- `matplotlib`, `seaborn` - Visualization
- `supervenn` - Set overlap visualization
