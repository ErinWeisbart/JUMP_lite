# JUMP_core Cleanup Plan

**Created:** 2026-03-09
**Status:** Ready for execution
**Original repo:** `/work/users/jfredinh/projects/JUMP_core` (UNTOUCHED)
**Working repo:** `/work/users/jfredinh/projects/cleaning-JUMP_CORE` (THIS DIRECTORY)

---

## Current State

- ✅ Repository copied with full git history preserved
- ✅ Large output directories excluded (saved ~1.6TB)
- **Total size:** 40GB (vs 1.6TB+ original)
- **Branch:** first-results-raw
- **Structure:** Same as original, ready for iterative cleanup

---

## Step 2: Remove Build Artifacts & Temporary Files

**Goal:** Clean up generated files that don't need version control

```bash
cd /work/users/jfredinh/projects/cleaning-JUMP_CORE

# Clean Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete

# Clean Jupyter checkpoints
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null

# Clean generated HTML explorers
rm -f normalization_pipeline_explorer*.html

# Clean norm_3 Hydra artifacts (outputs and multirun logs)
rm -rf src/norm_3/outputs src/norm_3/multirun

# Clean pixi cache in norm_3
rm -rf src/norm_3/.pixi src/norm_3/.nix-cache

# Estimate space saved
echo "Artifacts cleaned!"
du -sh .
```

**Expected savings:** ~1-2GB

---

## Step 3: Archive Old Run Scripts

**Goal:** Consolidate the 16 run_*.sh scripts by archiving old versions

```bash
cd /work/users/jfredinh/projects/cleaning-JUMP_CORE

# Create archive directory
mkdir -p _archive/old_sweep_scripts

# Move old version sweep scripts (keep v11/v11_lite only)
mv run_focused_v6_sweep.sh _archive/old_sweep_scripts/
mv run_focused_v7_sweep.sh _archive/old_sweep_scripts/
mv run_focused_v8_sweep.sh _archive/old_sweep_scripts/
mv run_focused_v9_sweep.sh _archive/old_sweep_scripts/
mv run_focused_v10_sweep.sh _archive/old_sweep_scripts/

# Move old variance_first scripts
mv run_variance_first_v5_*.sh _archive/old_sweep_scripts/

# Move other old/deprecated scripts
mv run_v9_test_cp_morphem.sh _archive/old_sweep_scripts/ 2>/dev/null || true
mv sweep_runner_single_loop.sh _archive/old_sweep_scripts/ 2>/dev/null || true

# Document what was kept
ls -1 run_*.sh > _archive/active_scripts.txt
echo "Old scripts archived. Active scripts:"
cat _archive/active_scripts.txt
```

**Keep active:**
- `run_focused_v11_sweep.sh`
- `run_focused_v11_lite_sweep.sh`
- `run_cp_v11_lite_sweep.sh`
- `run_extract_cl3.sh`
- `run_extract_lite_cl3.sh`
- `run_extract_lite_cl3_raw_data.sh`
- `run_openphenom_8clip_std_sweep.sh`
- `run_rerun_dl_v11_lite.sh`

---

## Step 4: Archive Old Hydra Configs

**Goal:** Keep only v11/v11_lite configs, archive v6-v10

```bash
cd /work/users/jfredinh/projects/cleaning-JUMP_CORE

# Create archive for old configs
mkdir -p _archive/old_configs/sweep
mkdir -p _archive/old_configs/preset

# Archive old sweep configs (v6-v10)
mv src/norm_3/conf/sweep/focused_dl_v6.yaml _archive/old_configs/sweep/ 2>/dev/null || true
mv src/norm_3/conf/sweep/focused_dl_v7.yaml _archive/old_configs/sweep/ 2>/dev/null || true
mv src/norm_3/conf/sweep/focused_dl_v8.yaml _archive/old_configs/sweep/ 2>/dev/null || true
mv src/norm_3/conf/sweep/focused_dl_v9.yaml _archive/old_configs/sweep/ 2>/dev/null || true
mv src/norm_3/conf/sweep/focused_dl_v10*.yaml _archive/old_configs/sweep/ 2>/dev/null || true

# Same for CP and cell_count
mv src/norm_3/conf/sweep/focused_cp_v10*.yaml _archive/old_configs/sweep/ 2>/dev/null || true
mv src/norm_3/conf/sweep/focused_cell_count_v10.yaml _archive/old_configs/sweep/ 2>/dev/null || true

# Archive old preset configs
mv src/norm_3/conf/preset/*_v9_*.yaml _archive/old_configs/preset/ 2>/dev/null || true
mv src/norm_3/conf/preset/*_v10_*.yaml _archive/old_configs/preset/ 2>/dev/null || true

# Document remaining configs
find src/norm_3/conf -name "*.yaml" -type f | sort > _archive/active_configs.txt
echo "Old configs archived. Active configs:"
cat _archive/active_configs.txt
```

**Expected result:** ~79 configs → ~30 configs (v11/v11_lite only)

---

## Step 5: Document Current State

**Goal:** Create inventory of what remains after cleanup

```bash
cd /work/users/jfredinh/projects/cleaning-JUMP_CORE

# Create documentation directory
mkdir -p _cleanup_docs

# Inventory of scripts
echo "=== Active Run Scripts ===" > _cleanup_docs/inventory.txt
ls -1 run_*.sh >> _cleanup_docs/inventory.txt

# Inventory of configs
echo -e "\n=== Active Hydra Configs ===" >> _cleanup_docs/inventory.txt
find src/norm_3/conf -name "*.yaml" -type f | wc -l >> _cleanup_docs/inventory.txt
find src/norm_3/conf -name "*.yaml" -type f >> _cleanup_docs/inventory.txt

# Inventory of notebooks
echo -e "\n=== Jupyter Notebooks ===" >> _cleanup_docs/inventory.txt
find . -name "*.ipynb" | wc -l >> _cleanup_docs/inventory.txt

# Inventory of Python scripts
echo -e "\n=== Python Scripts ===" >> _cleanup_docs/inventory.txt
find src -name "*.py" -type f | sort >> _cleanup_docs/inventory.txt
find analysis -name "*.py" -type f | sort >> _cleanup_docs/inventory.txt
find scripts -name "*.py" -type f | sort >> _cleanup_docs/inventory.txt

# Repository size
echo -e "\n=== Repository Size ===" >> _cleanup_docs/inventory.txt
du -sh . >> _cleanup_docs/inventory.txt
du -sh src/ analysis/ scripts/ >> _cleanup_docs/inventory.txt

# Show summary
cat _cleanup_docs/inventory.txt
```

---

## Step 6: Dependency Audit (TODO - Future Session)

**Goal:** Identify and remove unused Python packages

**Approach:**
1. Scan all Python files for actual imports:
   ```bash
   find . -name "*.py" -exec grep -h "^import\|^from" {} + | sort -u > _cleanup_docs/all_imports.txt
   ```

2. Compare against `pyproject.toml` dependencies

3. Identify packages that are never imported

4. Test removing unused packages incrementally

**Current dependencies in pyproject.toml:**
- aliby (local dependency)
- cellpose
- duckdb
- imagecodecs
- lpips
- matplotlib
- numpy, pandas, polars
- scikit-image, scikit-learn
- torch ecosystem
- hydra-core
- copairs, scib-metrics
- Many others...

---

## Step 7: Code Simplification (TODO - Future Session)

**Consolidation opportunities:**

### A. Extract features scripts
- `src/extract_features.py`
- `src/extract_features_fast.py`
- `src/extract_features_with_size_filter.py`

**Action:** Consolidate into single script with CLI flags

### B. Compression scripts
- `src/compress_tif.py`
- `src/compress_tif_single.py`

**Action:** Merge into one with better parameterization

### C. Analysis scripts
Look for duplicate/similar analysis in:
- `analysis/feature_similarity/` (multiple correlation scripts)
- `analysis/segmentation/` (multiple plotting scripts)

---

## Step 8: Create .gitignore Updates

**Add to .gitignore (if not already there):**

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/

# Environments
.venv/
.pixi/

# Outputs
output/
outputs/
logs/
multirun/
*.html

# Data (large directories)
src/norm_3/data/

# Archive
_archive/
_cleanup_docs/
```

---

## Step 9: Git Commit Strategy

After cleanup, commit changes:

```bash
cd /work/users/jfredinh/projects/cleaning-JUMP_CORE

# Create cleanup branch
git checkout -b cleanup-v1

# Stage changes
git add -A

# Commit with detailed message
git commit -m "Clean up repository: archive old scripts and configs

- Removed build artifacts (__pycache__, .pyc files)
- Removed Hydra multirun outputs and logs
- Archived old sweep scripts (v6-v10) to _archive/
- Archived old Hydra configs (v6-v10) to _archive/
- Kept only v11/v11_lite active versions
- Created cleanup documentation and inventory

Repository size reduced from 1.6TB+ to ~40GB
(1.6TB of data outputs excluded from copy)
"
```

---

## Success Metrics

After completing all steps:

- [ ] Repository size: ~35-38GB (from 40GB)
- [ ] Run scripts: ~8-10 active scripts (from 16)
- [ ] Hydra configs: ~30 configs (from 79)
- [ ] No __pycache__ or .pyc files
- [ ] No Hydra multirun artifacts
- [ ] Clean git history preserved
- [ ] Comprehensive inventory document
- [ ] Original repo still untouched at `/work/users/jfredinh/projects/JUMP_core`

---

## Next Steps (Future Sessions)

1. **Test core functionality** - ensure nothing broke
2. **Dependency audit** - remove unused packages
3. **Code consolidation** - merge duplicate scripts
4. **Documentation update** - reflect new structure
5. **Consider further simplifications:**
   - Consolidate analysis notebooks
   - Refactor norm_3 pipeline
   - Create unified CLI interface
   - Add comprehensive tests

---

## Notes

- Original repository remains completely untouched
- All changes are reversible (git history preserved)
- Can cherry-pick specific cleanup steps
- Archive directories are for reference, can be deleted later
- Focus on iterative, testable improvements
