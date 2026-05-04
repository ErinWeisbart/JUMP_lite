# Cleanup Quick Start

**Working directory:** `/work/users/jfredinh/projects/cleaning-JUMP_CORE`

## Execute All Steps at Once

```bash
cd /work/users/jfredinh/projects/cleaning-JUMP_CORE

# Step 2: Clean artifacts
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null
rm -f normalization_pipeline_explorer*.html
rm -rf src/norm_3/outputs src/norm_3/multirun
rm -rf src/norm_3/.pixi src/norm_3/.nix-cache

# Step 3: Archive old scripts
mkdir -p _archive/old_sweep_scripts
mv run_focused_v{6,7,8,9,10}_sweep.sh _archive/old_sweep_scripts/ 2>/dev/null
mv run_variance_first_v5_*.sh _archive/old_sweep_scripts/ 2>/dev/null
mv run_v9_test_cp_morphem.sh sweep_runner_single_loop.sh _archive/old_sweep_scripts/ 2>/dev/null

# Step 4: Archive old configs
mkdir -p _archive/old_configs/{sweep,preset}
mv src/norm_3/conf/sweep/focused_*_v{6,7,8,9,10}*.yaml _archive/old_configs/sweep/ 2>/dev/null
mv src/norm_3/conf/preset/*_v{9,10}_*.yaml _archive/old_configs/preset/ 2>/dev/null

# Step 5: Create inventory
mkdir -p _cleanup_docs
echo "=== Repository State ===" > _cleanup_docs/inventory.txt
echo "Active scripts: $(ls -1 run_*.sh | wc -l)" >> _cleanup_docs/inventory.txt
echo "Active configs: $(find src/norm_3/conf -name "*.yaml" | wc -l)" >> _cleanup_docs/inventory.txt
echo "Notebooks: $(find . -name "*.ipynb" | wc -l)" >> _cleanup_docs/inventory.txt
ls -1 run_*.sh >> _cleanup_docs/inventory.txt
du -sh . >> _cleanup_docs/inventory.txt

echo "✅ Cleanup complete! See _cleanup_docs/inventory.txt for summary"
cat _cleanup_docs/inventory.txt
```

## Execute Step-by-Step

See `CLEANUP_PLAN.md` for detailed explanations of each step.

## Verify Original Repo Untouched

```bash
cd /work/users/jfredinh/projects/JUMP_core
git status  # Should show original state
```

## Commit Changes

```bash
cd /work/users/jfredinh/projects/cleaning-JUMP_CORE
git checkout -b cleanup-v1
git add -A
git status
# Review changes, then commit (see CLEANUP_PLAN.md for commit message)
```
