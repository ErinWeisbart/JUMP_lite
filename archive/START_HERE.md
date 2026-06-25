# Welcome to Cleaning JUMP_CORE

This is a **cleaned-up copy** of the JUMP_core repository for simplification and refactoring.

## Current Status

✅ **Copied from:** `/work/users/jfredinh/projects/JUMP_core`
✅ **Git history:** Fully preserved
✅ **Size:** 40GB (vs 1.6TB+ original, large outputs excluded)
✅ **Branch:** first-results-raw
⏳ **Cleanup status:** Ready to begin

## Original Repo Status

The original repository at `/work/users/jfredinh/projects/JUMP_core` is **completely untouched** and remains your working reference.

## Next Steps

### Option 1: Quick Cleanup (Recommended)
Run all cleanup steps at once:
```bash
cd /work/users/jfredinh/projects/cleaning-JUMP_CORE
cat CLEANUP_QUICKSTART.md  # Review the commands
# Then copy-paste the command block from CLEANUP_QUICKSTART.md
```

### Option 2: Step-by-Step Cleanup
Follow the detailed plan:
```bash
cd /work/users/jfredinh/projects/cleaning-JUMP_CORE
less CLEANUP_PLAN.md  # Read the full plan
# Execute steps 2-5 one at a time
```

## What Will Be Cleaned Up?

1. **Build artifacts** - `__pycache__`, `.pyc` files, Jupyter checkpoints
2. **Old scripts** - Archive sweep scripts v6-v10 (keep v11/v11_lite)
3. **Old configs** - Archive Hydra configs v6-v10 (keep v11/v11_lite)
4. **Hydra outputs** - Remove multirun logs and outputs
5. **Documentation** - Create inventory of what remains

**Expected result:** ~35-38GB, cleaner structure, same functionality

## Important Notes

- All changes are **reversible** (git history preserved)
- Original repo remains **untouched**
- Cleanup is **iterative and safe**
- See `CLEANUP_PLAN.md` for full details
- Future steps include dependency audit and code consolidation

## Files in This Directory

- `START_HERE.md` - This file
- `CLEANUP_PLAN.md` - Detailed cleanup plan with explanations
- `CLEANUP_QUICKSTART.md` - Quick reference for running all steps
- Everything else - Same as original repo structure

## Ready to Begin?

When you start your next session:
```bash
cd /work/users/jfredinh/projects/cleaning-JUMP_CORE
# Review CLEANUP_QUICKSTART.md and execute the commands
```
