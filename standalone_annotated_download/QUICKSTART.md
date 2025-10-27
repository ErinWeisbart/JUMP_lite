# Quick Start - 3 Commands

## 1. Install
```bash
pixi install
```

## 2. Preview (No Download)
```bash
pixi run preview
```

Shows what will be downloaded without downloading anything.

## 3. Download Sample
```bash
pixi run sample
```

Downloads 10 compounds for testing (~2-5 GB, ~10 minutes).

---

## That's It!

Check the output:
```bash
ls data/manifests/          # CSV files with compound info
ls data/repurposing_hub_tiff/  # Downloaded TIFF files
```

## Next Steps

**Download all 2,721 compounds:**
```bash
pixi run download-all
```

**Filter by target (e.g., kinase inhibitors):**
```bash
python download_annotated_samples.py --target kinase --sample 20
```

**See all options:**
```bash
python download_annotated_samples.py --help
```

---

Full documentation: See [README.md](README.md)
