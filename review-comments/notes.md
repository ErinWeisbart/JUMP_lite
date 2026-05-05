# Working Notes

## Segmentation AP metric

StarDist's `matching.py` ([source](https://github.com/stardist/stardist/blob/e80c6de700693bc228ed3c9ba1dc19c3785667ee/stardist/matching.py#L100)) computes `accuracy = TP/(TP+FP+FN)`, commenting that it is "also known as average precision (?)" with a link to the [Kaggle DSB2018 evaluation](https://www.kaggle.com/c/data-science-bowl-2018#evaluation).

The DSB2018 competition defined its metric as the mean of `TP/(TP+FP+FN)` across IoU thresholds 0.5 to 0.95. What StarDist calls `accuracy` at a single threshold is the per-threshold component of that metric. The paper's Figure 3 label "Instance AP @ IoU=0.5" follows this convention.

Our IoU ablation figure (`plot_iou_ablation.py`) plots this `accuracy` metric at thresholds {0.5, 0.7, 0.8, 0.9}, labeled as "AP". We follow this convention to be consistent with StarDist and DSB2018.
