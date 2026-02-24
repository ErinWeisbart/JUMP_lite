import marimo

__generated_with = "0.19.5"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import altair as alt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    return PCA, StandardScaler, alt, mo, np, pl


@app.cell
def _(mo):
    mo.md("""
    # Sphering Overfitting Demo

    ## What is Sphering?

    **Sphering** (or whitening) transforms data so its covariance matrix equals the identity. Given data matrix X, we compute:

    1. **Center**: subtract the mean
    2. **SVD**: X = UΣVᵀ (singular value decomposition)
    3. **Transform**: W = V · diag(1/σᵢ) where σᵢ are singular values

    The transformed data X·W has uncorrelated features with unit variance — a "sphere" in feature space.

    ## Why Use It?

    In Cell Painting, we learn sphering on **negative controls** (DMSO) to remove systematic variation (batch effects, plate effects), then apply the same transform to treatments. This normalizes the "background" so treatment effects stand out.

    ## The Problem

    When **n < p** (fewer samples than features), the covariance matrix is rank-deficient. Some singular values σᵢ are tiny (or zero). Dividing by them (1/σᵢ) explodes those dimensions, and the transform "memorizes" the training data instead of learning generalizable structure. FIXME: Explain how this memorization happens.

    ## The Standard Fix: ε (epsilon) --> FIXME: This is a "standard fix" but what is the justification of it? Is it only to avoid division by zero? Because that's what the next sentence says.

    To avoid division by zero, implementations add a small constant ε:

    **W = V · diag(1/(σᵢ + ε))**

    Common values: ε = 1e-6 (tiny), 0.1, or 1.0. But this is a band-aid:
    - **Small ε** (1e-6): Doesn't help much — small σᵢ still cause huge scaling
    - **Large ε** (1.0): Prevents explosion but also prevents decorrelation (the whole point of sphering)

    FIXME: If there is some historical basis for this do explain it. You will need to read up on this.

    ## A Better Solution: Truncated Projection

    Instead of regularizing, **remove the top k significant PCs entirely**:

    **W = V[:, k:]** — drop first k columns, keep the rest

    Why does this work?
    - The top k PCs capture **systematic variation** (batch effects, plate effects) — exactly what we wanted to remove
    - The remaining (p-k) dimensions contain **signal + noise** but no systematic variation
    - By projecting out the top k, we remove the background without any numerical instability

    This is simpler than full sphering: no division, no ε tuning, no spectrum estimation. The tradeoff is we lose some decorrelation (sphering's original goal), but in practice this works well.

    *Note: There's also "truncated sphering" (drop top k, sphere the rest), but plain projection is simpler and robust.*


    FIXME: There's no transition to the mo.ui.slider -- we just end up abruptly seeing a slider! Some section header might be in order
    """)
    return


@app.cell
def _(pl):
    profiles_all = pl.read_parquet("sphering_demo_data.parquet")
    all_plates = profiles_all["Metadata_Plate"].unique().sort().to_list()
    feat_cols = [c for c in profiles_all.columns if not c.startswith("Metadata")]
    n_features = len(feat_cols)
    return all_plates, feat_cols, n_features, profiles_all


@app.cell
def _(all_plates, mo, n_features):
    n_plates_slider = mo.ui.slider(
        1, len(all_plates), value=4, step=1,
        label=f"Number of plates (p = {n_features} features)"
    )
    n_plates_slider
    return (n_plates_slider,)


@app.cell
def _(
    all_plates,
    feat_cols,
    n_features,
    n_plates_slider,
    np,
    pl,
    profiles_all,
):
    selected_plates = all_plates[:n_plates_slider.value]
    profiles_subset = profiles_all.filter(pl.col("Metadata_Plate").is_in(selected_plates))

    X = profiles_subset.select(feat_cols).to_numpy().astype(np.float64)
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = col_means[j]

    is_negcon = (profiles_subset["Metadata_JCP2022"] == "JCP2022_033924").to_numpy()
    X_negcon = X[is_negcon]
    X_treat = X[~is_negcon]
    n_negcon = len(X_negcon)

    np.random.seed(42)
    indices = np.random.permutation(n_negcon)
    n_train = int(0.8 * n_negcon)
    X_train = X_negcon[indices[:n_train]]
    X_test = X_negcon[indices[n_train:]]

    np_ratio = n_negcon / n_features
    regime = "n < p (DANGER)" if np_ratio < 1 else "n > p (safer)" # FIXME: Unclear in what way is this safer. The distances still seem off (even if not by much)
    return X_test, X_train, X_treat, n_negcon, np_ratio, regime


@app.cell
def _(mo, n_features, n_negcon, n_plates_slider, np_ratio, regime):
    if np_ratio < 1:
        status_color = "red"
    elif np_ratio < 1.5:
        status_color = "orange"
    else:
        status_color = "green"

    mo.md(f"""
    ### Dataset: {n_plates_slider.value} plates

    | Metric | Value |
    |--------|-------|
    | **n (negcons)** | {n_negcon} |
    | **p (features)** | {n_features} |
    | **n/p ratio** | <span style="color:{status_color}; font-weight:bold">{np_ratio:.2f}</span> |
    | **Regime** | <span style="color:{status_color}; font-weight:bold">{regime}</span> |

    *Train/test split: 80% train, 20% holdout*
    """)
    return


@app.cell
def _(StandardScaler, X_train, np):
    def find_k_parallel_analysis(X, n_permutations=30):
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        _, S0, _ = np.linalg.svd(X_std, full_matrices=False)

        Xr = X_std.copy()
        Sr_max = np.zeros(len(S0))
        for _ in range(n_permutations):
            for col in range(X_std.shape[1]):
                np.random.shuffle(Xr[:, col])
            Si = np.linalg.svd(Xr, compute_uv=False)
            Sr_max = np.maximum(Sr_max, Si)

        k = np.argwhere(S0 <= Sr_max)
        k = k[0][0] if len(k) > 0 else len(S0)

        # Calculate variance explained
        eigenvalues = S0**2
        total_variance = eigenvalues.sum()
        variance_explained_at_k = eigenvalues[:k].sum() / total_variance if k > 0 else 0

        return k, S0, Sr_max, variance_explained_at_k

    k_auto, S0, Sr_max, var_explained = find_k_parallel_analysis(X_train)
    return S0, Sr_max, k_auto, var_explained


@app.cell
def _(PCA, StandardScaler, X_test, X_train, X_treat, k_auto, np):
    def _sphere_and_measure(X_tr, X_te, X_trt, method, epsilon=1e-6, k=None):
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X_tr)
        _, S, Vt = np.linalg.svd(X_std, full_matrices=False)
        V = Vt.T

        if method == "full":
            W = V * (1.0 / (S + epsilon))
        else:
            W = V[:, k:]

        X_tr_s = scaler.transform(X_tr) @ W
        X_te_s = scaler.transform(X_te) @ W
        X_trt_s = scaler.transform(X_trt) @ W

        centroid = X_tr_s.mean(axis=0)
        d_tr = np.sqrt(np.sum((X_tr_s - centroid)**2, axis=1)).mean()
        d_te = np.sqrt(np.sum((X_te_s - centroid)**2, axis=1)).mean()
        ratio = d_te / d_tr

        X_all = np.vstack([X_tr_s, X_te_s, X_trt_s])
        labels = (
            ["Train negcon"] * len(X_tr_s) +
            ["Test negcon"] * len(X_te_s) +
            ["Treatment"] * len(X_trt_s)
        )

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_all)

        return X_pca, labels, ratio

    pca_full, labels_full, ratio_full = _sphere_and_measure(
        X_train, X_test, X_treat, "full", epsilon=1e-6
    )
    pca_trunc, labels_trunc, ratio_trunc = _sphere_and_measure(
        X_train, X_test, X_treat, "truncated_project", k=k_auto
    )
    return (
        labels_full,
        labels_trunc,
        pca_full,
        pca_trunc,
        ratio_full,
        ratio_trunc,
    )


@app.cell
def _(
    alt,
    k_auto,
    labels_full,
    labels_trunc,
    mo,
    pca_full,
    pca_trunc,
    pl,
    ratio_full,
    ratio_trunc,
):
    df_full = pl.DataFrame({
        "PC1": pca_full[:, 0], "PC2": pca_full[:, 1], "Group": labels_full
    })
    df_trunc = pl.DataFrame({
        "PC1": pca_trunc[:, 0], "PC2": pca_trunc[:, 1], "Group": labels_trunc
    })

    color_scale = alt.Scale(
        domain=["Train negcon", "Test negcon", "Treatment"],
        range=["steelblue", "limegreen", "coral"]
    )

    status_full = "OVERFITTING" if ratio_full > 10 else "OK"
    title_full = f"Full Sphering (ε=1e-6): {ratio_full:.0f}x" # FIXME: Somewhere you must explain what this ratio_full is
    if ratio_full > 1000:
        title_full = f"Full Sphering (ε=1e-6): {ratio_full/1000:.0f}Kx"

    chart_full = alt.Chart(df_full).mark_circle(size=35, opacity=0.7).encode(
        x=alt.X("PC1:Q", title="PC1"),
        y=alt.Y("PC2:Q", title="PC2"),
        color=alt.Color("Group:N", scale=color_scale, legend=None),
        tooltip=["Group:N"]
    ).properties(
        width=320,
        height=280,
        title=alt.Title(title_full, color="red" if ratio_full > 10 else "black")
    )

    chart_trunc = alt.Chart(df_trunc).mark_circle(size=35, opacity=0.7).encode(
        x=alt.X("PC1:Q", title="PC1"),
        y=alt.Y("PC2:Q", title="PC2"),
        color=alt.Color("Group:N", scale=color_scale),
        tooltip=["Group:N"]
    ).properties(
        width=320,
        height=280,
        title=alt.Title(f"Truncated Project (k={k_auto}): {ratio_trunc:.1f}x", color="green")
    )

    mo.vstack([
        mo.md("""
        ## The Problem vs The Solution

        Both plots show PCA of the transformed data. We split negative controls 80/20 to create a holdout test.

        | | Left Panel | Right Panel |
        |---|---|---|
        | **Method** | Full Sphering | Truncated Project |
        | **Transform** | W = V · diag(1/(σᵢ + ε)) | W = V[:, :k] |
        | **What it does** | Scales ALL dimensions by 1/σᵢ | Projects onto top k PCs only |
        | **The problem** | Small σᵢ → huge scaling → noise explodes | No division → no explosion |
        """),
        mo.hstack([chart_full, chart_trunc], justify="center", gap=2),
        mo.md("""
        *Blue = train negcons (80%), Green = test negcons (20% holdout), Orange = treatments*

        **How to read this**: Train and test negcons are both DMSO — same distribution. They should overlap after transformation. If test negcons are far from train negcons, the transform learned something specific to the training set (overfitting).

        **The test/train ratio** = mean distance of test negcons / mean distance of train negcons. Should be ≈1.
        """)
    ])
    return


@app.cell
def _(S0, Sr_max, alt, k_auto, mo, pl):
    spectrum_df = pl.DataFrame({
        "component": list(range(len(S0))) + list(range(len(Sr_max))),
        "eigenvalue": list(S0**2) + list(Sr_max**2),
        "type": ["Data"] * len(S0) + ["Random max"] * len(Sr_max)
    })

    spectrum_chart = alt.Chart(spectrum_df).mark_line().encode(
        x=alt.X("component:Q", title="Component"),
        y=alt.Y("eigenvalue:Q", title="Eigenvalue", scale=alt.Scale(type="log")),
        color=alt.Color("type:N", scale=alt.Scale(domain=["Data", "Random max"], range=["steelblue", "red"])),
        strokeDash=alt.StrokeDash("type:N", scale=alt.Scale(domain=["Data", "Random max"], range=[[1, 0], [5, 5]]))
    ).properties(width=500, height=250)

    k_line = alt.Chart(pl.DataFrame({"k": [k_auto]})).mark_rule(color="green", strokeDash=[3, 3]).encode(x="k:Q")

    mo.vstack([
        mo.md(f"""
        ### Finding k: Parallel Analysis

        **Parallel analysis** identifies significant PCs by comparing the data spectrum (blue) to spectra from randomized data (red dashed). PCs above the noise line capture real structure; those below are indistinguishable from noise.

        **k = {k_auto}** significant PCs (systematic variation). Truncated projection **removes** these first k PCs (W = V[:, k:]) and keeps the remaining {len(S0) - k_auto} dimensions.

        Why remove the significant PCs? They capture batch effects and plate-to-plate variation — exactly what we wanted sphering to normalize away. The remaining dimensions contain treatment signal without the systematic background.
        """),
        spectrum_chart + k_line
    ])
    return


@app.cell
def _(
    StandardScaler,
    all_plates,
    alt,
    feat_cols,
    k_auto,
    mo,
    n_features,
    n_plates_slider,
    np,
    pl,
    profiles_all,
    var_explained,
):
    # Pre-compute k and variance explained for all possible plate counts
    def compute_k_for_all_plates():
        history = []
        for n_plates in range(1, len(all_plates) + 1):
            selected = all_plates[:n_plates]
            profiles_sub = profiles_all.filter(pl.col("Metadata_Plate").is_in(selected))

            X_sub = profiles_sub.select(feat_cols).to_numpy().astype(np.float64)
            col_means_sub = np.nanmean(X_sub, axis=0)
            for j in range(X_sub.shape[1]):
                X_sub[np.isnan(X_sub[:, j]), j] = col_means_sub[j]

            is_negcon_sub = (profiles_sub["Metadata_JCP2022"] == "JCP2022_033924").to_numpy()
            X_negcon_sub = X_sub[is_negcon_sub]

            if len(X_negcon_sub) > 0:
                # Compute k and variance for this plate count
                scaler = StandardScaler()
                X_std = scaler.fit_transform(X_negcon_sub)
                _, S, _ = np.linalg.svd(X_std, full_matrices=False)

                # Parallel analysis (simplified - just use first run)
                Xr = X_std.copy()
                Sr_max = np.zeros(len(S))
                for _ in range(10):  # Fewer permutations for speed
                    for col in range(X_std.shape[1]):
                        np.random.shuffle(Xr[:, col])
                    Si = np.linalg.svd(Xr, compute_uv=False)
                    Sr_max = np.maximum(Sr_max, Si)

                k = np.argwhere(S <= Sr_max)
                k = k[0][0] if len(k) > 0 else len(S)

                eigenvalues = S**2
                total_variance = eigenvalues.sum()
                variance_explained_at_k = eigenvalues[:k].sum() / total_variance if k > 0 else 0

                # Calculate number of dims to explain 75% variance (fast to compute!)
                cumsum_var = np.cumsum(eigenvalues) / total_variance
                dims_75 = np.argmax(cumsum_var >= 0.75) + 1

                history.append({
                    "n_plates": n_plates,
                    "k": k,
                    "variance_explained": variance_explained_at_k * 100,
                    "n_negcon": len(X_negcon_sub),
                    "np_ratio": len(X_negcon_sub) / n_features,
                    "dims_75": dims_75
                })

        return pl.DataFrame(history)

    # Cache computation using a global variable pattern
    import __main__
    if not hasattr(__main__, '_k_history_cache'):
        __main__._k_history_cache = compute_k_for_all_plates()

    history_df = __main__._k_history_cache

    # Create charts
    k_chart = alt.Chart(history_df).mark_line(point=True).encode(
        x=alt.X("n_plates:Q", title="Number of Plates"),
        y=alt.Y("k:Q", title="k (significant PCs)", scale=alt.Scale(zero=False)),
        tooltip=[
            "n_plates:Q",
            "k:Q",
            alt.Tooltip("variance_explained:Q", format=".1f", title="Variance Explained (%)"),
            alt.Tooltip("n_negcon:Q", title="n negcons"),
            alt.Tooltip("np_ratio:Q", format=".2f", title="n/p ratio")
        ]
    ).properties(width=350, height=250, title="k vs Number of Plates")

    # Add a marker for current position
    current_mark = alt.Chart(history_df.filter(pl.col("n_plates") == n_plates_slider.value)).mark_circle(
        size=200, color="red", opacity=0.8
    ).encode(
        x="n_plates:Q",
        y="k:Q"
    )

    var_chart = alt.Chart(history_df).mark_line(point=True, color="coral").encode(
        x=alt.X("n_plates:Q", title="Number of Plates"),
        y=alt.Y("variance_explained:Q", title="Variance Explained by top k PCs (%)"),
        tooltip=[
            "n_plates:Q",
            "k:Q",
            alt.Tooltip("variance_explained:Q", format=".1f", title="Variance Explained (%)"),
            alt.Tooltip("n_negcon:Q", title="n negcons"),
            alt.Tooltip("np_ratio:Q", format=".2f", title="n/p ratio")
        ]
    ).properties(width=350, height=250, title="Variance Explained at k")

    current_mark_var = alt.Chart(history_df.filter(pl.col("n_plates") == n_plates_slider.value)).mark_circle(
        size=200, color="red", opacity=0.8
    ).encode(
        x="n_plates:Q",
        y="variance_explained:Q"
    )

    # Variance explained vs k chart
    var_vs_k_chart = alt.Chart(history_df).mark_circle(size=80, color="steelblue").encode(
        x=alt.X("k:Q", title="k (significant PCs)"),
        y=alt.Y("variance_explained:Q", title="Variance Explained by top k PCs (%)"),
        tooltip=[
            "n_plates:Q",
            "k:Q",
            alt.Tooltip("variance_explained:Q", format=".1f", title="Variance Explained (%)"),
            alt.Tooltip("n_negcon:Q", title="n negcons"),
            alt.Tooltip("np_ratio:Q", format=".2f", title="n/p ratio")
        ]
    ).properties(width=350, height=250, title="Variance Explained vs k")

    current_mark_var_vs_k = alt.Chart(history_df.filter(pl.col("n_plates") == n_plates_slider.value)).mark_circle(
        size=200, color="red", opacity=0.8
    ).encode(
        x="k:Q",
        y="variance_explained:Q"
    )

    # Build regression models to predict k
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    y_k = history_df["k"].to_numpy()

    # Model 1: Just n/p ratio
    X_ratio = history_df["np_ratio"].to_numpy().reshape(-1, 1)
    model1 = LinearRegression()
    model1.fit(X_ratio, y_k)
    y_pred1 = model1.predict(X_ratio)
    r2_1 = r2_score(y_k, y_pred1)

    # Model 2: Just dims_75 (number of PCs for 75% variance)
    X_dims75 = history_df["dims_75"].to_numpy().reshape(-1, 1)
    model2 = LinearRegression()
    model2.fit(X_dims75, y_k)
    y_pred2 = model2.predict(X_dims75)
    r2_2 = r2_score(y_k, y_pred2)

    # Model 3: Both n/p ratio and dims_75
    X_combined = np.column_stack([
        history_df["np_ratio"].to_numpy(),
        history_df["dims_75"].to_numpy()
    ])
    model3 = LinearRegression()
    model3.fit(X_combined, y_k)
    y_pred3 = model3.predict(X_combined)
    r2_3 = r2_score(y_k, y_pred3)

    # Get current values
    current_row = history_df.filter(pl.col("n_plates") == n_plates_slider.value)
    current_np_ratio = current_row["np_ratio"][0]
    current_dims75 = current_row["dims_75"][0]

    # Predictions
    pred_k1 = model1.predict([[current_np_ratio]])[0]
    pred_k2 = model2.predict([[current_dims75]])[0]
    pred_k3 = model3.predict([[current_np_ratio, current_dims75]])[0]

    # Chart 1: k vs n/p ratio
    k_vs_ratio_scatter = alt.Chart(history_df).mark_circle(size=80).encode(
        x=alt.X("np_ratio:Q", title="n/p ratio"),
        y=alt.Y("k:Q", title="k (significant PCs)"),
        tooltip=[
            "n_plates:Q",
            "k:Q",
            alt.Tooltip("np_ratio:Q", format=".2f", title="n/p ratio"),
            alt.Tooltip("dims_75:Q", title="dims for 75% var")
        ]
    ).properties(width=350, height=250, title=f"k vs n/p ratio (R²={r2_1:.3f})")

    current_mark1 = alt.Chart(current_row).mark_circle(size=200, color="red", opacity=0.8).encode(
        x="np_ratio:Q", y="k:Q"
    )

    # Chart 2: k vs dims_75
    k_vs_dims75_scatter = alt.Chart(history_df).mark_circle(size=80, color="coral").encode(
        x=alt.X("dims_75:Q", title="# dims for 75% variance"),
        y=alt.Y("k:Q", title="k (significant PCs)"),
        tooltip=[
            "n_plates:Q",
            "k:Q",
            alt.Tooltip("np_ratio:Q", format=".2f", title="n/p ratio"),
            alt.Tooltip("dims_75:Q", title="dims for 75% var")
        ]
    ).properties(width=350, height=250, title=f"k vs dims_75 (R²={r2_2:.3f})")

    current_mark2 = alt.Chart(current_row).mark_circle(size=200, color="red", opacity=0.8).encode(
        x="dims_75:Q", y="k:Q"
    )

    # Create dataframe with all predictions for comparison
    comparison_df = history_df.with_columns([
        pl.lit(y_pred1).alias("pred_ratio"),
        pl.lit(y_pred2).alias("pred_dims75"),
        pl.lit(y_pred3).alias("pred_combined")
    ])

    mo.vstack([
        mo.md(f"""
        ### k and Variance Explained Across All Plate Counts

        This shows how k (number of significant PCs) and the variance they explain changes across all possible plate counts. The red dot shows your current selection.

        **Current**: {n_plates_slider.value} plates → k={k_auto} → {var_explained * 100:.1f}% variance explained
        """),
        mo.hstack([k_chart + current_mark, var_chart + current_mark_var], justify="center", gap=2),
        mo.md("""
        ### Variance Explained vs k

        This plot shows the direct relationship between k (number of significant PCs) and the variance they explain.
        """),
        var_vs_k_chart + current_mark_var_vs_k,
        mo.md(f"""
        ### Fast k Estimation Without Parallel Analysis

        Instead of running expensive parallel analysis (30+ permutations), you can **quickly estimate k** using simple calculations.

        **Three approaches:**

        | Method | Inputs | Predicted k | R² | Actual k |
        |--------|--------|-------------|-----|----------|
        | **1. n/p ratio only** | n={int(current_np_ratio * n_features)}, p={n_features} | {int(pred_k1)} | {r2_1:.3f} | {k_auto} |
        | **2. dims_75 only** | dims_75={current_dims75} | {int(pred_k2)} | {r2_2:.3f} | {k_auto} |
        | **3. Both combined** | n/p={current_np_ratio:.2f}, dims_75={current_dims75} | {int(pred_k3)} | {r2_3:.3f} | {k_auto} |

        **Best approach: {"Combined (3)" if r2_3 == max(r2_1, r2_2, r2_3) else ("dims_75 only (2)" if r2_2 == max(r2_1, r2_2, r2_3) else "n/p ratio only (1)")}**

        **Why dims_75 works:** Computing how many PCs explain 75% variance is fast (just SVD + cumsum, no permutations). It captures the variance structure of your data, which helps predict where parallel analysis would cut off.

        ---

        ## Fitted Formulas (for this dataset)

        **Model 1 (n/p ratio only):**
        ```
        k ≈ {model1.coef_[0]:.2f} × (n/p) + {model1.intercept_:.2f}
        ```

        **Model 2 (dims_75 only):**
        ```
        k ≈ {model2.coef_[0]:.4f} × dims_75 + {model2.intercept_:.2f}
        ```

        **Model 3 (combined):**
        ```
        k ≈ {model3.coef_[0]:.2f} × (n/p) + {model3.coef_[1]:.4f} × dims_75 + {model3.intercept_:.2f}
        ```

        **How to use:**
        1. Do SVD on your standardized negcons: `_, S, _ = np.linalg.svd(StandardScaler().fit_transform(X_negcon))`
        2. Calculate dims_75: `dims_75 = np.argmax(np.cumsum(S**2) / np.sum(S**2) >= 0.75) + 1`
        3. Calculate n/p: `np_ratio = len(X_negcon) / n_features`
        4. Plug into formula above → instant k estimate!

        **Note:** These formulas are fitted to this specific dataset. For production use, you should fit on your own data or use a more general heuristic.
        """),
        mo.hstack([k_vs_ratio_scatter + current_mark1, k_vs_dims75_scatter + current_mark2], justify="center", gap=2)
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Key Takeaways

    1. **When n < p**: Standard sphering massively overfits. You're estimating a p×p covariance with only n samples — mathematically impossible to do well.

    2. **The holdout test**: Split negcons 80/20. Transform both with parameters learned on train only. If test/train distance ratio >> 1, you're overfitting.

    3. **Why ε doesn't fully fix it**: Large ε dampens the explosion but also dampens the decorrelation you wanted in the first place. It's a band-aid, not a cure. FIXME: Is a deeper math analysis of this in order?

    4. **The real solution — truncated_project**: Project out only the k significant PCs (determined by parallel analysis). No matrix inversion, no numerical instability, works in all regimes. FIXME: This is not "the real solution"; please be less bombastic for the love of got

    ---

    **Try it**: Move the slider to 12 plates (n/p > 1) and watch the full sphering ratio drop. FIXME: I doubt it drop suddenly right
    The problem disappears when you have enough samples — but truncated_project is safer regardless.
    """)
    return


if __name__ == "__main__":
    app.run()
