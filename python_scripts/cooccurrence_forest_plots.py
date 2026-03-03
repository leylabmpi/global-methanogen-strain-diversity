#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cooccurrence_plots_adult.py
"""

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.rcParams["svg.fonttype"] = "none"  
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency, fisher_exact, norm, beta

try:
    from statsmodels.stats.contingency_tables import Table2x2

    _HAVE_STATSMODELS = True
except Exception:
    Table2x2 = None
    _HAVE_STATSMODELS = False

FOREST_FIGSIZE = (5.4, 3.6)
FOREST_AXES_RECT = [0.22, 0.18, 0.70, 0.74] 

COUNTRY_ORDER = ["Gabon", "Germany", "Vietnam"] 

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def savefig_both(fig, outbase: Path) -> None:
    png = str(outbase) + ".png"
    svg = str(outbase) + ".svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, format="svg", bbox_inches="tight")
    print(f"Saved: {png}\n       {svg}")

def _require_columns(df: pd.DataFrame, cols) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

def coerce_nonneg_int_series(
    s: pd.Series, colname: str, allow_missing: bool = False
) -> pd.Series:
    """
    Coerce a column to pandas nullable integer with validation:
    - must be integer-valued and >= 0
    - missing allowed only if allow_missing=True
    """
    x = pd.to_numeric(s, errors="coerce")

    if not allow_missing:
        if x.isna().any():
            bad = s[x.isna()].index.tolist()[:10]
            raise ValueError(
                f"Column '{colname}' contains missing/non-numeric values "
                f"(showing up to 10 row indices): {bad}"
            )

    non_missing = x.dropna()
    if not np.all(np.isclose(non_missing, np.round(non_missing))):
        bad_idx = non_missing[~np.isclose(non_missing, np.round(non_missing))].index.tolist()[:10]
        raise ValueError(
            f"Column '{colname}' contains non-integer values "
            f"(showing up to 10 row indices): {bad_idx}"
        )

    if (non_missing < 0).any():
        bad_idx = non_missing[non_missing < 0].index.tolist()[:10]
        raise ValueError(
            f"Column '{colname}' contains negative counts "
            f"(showing up to 10 row indices): {bad_idx}"
        )

    return x.round().astype("Int64")

def haldane_anscombe_2x2(a, b, c, d):
    """Return corrected cells if any are zero (for OR/CI only)."""
    if min(a, b, c, d) == 0:
        return (a + 0.5, b + 0.5, c + 0.5, d + 0.5), True
    return (a, b, c, d), False

def or_and_ci_2x2_wald(a, b, c, d, alpha=0.05):
    """Odds ratio with Wald CI on log scale; uses Haldane–Anscombe if needed."""
    (ac, bc, cc, dc), corrected = haldane_anscombe_2x2(a, b, c, d)
    or_ = (ac * dc) / (bc * cc)
    se = math.sqrt(1 / ac + 1 / bc + 1 / cc + 1 / dc)
    z = norm.ppf(1 - alpha / 2.0)
    lo = math.exp(math.log(or_) - z * se)
    hi = math.exp(math.log(or_) + z * se)
    return or_, lo, hi, se, corrected

def or_ci_statsmodels(a, b, c, d, alpha=0.05, method="exact"):
    """
    Odds ratio + CI via statsmodels Table2x2.

    method:
      - "exact": exact conditional (Fisher inversion) CI for OR
      - "profile": profile likelihood CI for OR

    Returns dict with OR point estimate (unadjusted sample OR),
    and CI bounds (may be 0/inf in sparse settings).
    """
    if not _HAVE_STATSMODELS:
        raise RuntimeError(
            "statsmodels is required for exact/profile OR confidence intervals. "
            "Install it (e.g., pip install statsmodels) or run with --assoc-ci-set wald only."
        )

    tbl = np.array([[a, b], [c, d]], dtype=float)
    t = Table2x2(tbl)

    try:
        or_ = float(t.oddsratio)  # may be inf if b*c==0
    except Exception:
        or_ = np.nan

    try:
        lo, hi = t.oddsratio_confint(alpha=alpha, method=method)
        lo, hi = float(lo), float(hi)
    except Exception:
        lo, hi = np.nan, np.nan

    return dict(OR=or_, OR_lo=lo, OR_hi=hi)

def phi_coefficient(a, b, c, d):
    n = a + b + c + d
    if n == 0:
        return np.nan
    num = a * d - b * c
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if den == 0:
        return np.nan
    return num / den

def chisq_and_residuals(a, b, c, d):
    table = np.array([[a, b], [c, d]], dtype=float)
    chi2, p, dof, expected = chi2_contingency(table, correction=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        resid = (table - expected) / np.sqrt(expected)
    return chi2, p, dof, expected, resid

def primary_p_value(a, b, c, d, expected, fisher_expected_threshold=5.0):
    """
    Define ONE primary p-value per stratum:
    - Fisher's exact (two-sided) if any expected cell < threshold
    - else Pearson chi-square (no continuity correction)
    """
    if np.any(expected < fisher_expected_threshold):
        try:
            p = fisher_exact([[a, b], [c, d]], alternative="two-sided")[1]
            return p, "Fisher"
        except Exception:
            return np.nan, "Fisher"
    else:
        try:
            p = chi2_contingency(np.array([[a, b], [c, d]], dtype=float), correction=False)[1]
            return p, "Chi-square"
        except Exception:
            return np.nan, "Chi-square"

def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """
    Standard BH-FDR adjustment. Returns adjusted p-values in original order.
    NaNs are left as NaN.
    """
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)

    mask = np.isfinite(p)
    if mask.sum() == 0:
        return out

    pv = p[mask]
    m = pv.size
    order = np.argsort(pv)
    pv_sorted = pv[order]

    ranks = np.arange(1, m + 1, dtype=float)
    adj_sorted = pv_sorted * m / ranks

    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0, 1)

    adj = np.empty_like(pv)
    adj[order] = adj_sorted
    out[mask] = adj
    return out


def fixed_and_random_meta(logOR, se):
    """Inverse-variance fixed effect and DerSimonian–Laird random effects for log(OR)."""
    w_fixed = 1 / (se**2)
    logOR_FE = np.sum(w_fixed * logOR) / np.sum(w_fixed)
    se_FE = math.sqrt(1 / np.sum(w_fixed))

    Q = np.sum(w_fixed * (logOR - logOR_FE) ** 2)
    df = len(logOR) - 1
    C = np.sum(w_fixed) - (np.sum(w_fixed**2) / np.sum(w_fixed))
    tau2 = max(0.0, (Q - df) / C) if df > 0 and C > 0 else 0.0
    I2 = max(0.0, (Q - df) / Q) if Q > 0 and df > 0 else 0.0

    w_RE = 1 / (se**2 + tau2)
    logOR_RE = np.sum(w_RE * logOR) / np.sum(w_RE)
    se_RE = math.sqrt(1 / np.sum(w_RE))

    return dict(
        logOR_FE=logOR_FE,
        se_FE=se_FE,
        logOR_RE=logOR_RE,
        se_RE=se_RE,
        Q=Q,
        df=df,
        I2=I2,
        tau2=tau2,
    )

def clopper_pearson_ci(k: int, n: int, alpha=0.05):
    """Exact binomial proportion CI via Beta quantiles."""
    if n <= 0 or k < 0 or k > n:
        return (np.nan, np.nan)

    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi

def odds_from_p(p):
    if p <= 0:
        return 0.0
    if p >= 1:
        return np.inf
    return p / (1 - p)

def dominance_or_and_ci(sd: int, idom: int, alpha=0.05):
    """
    Odds of smithii dominance among co-occurrences:
      odds = sd / idom (with 0.5 correction if sd==0 or idom==0)

    CI for plotting:
      Prefer Clopper–Pearson CI on p=sd/(sd+idom), transformed to odds,
      when finite; otherwise fall back to Wald log-odds w/ 0.5 correction.
    """
    n = sd + idom
    if n <= 0:
        return dict(
            cooccur_n=0,
            p_smithii=np.nan,
            p_smithii_lo=np.nan,
            p_smithii_hi=np.nan,
            OR_dom=np.nan,
            OR_dom_lo=np.nan,
            OR_dom_hi=np.nan,
            method="NA",
            corrected=False,
        )

    p = sd / n
    p_lo, p_hi = clopper_pearson_ci(sd, n, alpha=alpha)

    corrected = False
    sdc, idc = float(sd), float(idom)
    if sd == 0 or idom == 0:
        sdc += 0.5
        idc += 0.5
        corrected = True

    OR_dom = sdc / idc
    se = math.sqrt(1 / sdc + 1 / idc)
    z = norm.ppf(1 - alpha / 2)
    OR_lo_w = math.exp(math.log(OR_dom) - z * se)
    OR_hi_w = math.exp(math.log(OR_dom) + z * se)

    odds_lo = odds_from_p(p_lo)
    odds_hi = odds_from_p(p_hi)
    if np.isfinite(odds_lo) and np.isfinite(odds_hi) and odds_lo > 0 and odds_hi > 0:
        OR_dom_lo_plot = odds_lo
        OR_dom_hi_plot = odds_hi
        method = "Exact(p)->odds (Clopper–Pearson)"
    else:
        OR_dom_lo_plot = OR_lo_w
        OR_dom_hi_plot = OR_hi_w
        method = "Wald(log odds) w/ 0.5 corr if needed"

    return dict(
        cooccur_n=n,
        p_smithii=p,
        p_smithii_lo=p_lo,
        p_smithii_hi=p_hi,
        OR_dom=OR_dom,
        OR_dom_lo=OR_dom_lo_plot,
        OR_dom_hi=OR_dom_hi_plot,
        method=method,
        corrected=corrected,
    )

def _order_countries(sdf: pd.DataFrame) -> pd.DataFrame:
    sdf = sdf.copy()
    sdf["country"] = pd.Categorical(sdf["country"], categories=COUNTRY_ORDER, ordered=True)
    in_order = sdf["country"].notna()
    sdf = pd.concat(
        [
            sdf[in_order].sort_values("country"),
            sdf[~in_order].sort_values("country", key=lambda s: s.astype(str)),
        ],
        ignore_index=True,
    )
    return sdf

def symmetric_log_axis_from_cis(los: np.ndarray, his: np.ndarray, max_decades: int = 12):
    """
    Compute symmetric log10 x-limits and decade ticks from CI bounds.
    Ensures OR < 1 is never clipped.

    Returns (xmin, xmax, ticks).
    """
    los = np.asarray(los, dtype=float)
    his = np.asarray(his, dtype=float)

    vals = np.concatenate([los, his])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        decade = 2
        xmin, xmax = 10 ** (-decade), 10**decade
        ticks = [10**k for k in range(-decade, decade + 1)]
        return xmin, xmax, ticks

    logabs = np.max(np.abs(np.log10(vals)))
    decade = int(math.ceil(logabs))
    decade = max(1, min(decade, max_decades))

    xmin, xmax = 10 ** (-decade), 10**decade
    ticks = [10**k for k in range(-decade, decade + 1)]
    return xmin, xmax, ticks

def analyze(
    df: pd.DataFrame,
    outdir: Path,
    *,
    fisher_expected_threshold: float = 5.0,
    allow_N_mismatch: bool = False,
    allow_dominance_mismatch: bool = False,
    save_stats_table: bool = True,
    assoc_ci_set=("wald", "exact", "profile"),
):
    _require_columns(
        df,
        [
            "country",
            "age_group",
            "a_both",
            "b_intestini_only",
            "c_smithii_only",
            "d_neither",
            "N",
            "smithii_dom",
            "intestini_dom",
        ],
    )

    assoc_ci_set = tuple(assoc_ci_set)

    if any(m in ("exact", "profile") for m in assoc_ci_set) and not _HAVE_STATSMODELS:
        raise RuntimeError(
            "You requested exact/profile association CIs, but statsmodels is not available. "
            "Install statsmodels or run with --assoc-ci-set wald."
        )

    for col in ["a_both", "b_intestini_only", "c_smithii_only", "d_neither", "smithii_dom", "intestini_dom"]:
        df[col] = coerce_nonneg_int_series(df[col], colname=col, allow_missing=False)

    df["N"] = coerce_nonneg_int_series(df["N"], colname="N", allow_missing=True)

    row_total = (df["a_both"] + df["b_intestini_only"] + df["c_smithii_only"] + df["d_neither"]).astype("Int64")

    missing_N = df["N"].isna()
    df.loc[missing_N, "N"] = row_total[missing_N]

    mismatch = (df["N"] != row_total) & df["N"].notna() & row_total.notna()
    if mismatch.any():
        bad_rows = df.index[mismatch].tolist()[:10]
        msg = (
            f"N mismatch found in {mismatch.sum()} row(s). "
            f"Example row indices (up to 10): {bad_rows}. "
            f"Expected N=a+b+c+d."
        )
        if allow_N_mismatch:
            warnings.warn(msg + " Proceeding due to --allow-N-mismatch.", RuntimeWarning)
        else:
            raise ValueError(msg + " Fix input or pass --allow-N-mismatch.")

    a_both = df["a_both"].astype(int)
    dom_sum = (df["smithii_dom"] + df["intestini_dom"]).astype(int)

    dom_bad = ((a_both == 0) & (dom_sum != 0)) | ((a_both > 0) & (dom_sum != a_both))
    if dom_bad.any():
        bad_rows = df.index[dom_bad].tolist()[:10]
        msg = (
            f"Dominance mismatch found in {dom_bad.sum()} row(s). "
            f"Example row indices (up to 10): {bad_rows}. "
            f"Requirement: smithii_dom + intestini_dom == a_both (or both 0 when a_both==0)."
        )
        if allow_dominance_mismatch:
            warnings.warn(msg + " Proceeding due to --allow-dominance-mismatch.", RuntimeWarning)
        else:
            raise ValueError(msg + " Fix input or pass --allow-dominance-mismatch.")

    df_int = df.copy()
    for col in ["a_both", "b_intestini_only", "c_smithii_only", "d_neither", "N", "smithii_dom", "intestini_dom"]:
        df_int[col] = df_int[col].astype(int)

    stats_rows = []
    for _, r in df_int.iterrows():
        a, b, c, d = int(r.a_both), int(r.b_intestini_only), int(r.c_smithii_only), int(r.d_neither)
        n = a + b + c + d
        if n == 0:
            continue

        chi2, p_chi, dof, expected, resid = chisq_and_residuals(a, b, c, d)
        phi = phi_coefficient(a, b, c, d)

        p_primary, p_method = primary_p_value(a, b, c, d, expected, fisher_expected_threshold=fisher_expected_threshold)

        try:
            fisher_p = fisher_exact([[a, b], [c, d]], alternative="two-sided")[1]
        except Exception:
            fisher_p = np.nan

        OR_w, lo_w, hi_w, se_w, corrected_w = or_and_ci_2x2_wald(a, b, c, d)

        ex = dict(OR=np.nan, OR_lo=np.nan, OR_hi=np.nan)
        pr = dict(OR=np.nan, OR_lo=np.nan, OR_hi=np.nan)

        if "exact" in assoc_ci_set:
            try:
                ex = or_ci_statsmodels(a, b, c, d, alpha=0.05, method="exact")
            except Exception:
                ex = dict(OR=np.nan, OR_lo=np.nan, OR_hi=np.nan)

        if "profile" in assoc_ci_set:
            try:
                pr = or_ci_statsmodels(a, b, c, d, alpha=0.05, method="profile")
            except Exception:
                pr = dict(OR=np.nan, OR_lo=np.nan, OR_hi=np.nan)

        co_rate = a / n if n > 0 else np.nan

        sd, idom = int(r.smithii_dom), int(r.intestini_dom)
        dom = dominance_or_and_ci(sd, idom, alpha=0.05)

        stats_rows.append(
            dict(
                country=r.country,
                age_group=r.age_group,
                N=n,
                a=a,
                b=b,
                c=c,
                d=d,
                chi2=chi2,
                p_chi=p_chi,
                fisher_p=fisher_p,
                p_primary=p_primary,
                p_method=p_method,
                phi=phi,
                resid_00=float(resid[0, 0]) if np.isfinite(resid[0, 0]) else np.nan,
                resid_01=float(resid[0, 1]) if np.isfinite(resid[0, 1]) else np.nan,
                resid_10=float(resid[1, 0]) if np.isfinite(resid[1, 0]) else np.nan,
                resid_11=float(resid[1, 1]) if np.isfinite(resid[1, 1]) else np.nan,
                OR_wald=OR_w,
                OR_wald_lo=lo_w,
                OR_wald_hi=hi_w,
                logOR_wald=float(np.log(OR_w)) if OR_w > 0 else np.nan,
                se_logOR_wald=se_w,
                OR_wald_corrected=corrected_w,
                OR_exact=ex["OR"],
                OR_exact_lo=ex["OR_lo"],
                OR_exact_hi=ex["OR_hi"],
                OR_profile=pr["OR"],
                OR_profile_lo=pr["OR_lo"],
                OR_profile_hi=pr["OR_hi"],
                cooccurrence_rate=co_rate,
                smithii_dom=sd,
                intestini_dom=idom,
                cooccur_n=dom["cooccur_n"],
                p_smithii=dom["p_smithii"],
                p_smithii_lo=dom["p_smithii_lo"],
                p_smithii_hi=dom["p_smithii_hi"],
                OR_dom=dom["OR_dom"],
                OR_dom_lo=dom["OR_dom_lo"],
                OR_dom_hi=dom["OR_dom_hi"],
                OR_dom_method=dom["method"],
                OR_dom_corrected=dom["corrected"],
            )
        )

    stats = pd.DataFrame(stats_rows)

    stats_adult = stats[stats["age_group"] == "Adult"].copy()

    if len(stats_adult) > 0:
        stats_adult["p_BH"] = benjamini_hochberg(stats_adult["p_primary"].values)
    else:
        stats_adult["p_BH"] = np.array([])

    if save_stats_table and len(stats_adult) > 0:
        out_stats = outdir / "02_stats_Adult.tsv"
        stats_adult.to_csv(out_stats, sep="\t", index=False)
        print(f"Saved: {out_stats}")

    def _prep_forest_df(sdf, or_col, lo_col, hi_col):
        sdf = sdf.copy()
        sdf = sdf.replace([np.inf, -np.inf], np.nan)
        sdf = sdf.dropna(subset=[or_col, lo_col, hi_col])
        sdf = sdf[(sdf[or_col] > 0) & (sdf[lo_col] > 0) & (sdf[hi_col] > 0)]
        return sdf

    assoc_bounds_lo = []
    assoc_bounds_hi = []
    for m in assoc_ci_set:
        if m == "wald":
            lo_c, hi_c = "OR_wald_lo", "OR_wald_hi"
        elif m == "exact":
            lo_c, hi_c = "OR_exact_lo", "OR_exact_hi"
        elif m == "profile":
            lo_c, hi_c = "OR_profile_lo", "OR_profile_hi"
        else:
            continue
        tmp = stats_adult.replace([np.inf, -np.inf], np.nan).dropna(subset=[lo_c, hi_c])
        tmp = tmp[(tmp[lo_c] > 0) & (tmp[hi_c] > 0)]
        assoc_bounds_lo.append(tmp[lo_c].values)
        assoc_bounds_hi.append(tmp[hi_c].values)

    if assoc_bounds_lo:
        assoc_bounds_lo = np.concatenate(assoc_bounds_lo) if len(assoc_bounds_lo) > 1 else assoc_bounds_lo[0]
        assoc_bounds_hi = np.concatenate(assoc_bounds_hi) if len(assoc_bounds_hi) > 1 else assoc_bounds_hi[0]
        ASSOC_XMIN, ASSOC_XMAX, ASSOC_XTICKS = symmetric_log_axis_from_cis(assoc_bounds_lo, assoc_bounds_hi)
    else:
        ASSOC_XMIN, ASSOC_XMAX, ASSOC_XTICKS = symmetric_log_axis_from_cis(np.array([0.1]), np.array([10.0]))

    tmpd = stats_adult.replace([np.inf, -np.inf], np.nan).dropna(subset=["OR_dom_lo", "OR_dom_hi"])
    tmpd = tmpd[(tmpd["OR_dom_lo"] > 0) & (tmpd["OR_dom_hi"] > 0)]
    DOM_XMIN, DOM_XMAX, DOM_XTICKS = symmetric_log_axis_from_cis(tmpd["OR_dom_lo"].values, tmpd["OR_dom_hi"].values)

    def forest_assoc(sdf: pd.DataFrame, outbase: Path, *, or_col: str, lo_col: str, hi_col: str, title_suffix: str, add_pooled: bool):
        sdf = _prep_forest_df(sdf, or_col, lo_col, hi_col)
        if len(sdf) == 0:
            return

        sdf = _order_countries(sdf)
        labels = [f"{r.country} (N={int(r.N)})" for _, r in sdf.iterrows()]
        y = np.arange(len(sdf))

        fig = plt.figure(figsize=FOREST_FIGSIZE)
        ax = fig.add_axes(FOREST_AXES_RECT)

        ax.errorbar(
            sdf[or_col],
            y,
            xerr=[sdf[or_col] - sdf[lo_col], sdf[hi_col] - sdf[or_col]],
            fmt="o",
            capsize=3,
        )
        ax.axvline(1, linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlim(ASSOC_XMIN, ASSOC_XMAX)
        ax.set_xticks(ASSOC_XTICKS)

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Odds ratio (log scale)")
        ax.set_title(f"Association of presence (intestini ↔ smithii): Adult\n{title_suffix}")

        if add_pooled:
            sub = sdf.dropna(subset=["logOR_wald", "se_logOR_wald"])
            if len(sub) >= 2:
                res = fixed_and_random_meta(sub["logOR_wald"].values, sub["se_logOR_wald"].values)
                z = norm.ppf(0.975)

                for k, (L, S, ttl) in enumerate(
                    [
                        (res["logOR_FE"], res["se_FE"], "Pooled (FE)"),
                        (res["logOR_RE"], res["se_RE"], f"Pooled (RE)  I²={res['I2']*100:.0f}%, τ²={res['tau2']:.3g}"),
                    ]
                ):
                    ORp = math.exp(L)
                    lo = math.exp(L - z * S)
                    hi = math.exp(L + z * S)
                    yy = -1.2 - 0.4 * k
                    ax.errorbar([ORp], [yy], xerr=[[max(ORp - lo, 1e-12)], [max(hi - ORp, 1e-12)]], fmt="D", ms=6)
                    ax.text(ASSOC_XMIN * 1.02, yy, ttl, va="center", ha="left")

                ax.set_ylim(-2.0, len(sdf) - 0.5)

        savefig_both(fig, outbase)
        plt.close(fig)

    if "wald" in assoc_ci_set:
        forest_assoc(
            stats_adult,
            outdir / "03_forest_assoc_Adult_wald",
            or_col="OR_wald",
            lo_col="OR_wald_lo",
            hi_col="OR_wald_hi",
            title_suffix="CI: Wald log(OR) (Haldane–Anscombe if needed)",
            add_pooled=True,
        )

        forest_assoc(
            stats_adult,
            outdir / "03_forest_assoc_Adult",
            or_col="OR_wald",
            lo_col="OR_wald_lo",
            hi_col="OR_wald_hi",
            title_suffix="CI: Wald log(OR) (Haldane–Anscombe if needed)",
            add_pooled=True,
        )

    if "exact" in assoc_ci_set:
        forest_assoc(
            stats_adult,
            outdir / "04_forest_assoc_Adult_exact",
            or_col="OR_exact",
            lo_col="OR_exact_lo",
            hi_col="OR_exact_hi",
            title_suffix="CI: Exact conditional OR (Fisher inversion; statsmodels)",
            add_pooled=False,
        )

    if "profile" in assoc_ci_set:
        forest_assoc(
            stats_adult,
            outdir / "04_forest_assoc_Adult_profile",
            or_col="OR_profile",
            lo_col="OR_profile_lo",
            hi_col="OR_profile_hi",
            title_suffix="CI: Profile likelihood OR (statsmodels)",
            add_pooled=False,
        )

    def forest_dom(sdf: pd.DataFrame, outbase: Path):
        sdf = sdf.replace([np.inf, -np.inf], np.nan).dropna(subset=["OR_dom", "OR_dom_lo", "OR_dom_hi"])
        sdf = sdf[(sdf["cooccur_n"] > 0) & (sdf["OR_dom"] > 0) & (sdf["OR_dom_lo"] > 0) & (sdf["OR_dom_hi"] > 0)]
        if len(sdf) == 0:
            return

        sdf = _order_countries(sdf)
        labels = [f"{r.country} (co-occur N={int(r.cooccur_n)})" for _, r in sdf.iterrows()]
        y = np.arange(len(sdf))

        fig = plt.figure(figsize=FOREST_FIGSIZE)
        ax = fig.add_axes(FOREST_AXES_RECT)

        ax.errorbar(
            sdf["OR_dom"],
            y,
            xerr=[sdf["OR_dom"] - sdf["OR_dom_lo"], sdf["OR_dom_hi"] - sdf["OR_dom"]],
            fmt="o",
            capsize=3,
        )
        ax.axvline(1, linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlim(DOM_XMIN, DOM_XMAX)
        ax.set_xticks(DOM_XTICKS)

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Odds of smithii dominance (log scale)")
        ax.set_title("Dominance odds (smithii vs intestini): Adult")

        savefig_both(fig, outbase)
        plt.close(fig)

    forest_dom(stats_adult, outdir / "05_forest_dominance_Adult")

    print("All done.")

def _parse_assoc_ci_set(s: str):
    """
    Parse comma-separated list of CI methods for association.
    Allowed: wald, exact, profile
    """
    allowed = {"wald", "exact", "profile"}
    items = [x.strip().lower() for x in s.split(",") if x.strip()]
    bad = [x for x in items if x not in allowed]
    if bad:
        raise argparse.ArgumentTypeError(f"Invalid --assoc-ci-set entries: {bad}. Allowed: wald,exact,profile")
    if not items:
        raise argparse.ArgumentTypeError("Empty --assoc-ci-set. Allowed: wald,exact,profile")
    return tuple(items)

def main():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument("--input", required=True, help="")
    parser.add_argument(
        "--sep",
        default="\t",
        help="",
    )
    parser.add_argument(
        "--outdir",
        default="",
        help="",
    )
    parser.add_argument(
        "--fisher-expected-threshold",
        type=float,
        default=5.0,
        help="",
    )
    parser.add_argument(
        "--allow-N-mismatch",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--allow-dominance-mismatch",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--no-save-stats-table",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--assoc-ci-set",
        type=_parse_assoc_ci_set,
        default=("wald", "exact", "profile"),
        help="",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df = pd.read_csv(args.input, sep=args.sep)

    analyze(
        df,
        outdir,
        fisher_expected_threshold=args.fisher_expected_threshold,
        allow_N_mismatch=args.allow_N_mismatch,
        allow_dominance_mismatch=args.allow_dominance_mismatch,
        save_stats_table=(not args.no_save_stats_table),
        assoc_ci_set=args.assoc_ci_set,
    )

if __name__ == "__main__":
    main()