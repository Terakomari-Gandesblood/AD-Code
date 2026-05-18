import os
import pandas as pd
import numpy as np

INPUT_PATH = r"C:\Users\Chisato\Desktop\毕设\Offline\总.xlsx"

ALGO_ORDER = [
    "rule_pid",
    "pure_ppo",
    "sac",
    "bc",
    "hier_ppo",
    "gnn_flat_ppo",
    "gnn_hier_ppo",
]
ALGO_DISPLAY = {
    "rule_pid": "PID",
    "pure_ppo": "PPO",
    "sac": "SAC",
    "bc": "BC",
    "hier_ppo": "HP",
    "gnn_flat_ppo": "GFP",
    "gnn_hier_ppo": "GHP",
}

SCEN_ORDER = [
    "ego",
    "gnss",
    "lane_has_next",
    "lane_yaw",
    "tl_dist",
    "veh_fake",
    "veh_hide",
    "veh_noise",
]
SCEN_DISPLAY = {
    "ego": "自车",
    "gnss": "GNSS",
    "lane_has_next": "路点",
    "lane_yaw": "转向",
    "tl_dist": "灯距",
    "veh_fake": "欺骗",
    "veh_hide": "漏检",
    "veh_noise": "噪声",
}

LEVEL_ORDER = ["L1", "L2", "L3"]

METRICS = [
    ("mean_action_l2", "离线动作误差（L2范数）退化倍数对比（top-50 scenes）", "tab:offline_action_l2_ratio"),
    ("mean_action_l2_on_ttc_risk", "风险子集动作误差退化倍数对比（TTC-risk subset, top-50 scenes）", "tab:offline_action_l2_risk_ratio"),
]

DECIMALS = 3


def _try_read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    # csv: try utf-8-sig -> utf-8 -> gbk
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    # last resort
    return pd.read_csv(path, encoding="latin1")


def _norm_str(x):
    if isinstance(x, str):
        return x.strip()
    return x


def _pick_col(cols, candidates):
    cols_l = {c.lower(): c for c in cols}
    for cand in candidates:
        c = cols_l.get(cand.lower())
        if c is not None:
            return c
    return None


def load_and_clean(path: str) -> pd.DataFrame:
    df = _try_read_table(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.applymap(_norm_str)

    col_algo = _pick_col(df.columns, ["算法", "algo", "algorithm", "method"])
    col_scen = _pick_col(df.columns, ["scenario", "场景", "攻击", "attack_scenario"])
    col_lvl  = _pick_col(df.columns, ["强度", "level", "intensity"])
    col_pairs = _pick_col(df.columns, ["pairs", "pair", "样本", "n_pairs"])

    if col_algo is None or col_scen is None:
        raise RuntimeError(f"Cannot find required columns. Got columns={list(df.columns)}")

    rename_map = {col_algo: "algo", col_scen: "scenario"}
    if col_lvl is not None:
        rename_map[col_lvl] = "level"
    if col_pairs is not None:
        rename_map[col_pairs] = "pairs"
    df = df.rename(columns=rename_map)

    if "level" not in df.columns:
        df["level"] = ""

    df["algo"] = df["algo"].astype(str).str.strip()
    df["scenario"] = df["scenario"].astype(str).str.strip()
    df["level"] = df["level"].astype(str).str.strip()

    df = df[(df["algo"] != "") & (df["scenario"] != "")]

    return df


def build_pivots(df: pd.DataFrame, metric: str) -> dict:
    if metric not in df.columns:
        raise RuntimeError(f"Metric column '{metric}' not found. Available={list(df.columns)}")

    df2 = df.copy()
    df2["algo"] = df2["algo"].astype(str).str.strip()
    df2["scenario"] = df2["scenario"].astype(str).str.strip()
    df2["level"] = df2["level"].astype(str).str.strip()

    df2 = df2[df2["algo"].isin(ALGO_DISPLAY.keys())]
    df2 = df2[df2["scenario"].str.lower() != "clean"]
    df2 = df2[df2["level"].isin(LEVEL_ORDER)]

    scen_present = [s for s in SCEN_ORDER if s in set(df2["scenario"])]
    scen_extra = sorted(list(set(df2["scenario"]) - set(SCEN_ORDER)))
    scen_final = scen_present + scen_extra

    pivots = {}
    for lvl in LEVEL_ORDER:
        sub = df2[df2["level"] == lvl]
        pv = sub.pivot_table(
            index="algo", columns="scenario", values=metric, aggfunc="mean"
        )
        pv = pv.reindex(index=ALGO_ORDER)
        pv = pv.reindex(columns=scen_final)
        pv["Mean"] = pv.mean(axis=1, skipna=True)
        pivots[lvl] = pv

    return pivots


def fmt_val(x) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "--"
    try:
        return f"{float(x):.{DECIMALS}f}"
    except Exception:
        return "--"


def latex_table(pivots: dict, metric: str, caption: str, label: str) -> str:
    any_lvl = LEVEL_ORDER[0]
    cols = list(pivots[any_lvl].columns)
    scen_cols = [c for c in cols if c != "Mean"]
    scen_headers = [SCEN_DISPLAY.get(s, s) for s in scen_cols]

    # tabularx spec: 强度(c) 方法(l) + scenarios + Mean
    n_scen = len(scen_cols)
    col_spec = "c l " + f"*{{{n_scen}}}{{X}} X"

    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(rf"\begin{{tabularx}}{{\textwidth}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header = "强度 & 方法 & " + " & ".join(scen_headers) + r" & Mean \\"
    lines.append(header)
    lines.append(r"\midrule")

    for li, lvl in enumerate(LEVEL_ORDER):
        pv = pivots[lvl]

        # multirow block
        k = len(ALGO_ORDER)
        first = True
        for algo in ALGO_ORDER:
            disp = ALGO_DISPLAY.get(algo, algo)
            row_vals = []
            for s in scen_cols:
                row_vals.append(fmt_val(pv.loc[algo, s]) if algo in pv.index else "--")
            row_vals.append(fmt_val(pv.loc[algo, "Mean"]) if algo in pv.index else "--")

            if first:
                lines.append(rf"\multirow{{{k}}}{{*}}{{{lvl}}} & {disp} & " + " & ".join(row_vals) + r" \\")
                first = False
            else:
                lines.append(rf"& {disp} & " + " & ".join(row_vals) + r" \\")

        if li != len(LEVEL_ORDER) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    df = load_and_clean(INPUT_PATH)

    print("=== Loaded rows:", len(df), "===")
    print("Columns:", list(df.columns))
    print()

    for metric, caption, label in METRICS:
        pivots = build_pivots(df, metric=metric)
        tex = latex_table(pivots, metric=metric, caption=caption, label=label)

        print("\n" + "=" * 90)
        print(f"LATEX TABLE for {metric}")
        print("=" * 90)
        print(tex)
        print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
