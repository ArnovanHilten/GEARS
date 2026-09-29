"""
GEARS baseline training script for the Replogle K562 dataset.

Uses the same STATE toml split as the scDFM / CellFlow benchmarks so that
train/val/test conditions are identical across all baselines.

Pipeline
--------
1. Load h5ad; normalize/log1p if not preprocessed
2. Select HVGs; HVG subset becomes GEARS gene space
3. Remap control label ("non-targeting" → "ctrl")
   and reformat single-gene conditions ("GENE" → "GENE+ctrl")
4. Parse STATE toml to obtain train/val/test gene lists
5. Write GEARS custom split dict (pkl)
6. Run PertData.new_data_process + prepare_split + get_dataloader
7. Train GEARS
8. Predict on test conditions; save predicted AnnData
9. Evaluate with cell-eval MetricsEvaluator (in a subprocess to avoid
   fork/deadlock issues with any JAX-backed C extension)
10. Save agg_results.csv
"""

import argparse
import os
import pickle
import subprocess
import sys
import tempfile
import textwrap

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import wandb


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True,
                   help="Directory containing <data_name>.h5ad")
    p.add_argument("--data_name", default="emb_Replogle",
                   help="Basename of the h5ad file (without extension)")
    p.add_argument("--condition_col", default="gene",
                   help="obs column holding perturbation gene names")
    p.add_argument("--control_value", default="non-targeting",
                   help="Value in condition_col that marks control cells")
    p.add_argument("--cell_type_col", default="cell_line",
                   help="obs column holding cell type labels")
    p.add_argument("--preprocessed", action="store_true",
                   help="Data is already log1p-normalised; skip normalisation")
    p.add_argument("--split_toml", required=True,
                   help="Path to STATE toml defining train/val/test gene sets")
    p.add_argument("--result_path", required=True,
                   help="Output directory for saved model and results")
    p.add_argument("--run_id", default="",
                   help="Human-readable run name prefix")
    p.add_argument("--gears_data_path", default="/workspace/gears_data",
                   help="Directory containing pre-downloaded GEARS reference files "
                        "(gene2go_all.pkl, essential_all_data_pert_genes.pkl)")
    p.add_argument("--n_top_genes", type=int, default=2000,
                   help="Number of highly variable genes to select")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_go_gnn_layers", type=int, default=1)
    p.add_argument("--num_gene_gnn_layers", type=int, default=1)
    p.add_argument("--num_similar_genes_go_graph", type=int, default=20)
    p.add_argument("--num_similar_genes_co_express_graph", type=int, default=20)
    p.add_argument("--coexpress_threshold", type=float, default=0.4)
    p.add_argument("--wandb_project", default="")
    p.add_argument("--wandb_entity", default="")
    p.add_argument("--wandb_tags", default="gears,replogle")
    return p.parse_args()


# ── toml parsing ─────────────────────────────────────────────────────────────

def load_toml_split(toml_path):
    """Return (val_genes, test_genes) as sets of gene-name strings."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # Python <3.11 fallback

    with open(toml_path, "rb") as f:
        cfg = tomllib.load(f)

    val_genes, test_genes = set(), set()
    for block in cfg.get("fewshot", {}).values():
        val_genes.update(block.get("val", []))
        test_genes.update(block.get("test", []))
    return val_genes, test_genes


# ── data preparation ─────────────────────────────────────────────────────────

def prepare_adata(adata, condition_col, control_value, cell_type_col,
                  preprocessed, n_top_genes):
    """
    Return a copy of adata formatted for GEARS:
      - .X  log1p-normalised counts for selected HVGs
      - .obs["condition"]  GEARS-format labels ("GENE+ctrl" or "ctrl")
      - .var["gene_name"]  set from var.index
      - .obs["cell_type"]  copied from cell_type_col
    """
    adata = adata.copy()

    # Normalise if needed
    if not preprocessed:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # HVG selection; force all perturbation gene names into the HVG set
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes,
                                subset=False, flavor="seurat")
    pert_genes = set(
        g for g in adata.obs[condition_col].unique()
        if g != control_value and g in adata.var_names
    )
    force_hvg = adata.var_names.isin(pert_genes)
    adata.var["highly_variable"] = adata.var["highly_variable"] | force_hvg
    adata = adata[:, adata.var["highly_variable"]].copy()

    # gene_name column (required by GEARS; must match .var_names)
    adata.var["gene_name"] = adata.var_names.astype(str)

    # cell_type column (required by GEARS)
    if "cell_type" not in adata.obs.columns:
        if cell_type_col in adata.obs.columns:
            adata.obs["cell_type"] = adata.obs[cell_type_col].astype(str)
        else:
            adata.obs["cell_type"] = "unknown"

    # condition column in GEARS format: "ctrl" or "GENE+ctrl"
    raw_cond = adata.obs[condition_col].astype(str)
    def to_gears_cond(c):
        if c == control_value:
            return "ctrl"
        return f"{c}+ctrl"
    adata.obs["condition"] = raw_cond.map(to_gears_cond)

    print(f"  HVG-filtered adata: {adata.shape[0]} cells × {adata.shape[1]} genes")
    ctrl_n = (adata.obs["condition"] == "ctrl").sum()
    pert_n = adata.shape[0] - ctrl_n
    print(f"  ctrl cells: {ctrl_n}  perturbed cells: {pert_n}")

    return adata


def build_split_dict(adata, val_genes, test_genes, target_cell_line=None):
    """
    Map toml gene sets → GEARS condition strings and return
    {'train': [...], 'val': [...], 'test': [...]} including 'ctrl' in train.

    Val/test conditions are derived exclusively from target_cell_line cells
    (e.g. k562).  Cells from all other cell lines always go to train, even if
    they were perturbed with a val/test gene.
    """
    all_conds = set(adata.obs["condition"].unique()) - {"ctrl"}

    # Determine which conditions can be val/test
    if target_cell_line and "cell_line" in adata.obs.columns:
        tcl_obs = adata.obs[adata.obs["cell_line"] == target_cell_line]
        source_conds = set(tcl_obs["condition"].unique()) - {"ctrl"}
    else:
        source_conds = all_conds

    test_set = {c for c in source_conds if c.split("+")[0] in test_genes}
    val_set  = {c for c in source_conds if c.split("+")[0] in val_genes} - test_set

    val_conds   = sorted(val_set)
    test_conds  = sorted(test_set)
    # train = everything not held out (includes all non-target-cell-line conditions)
    train_conds = sorted(all_conds - val_set - test_set)
    train_conds = ["ctrl"] + train_conds

    print(f"  Split sizes ({target_cell_line or 'all'} as competition cell line) — "
          f"train: {len(train_conds)-1} perturbations ({len(train_conds)} with ctrl)  "
          f"val: {len(val_conds)}  test: {len(test_conds)}")
    return {"train": train_conds, "val": val_conds, "test": test_conds}


# ── evaluation (subprocess to avoid fork issues) ─────────────────────────────

EVAL_SCRIPT = textwrap.dedent("""\
import sys, json, os
import anndata, numpy as np

pred_path, true_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]

from cell_eval import MetricsEvaluator
pred_adata = anndata.read_h5ad(pred_path)
true_adata = anndata.read_h5ad(true_path)

evaluator = MetricsEvaluator(
    adata_pred=pred_adata, adata_real=true_adata,
    control_pert="ctrl", pert_col="condition",
    num_threads=32,
)
results, agg = evaluator.compute()
os.makedirs(out_dir, exist_ok=True)
results.write_csv(os.path.join(out_dir, "results.csv"))
agg.write_csv(os.path.join(out_dir, "agg_results.csv"))
agg_df = agg.to_pandas()
mean_row = agg_df[agg_df["statistic"] == "mean"].iloc[0].to_dict()
print(json.dumps({k: v for k, v in mean_row.items() if isinstance(v, float)}))
""")


def run_eval_subprocess(pred_adata, true_adata, out_dir):
    """Evaluate in a fresh subprocess to avoid any fork/JAX deadlock.

    Returns the mean-row metrics dict (parsed from subprocess stdout), or None on failure.
    """
    import json as _json
    with tempfile.TemporaryDirectory() as tmpdir:
        pred_path = os.path.join(tmpdir, "pred.h5ad")
        true_path = os.path.join(tmpdir, "true.h5ad")
        script_path = os.path.join(tmpdir, "eval.py")

        pred_adata.write_h5ad(pred_path)
        true_adata.write_h5ad(true_path)

        with open(script_path, "w") as f:
            f.write(EVAL_SCRIPT)

        result = subprocess.run(
            [sys.executable, script_path, pred_path, true_path, out_dir],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print("Evaluation stderr:\n", result.stderr)
            raise RuntimeError("Evaluation subprocess failed")
        print(result.stdout)
        try:
            return _json.loads(result.stdout.strip().splitlines()[-1])
        except Exception:
            return None


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── build run name / output path ─────────────────────────────────────────
    import hashlib, json
    key = {
        "data_name": args.data_name,
        "split_toml": os.path.basename(args.split_toml),
        "n_top_genes": args.n_top_genes,
        "epochs": args.epochs,
        "hidden_size": args.hidden_size,
    }
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
    run_name = f"{args.run_id}_gears_{h}" if args.run_id else f"gears_{args.data_name}_{h}"
    save_path = os.path.join(args.result_path, run_name)
    os.makedirs(save_path, exist_ok=True)
    print(f"\n=== GEARS run: {run_name} ===")
    print(f"    output: {save_path}")

    # ── W&B ──────────────────────────────────────────────────────────────────
    use_wandb = bool(args.wandb_project)
    if use_wandb:
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=run_name,
            tags=tags,
            config=vars(args),
        )

    # ── Step 1: load data ─────────────────────────────────────────────────────
    h5ad_path = os.path.join(args.data_path, f"{args.data_name}.h5ad")
    print(f"\n[1] Loading {h5ad_path}")
    adata_raw = sc.read_h5ad(h5ad_path)
    print(f"    shape: {adata_raw.shape}")

    # ── Step 2: prepare GEARS-format adata ───────────────────────────────────
    print(f"\n[2] Preparing GEARS-format AnnData (n_top_genes={args.n_top_genes})")
    adata = prepare_adata(
        adata_raw,
        condition_col=args.condition_col,
        control_value=args.control_value,
        cell_type_col=args.cell_type_col,
        preprocessed=args.preprocessed,
        n_top_genes=args.n_top_genes,
    )

    # ── Step 3: parse toml split ─────────────────────────────────────────────
    # The fewshot key (e.g. "replogle.k562") names the competition cell line.
    # Val/test conditions come only from that cell line; all other cell lines
    # (jurkat, rpe1, hepg2) contribute ALL their cells to training.
    print(f"\n[3] Parsing toml split: {args.split_toml}")
    val_genes, test_genes = load_toml_split(args.split_toml)
    print(f"    val genes in toml: {len(val_genes)}  "
          f"test genes in toml: {len(test_genes)}")

    # Extract the competition cell line from toml fewshot keys
    target_cell_line = None
    try:
        import tomllib as _tl
    except ImportError:
        import tomli as _tl
    with open(args.split_toml, "rb") as _f:
        _fewshot_keys = list(_tl.load(_f).get("fewshot", {}).keys())
    _lines_in_toml = {k.split(".")[-1] for k in _fewshot_keys}
    if "cell_line" in adata.obs.columns:
        _matched = _lines_in_toml & set(adata.obs["cell_line"].unique())
        if _matched:
            target_cell_line = sorted(_matched)[0]
            print(f"    Competition cell line: '{target_cell_line}' "
                  f"(other cell lines are all-train)")

    split_dict = build_split_dict(adata, val_genes, test_genes,
                                  target_cell_line=target_cell_line)

    split_pkl_path = os.path.join(save_path, "gears_split.pkl")
    with open(split_pkl_path, "wb") as f:
        pickle.dump(split_dict, f)
    print(f"    split dict saved: {split_pkl_path}")

    # ── Step 4: PertData — process + split ───────────────────────────────────
    print(f"\n[4] PertData.new_data_process (gears_data_path={args.gears_data_path})")
    from gears import PertData, GEARS
    import scipy.sparse as sp

    # GEARS assumes adata.X is always a scipy sparse matrix — every .toarray()
    # call in its source relies on this. Scanpy ops may have densified X.
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)

    pert_data = PertData(args.gears_data_path)
    pert_data.new_data_process(
        dataset_name=args.data_name.lower(),
        adata=adata,
        skip_calc_de=False,
    )
    pert_data.prepare_split(split="custom", split_dict_path=split_pkl_path)
    pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=args.batch_size)
    print("    Dataloaders ready.")

    # ── Step 5: model init ────────────────────────────────────────────────────
    print(f"\n[5] Initialising GEARS model")
    gears_model = GEARS(pert_data, device="cuda",
                        weight_bias_track=use_wandb,
                        proj_name=args.wandb_project,
                        exp_name=run_name)
    gears_model.model_initialize(
        hidden_size=args.hidden_size,
        num_go_gnn_layers=args.num_go_gnn_layers,
        num_gene_gnn_layers=args.num_gene_gnn_layers,
        num_similar_genes_go_graph=args.num_similar_genes_go_graph,
        num_similar_genes_co_express_graph=args.num_similar_genes_co_express_graph,
        coexpress_threshold=args.coexpress_threshold,
    )

    # ── Step 6: train ─────────────────────────────────────────────────────────
    print(f"\n[6] Training GEARS for {args.epochs} epochs")
    gears_model.train(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── Step 7: save model ────────────────────────────────────────────────────
    model_dir = os.path.join(save_path, "model")
    os.makedirs(model_dir, exist_ok=True)
    gears_model.save_model(model_dir)
    print(f"\n[7] Model saved to {model_dir}")

    # ── Step 8: predict on test conditions ────────────────────────────────────
    print(f"\n[8] Predicting test conditions")
    test_conds = split_dict["test"]

    # Filter to genes present in GEARS's perturbation graph (GO network coverage
    # may exclude some genes that are in the toml test set)
    gears_pert_set = set(gears_model.pert_list)
    skipped = [c for c in test_conds if c.split("+")[0] not in gears_pert_set]
    test_conds = [c for c in test_conds if c.split("+")[0] in gears_pert_set]
    if skipped:
        print(f"  Skipped {len(skipped)} test conditions not in GEARS pert graph: {skipped}")
    print(f"  Predicting {len(test_conds)} test conditions")

    # GEARS predict format: list of lists, each inner list is one condition's genes
    pert_list = [[c.split("+")[0]] for c in test_conds]
    pred_dict = gears_model.predict(pert_list)

    # Build predicted AnnData: one row per test condition (mean prediction)
    gene_names = pert_data.gene_names.values
    pred_X = np.stack([pred_dict[c.split("+")[0]] for c in test_conds], axis=0).astype(np.float32)
    pred_adata = anndata.AnnData(
        X=pred_X,
        obs=pd.DataFrame({"condition": test_conds},
                         index=[f"pred_{i}" for i in range(len(test_conds))]),
        var=pd.DataFrame(index=gene_names),
    )
    pred_adata.var_names_make_unique()

    pred_out_path = os.path.join(save_path, "final_test", "pred.h5ad")
    os.makedirs(os.path.dirname(pred_out_path), exist_ok=True)
    pred_adata.write_h5ad(pred_out_path)
    print(f"    Predictions saved: {pred_out_path}  "
          f"(shape {pred_adata.shape})")

    # ── Step 9: ground-truth adata for test conditions ────────────────────────
    test_conds_set = set(test_conds)
    true_mask = adata.obs["condition"].isin(test_conds_set)
    true_adata = adata[true_mask].copy()
    # restrict to same gene columns as pred_adata
    common_genes = pred_adata.var_names.intersection(true_adata.var_names)
    true_adata = true_adata[:, common_genes].copy()
    pred_adata = pred_adata[:, common_genes].copy()

    # cell_eval needs control cells to compute pearson_delta (pert - ctrl baseline).
    # Append actual control cells to true_adata and mean control expression to pred_adata.
    ctrl_true = adata[adata.obs["condition"] == "ctrl", common_genes].copy()
    ctrl_X = np.asarray(
        ctrl_true.X.toarray() if hasattr(ctrl_true.X, "toarray") else ctrl_true.X,
        dtype=np.float32,
    )
    mean_ctrl = ctrl_X.mean(axis=0, keepdims=True)
    ctrl_pred = anndata.AnnData(
        X=mean_ctrl,
        obs=pd.DataFrame({"condition": ["ctrl"]}, index=["ctrl_pred"]),
        var=pred_adata.var,
    )
    pred_adata = anndata.concat([ctrl_pred, pred_adata], join="outer")
    true_adata = anndata.concat([ctrl_true, true_adata], join="outer")
    # GEARS NN output is unbounded; clip negatives so cell_eval doesn't reject it
    import scipy.sparse as _sp
    _X = pred_adata.X.toarray() if _sp.issparse(pred_adata.X) else np.asarray(pred_adata.X)
    pred_adata.X = np.clip(_X, 0, None)
    print(f"  pred_adata: {pred_adata.shape}  true_adata: {true_adata.shape}")

    # ── Step 10: cell-eval ────────────────────────────────────────────────────
    print(f"\n[9] Running cell-eval in subprocess")
    eval_dir = os.path.join(save_path, "final_test")
    metrics_json = run_eval_subprocess(pred_adata, true_adata, eval_dir)

    print(f"\n=== Done. Results in {eval_dir}/ ===")

    if use_wandb:
        if metrics_json:
            wandb.run.summary.update(metrics_json)
        wandb.run.summary["eval_dir"] = eval_dir
        wandb.finish()


if __name__ == "__main__":
    main()
