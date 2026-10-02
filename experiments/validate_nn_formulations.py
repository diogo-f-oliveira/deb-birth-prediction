"""Small T05 engineering runs, not tuning or a test-set benchmark.

Run from the repository root: conda run -n debbirth python -m experiments.validate_nn_formulations
Only supplied train/val CSVs are read. All outputs go to a distinct ignored run.
"""
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.debbirth.data.prepare import prepare_split
from src.debbirth.data.schema import DatasetSpec, FULL_PAR_COLS
from src.debbirth.evaluate.predict import evaluate_pytorch_binary_classifier, metrics_from_predictions
from src.debbirth.formulations import output_to_logit
from src.debbirth.models.nn.config import TrainDEBBirthNetConfig, DEBBirthNetConfig
from src.debbirth.models.nn.predict import NNPredictor
from src.debbirth.models.nn.structure import build_net
from src.debbirth.models.nn.train import train_net, load_trained_nn
from src.debbirth.utils.paths import REPO_ROOT
from src.debbirth.utils.results import create_run_outdir


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def assert_raises(function, error=ValueError):
    try:
        function()
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}")


def verify_run(result, physical_val):
    cfg = result["train_config"]
    val = result["prepared"]["val"]
    predictor = result["predictor"]
    details = predictor.predict_details(physical_val)
    for key, value in details.items():
        assert value.shape == (len(val.labels),) and np.isfinite(value).all()
        np.testing.assert_array_equal(value, predictor.predict_prepared(val)[key])
    loaded = load_trained_nn(result["outdir"], "cpu")
    assert not loaded["model"].training
    for key, value in details.items():
        np.testing.assert_array_equal(value, loaded["predictor"].predict_details(physical_val)[key])
    assert loaded["train_cfg"].data_spec == cfg.data_spec
    assert loaded["train_cfg"].boundary_temperature == cfg.boundary_temperature
    # Count rows seen by the forward pass; use a short final batch too.
    seen = []
    handle = result["model"].register_forward_hook(lambda module, args, out: seen.append(len(args[0])))
    pw = torch.tensor(cfg.pos_weight) if cfg.use_pos_weight else None
    metrics, loss = evaluate_pytorch_binary_classifier(
        result["model"], DataLoader(result["datasets"]["val"], batch_size=31),
        torch.nn.BCEWithLogitsLoss(pos_weight=pw))
    handle.remove()
    assert sum(seen) == len(val.labels)
    expected_losses = np.logaddexp(0, (1 - 2 * val.labels) * details["logit"].astype(float))
    if pw is not None:
        expected_losses *= np.where(val.labels == 1, cfg.pos_weight, 1)
    np.testing.assert_allclose(loss, expected_losses.mean(), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(loss, result["val_metrics"].val_loss, rtol=2e-6, atol=2e-6)
    expected = metrics_from_predictions(val.labels, details["probability"], y_pred=details["prediction"])
    for key, value in asdict(expected).items():
        np.testing.assert_allclose(getattr(metrics, key), value, rtol=2e-6, atol=2e-6)
    saved_metrics = json.loads((result["outdir"] / "metrics/val_metrics.json").read_text())
    assert saved_metrics == asdict(result["val_metrics"])
    assert saved_metrics["epoch"] == loaded["checkpoint"]["selected_epoch"]
    np.savez(result["outdir"] / "reload_probe.npz", physical=physical_val[list(FULL_PAR_COLS)].to_numpy(),
             **details)
    return {"path": str(result["outdir"].relative_to(REPO_ROOT)),
            "selected_epoch": result["selected_epoch"], "metrics": saved_metrics,
            "single_pass_rows": sum(seen), "reload_exact": True}


def reload_check(directory):
    for path in directory.glob("*/reload_probe.npz"):
        stored = np.load(path)
        predictor = load_trained_nn(path.parent)["predictor"]
        for key, values in predictor.predict_details(stored["physical"]).items():
            np.testing.assert_array_equal(values, stored[key])
    print("Fresh-process NN reload checks passed.")


def main():
    torch.set_num_threads(1)
    source_paths = [REPO_ROOT / f"data/processed/{name}.csv" for name in ("train", "val")]
    preserved = list((REPO_ROOT / "results/models").rglob("*")) + source_paths
    preserved += [REPO_ROOT / p for p in ("experiments/nn_full_par.json", "experiments/gp_full_par.json",
                                        "src/debbirth/models/gp/calibrate.py")]
    hashes_before = {str(p.relative_to(REPO_ROOT)): digest(p) for p in preserved if p.is_file()}
    frames, indices = {}, {}
    rng = np.random.default_rng(42)
    for path, count in zip(source_paths, (512, 128)):
        name = path.stem
        frames[name] = pd.read_csv(path, float_precision="round_trip")
        indices[name] = rng.choice(len(frames[name]), count, replace=False)
    outdir = Path(create_run_outdir("t05_nn_validation"))
    metadata = {"source_hashes": {str(p.relative_to(REPO_ROOT)): digest(p) for p in source_paths},
                "rows": {"train": 512, "val": 128}, "sampling_seed": 42,
                "target_policy": "Recorded reached_birth, including six-second get_lb2 failures/timeouts",
                "test_set_used": False, "script_sha256": digest(Path(__file__))}
    runs, outputs = {}, {}
    for formulation, scaling, xb, epochs in [
        ("normalized", "log_standardize", False, 8),
        ("boundary", "log_standardize", False, 8),
        ("normalized", "none", False, 2),
        ("boundary", "none", False, 2),
        ("boundary", "standardize", True, 2),
    ]:
        name = f"{formulation}_{scaling}" + ("_xb" if xb else "")
        spec = DatasetSpec(formulation=formulation, include_x_b=xb)
        prepared = {key: prepare_split(frame, spec, source_name=f"data/processed/{key}.csv").select(indices[key])
                    for key, frame in frames.items()}
        cfg = TrainDEBBirthNetConfig(
            data_spec=spec, net_config=DEBBirthNetConfig(spec.n_features, [16, 16], dropout=0.2),
            scaling_type=scaling, boundary_temperature=0.7 if formulation == "boundary" else 1,
            epochs=epochs, batch_size=64, use_pos_weight=True, checkpoint_selection="best_val_loss",
            outdir=outdir / name, device="cpu")
        initial_weight = cfg.pos_weight
        result = train_net(cfg, save=True, prepared=prepared, data_metadata=metadata)
        assert cfg.pos_weight is initial_weight
        for key, split in prepared.items():
            split.source_rows.to_csv(result["outdir"] / f"{key}_rows.csv", index=False)
        if scaling == "none":
            assert result["scaler"] is None and not (result["outdir"] / "model/scaler.pth").exists()
        else:
            values = torch.tensor(prepared["train"].features, dtype=torch.float32)
            if scaling == "log_standardize":
                values = values.log()
            torch.testing.assert_close(result["scaler"].mean_, values.mean(0))
        assert result["selected_epoch"] == 1 + np.argmin([h.val_loss for h in result["history"]])
        # Audit shuffled whole-record batches against the prepared source.
        for batch in result["dataloaders"]["train"]:
            idx = batch["row_index"].numpy()
            np.testing.assert_array_equal(batch["y"].numpy(), prepared["train"].labels[idx])
            if formulation == "boundary":
                np.testing.assert_array_equal(batch["log_nu_b"].numpy(), prepared["train"].log_nu_b[idx].astype(np.float32))
        runs[name] = verify_run(result, frames["val"].iloc[indices["val"]])
        outputs[name] = result

    boundary = outputs["boundary_log_standardize"]["predictor"]
    for k in (0.3, 1, 3):
        physical = pd.DataFrame({"g": 1., "k": k, "v_Hb": np.logspace(-4, 4, 81), "f": 1.})
        details = boundary.predict_details(physical)
        assert np.ptp(details["learned_output"]) == 0
        assert (np.diff(details["margin"]) < 0).all()
        assert (np.diff(details["probability"]) <= 0).all() and np.ptp(details["probability"]) > 0
        np.testing.assert_allclose(boundary.critical_maturity(physical),
                                   np.exp(boundary.critical_maturity(physical, log=True)))
    equivalent = pd.DataFrame([[1, 3, .125, 1], [.5, 3, .015625, .5]], columns=FULL_PAR_COLS)
    for name in ("normalized_log_standardize", "boundary_log_standardize"):
        values = outputs[name]["predictor"].predict_proba(equivalent)
        np.testing.assert_array_equal(values[:1], values[1:])
    critical = boundary.critical_maturity(equivalent, normalized=False)
    np.testing.assert_allclose(critical[1], critical[0] / 8)
    # Synthetic F=0 verifies strict ties and temperature semantics; no learned k=1 identity is claimed.
    cfg = replace(outputs["boundary_log_standardize"]["train_config"], scaling_type="none")
    zero = build_net(cfg)
    with torch.no_grad():
        for parameter in zero.parameters():
            parameter.zero_()
    for temperature in (0.1, 1., 10.):
        predictor = NNPredictor(zero, replace(cfg, boundary_temperature=temperature))
        fixture = np.array([[1, 1, .5, 1], [1, 1, 1, 1], [1, 1, 2, 1]])
        np.testing.assert_array_equal(predictor.predict(fixture), [1, 0, 0])
        assert predictor.predict_proba(fixture)[1] == 0.5
        assert predictor.predict(fixture, probability_threshold=.5)[1] == 1
    # Positive temperature must not erase a positive margin through logit underflow.
    with torch.no_grad():
        zero.net[-1].bias.fill_(1e-30)
    tiny_margin = NNPredictor(zero, replace(cfg, boundary_temperature=1e30))
    tiny_details = tiny_margin.predict_details([[1, 1, 1, 1]])
    assert tiny_details["margin"][0] > 0 and tiny_details["logit"][0] == 0
    assert tiny_details["prediction"][0] == 1
    assert_raises(lambda: boundary.predict_proba([[1, 1, 0, 1]]))
    assert_raises(lambda: boundary.predict_proba([[1, 1, 1, np.inf]]))
    assert_raises(lambda: boundary.predict_proba([[1, 1, 1]]))
    assert_raises(lambda: boundary.predict_prepared(outputs["normalized_none"]["prepared"]["val"]))
    assert_raises(lambda: replace(cfg, boundary_temperature=0))
    assert_raises(lambda: replace(cfg, net_config=replace(cfg.net_config, threshold=.6)))
    assert boundary.predict_proba(np.empty((0, 4))).shape == (0,)
    extremes = np.array([[1, 1, 1e-300, 1], [1, 1, 1e300, 1]])
    assert np.isfinite(boundary.predict_proba(extremes)).all()
    F = torch.tensor([-1000., 1000.], requires_grad=True)
    z = output_to_logit(F, formulation="boundary", log_nu_b=torch.zeros(2), temperature=.7)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, torch.tensor([1., 0.]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(F.grad).all()

    # Force validation deterioration to prove earlier weights are actually restored.
    synthetic = pd.DataFrame(np.ones((8, 4)), columns=FULL_PAR_COLS)
    synthetic["reached_birth"] = [1] * 7 + [0]
    other = synthetic.assign(reached_birth=[0] * 7 + [1])
    spec = DatasetSpec(formulation="boundary")
    prepared = {"train": prepare_split(synthetic, spec), "val": prepare_split(other, spec)}
    cfg = TrainDEBBirthNetConfig(data_spec=spec, net_config=DEBBirthNetConfig(2, [], dropout=0),
                                scaling_type="standardize", epochs=3, batch_size=8, lr=.1,
                                checkpoint_selection="best_val_loss", outdir=outdir / "checkpoint_restore", device="cpu")
    selected = train_net(cfg, save=True, prepared=prepared)
    first_cfg = replace(cfg, epochs=1, outdir=None, checkpoint_selection="final_epoch")
    first = train_net(first_cfg, prepared=prepared)
    assert first_cfg.outdir is None and first["outdir"] is None
    assert selected["selected_epoch"] == 1 and selected["history"][-1].val_loss > selected["history"][0].val_loss
    for key, value in selected["model"].state_dict().items():
        torch.testing.assert_close(value, first["model"].state_dict()[key], rtol=0, atol=0)
    runs["checkpoint_restore"] = verify_run(selected, other)

    # Exercise ordinary CSV loading for all formulations with a synthetic fixture.
    fixture_dir = outdir / "csv_fixture"
    fixture_dir.mkdir()
    for key, frame in (("train", synthetic), ("val", other), ("test", other)):
        frame.to_csv(fixture_dir / f"{key}.csv", index=False)
    for formulation in ("full_par", "normalized", "boundary"):
        spec = DatasetSpec(formulation=formulation)
        cfg = TrainDEBBirthNetConfig(data_spec=spec, net_config=DEBBirthNetConfig(spec.n_features, [8]),
                                    data_dir=fixture_dir, epochs=1, batch_size=4, scaling_type="none",
                                    outdir=outdir / f"csv_{formulation}", device="cpu")
        result = train_net(cfg, save=True)
        runs[f"csv_{formulation}"] = verify_run(result, other)
        assert result["selected_epoch"] == 1 and result["train_config"].checkpoint_selection == "final_epoch"

    archived = load_trained_nn(REPO_ROOT / "results/models/DEBBirthNet")
    physical = np.array([[1, .3, .1, 1], [1, 1, .5, 1], [1, 1, 2, 1], [.1, 3, .01, .8]])
    with torch.inference_mode():
        original = archived["model"].predict_proba(archived["scaler"].transform(torch.tensor(physical, dtype=torch.float32))).numpy()
    np.testing.assert_array_equal(original, archived["predictor"].predict_proba(physical))
    np.testing.assert_allclose(original, [.99953759, .99282008, 1.53183e-8, .99719739], rtol=2e-6)
    subprocess.run([sys.executable, "-m", "experiments.validate_nn_formulations", "--reload", str(outdir)], check=True)
    assert hashes_before == {p: digest(REPO_ROOT / p) for p in hashes_before}
    summary = {"runs": runs, "checks": "finite losses/probabilities; exact reload; selected weights/metrics; single-pass evaluation; shuffled offsets; train-only scaling; monotonicity; strict ties; temperature; invariance; invalid/extreme inputs; gradients; historical inference",
               "preserved_hashes": hashes_before, "metadata": metadata, "fresh_process_reload": True}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"T05 validation passed: {outdir}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--reload":
        reload_check(Path(sys.argv[2]))
    else:
        main()
