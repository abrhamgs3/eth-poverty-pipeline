import os

from eth_poverty_pipeline.config import PipelineConfig
from eth_poverty_pipeline.pipeline import run_full_pipeline


def _config(tmp_path, **overrides):
    kwargs = dict(
        seed=1,
        real_data_path=str(tmp_path / "does_not_exist.parquet"),  # forces synthetic fallback
        output_dir=str(tmp_path / "output"),
        n_bootstrap=20,
    )
    kwargs.update(overrides)
    return PipelineConfig(**kwargs)


def test_full_pipeline_runs_end_to_end(tmp_path):
    results = run_full_pipeline(_config(tmp_path), client_name="mock")

    assert results["data_source"] == "synthetic fallback"
    assert set(results["regression_results"].keys()) == {
        "Benchmark\n(Real Poverty Data)", "Naive\n(AI Screen)", "Corrected\n(Matrix Method)",
    }
    assert os.path.isfile(results["report_path"])
    assert os.path.isfile(results["chart_path"])

    with open(results["report_path"], encoding="utf-8") as f:
        report_text = f.read()
    for line in report_text.splitlines():
        if line.startswith("| Benchmark") or line.startswith("| Naive") or line.startswith("| Corrected"):
            assert line.count("|") == 5  # well-formed 4-column row, no stray \n


def test_pipeline_is_reproducible_given_same_seed(tmp_path):
    r1 = run_full_pipeline(_config(tmp_path, output_dir=str(tmp_path / "a")), client_name="mock")
    r2 = run_full_pipeline(_config(tmp_path, output_dir=str(tmp_path / "b")), client_name="mock")

    key = "Corrected\n(Matrix Method)"
    assert r1["regression_results"][key]["beta"] == r2["regression_results"][key]["beta"]
