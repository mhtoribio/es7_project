# Evaluation metrics

## outputs:
# metrics.json:
# - json file of min, max, avg scores for PESQ, DNSMOS, SRMR, ESTOI
# - meta section for number of scenarios, rooms, noise wavfiles, etc.
# <metric_name>.csv files
# - Metric scores for each scenario per metric (csv file with enhancement algorithm as columns)

import os
import numpy as np
import csv
from collections import defaultdict, OrderedDict
from scipy.io import wavfile
from tqdm import tqdm
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from pystoi import stoi
from pesq import pesq

from seadge.utils import dsp
from seadge.utils.log import log
from seadge.utils.files import files_in_path_recursive
from seadge.utils.scenario import load_scenario
from seadge.utils.cache import make_pydantic_cache_key
from seadge.utils.visualization import spectrogram
from seadge.utils.wavfiles import load_wav
from seadge import config

def _metrics(
    target: np.ndarray,
    distant: np.ndarray,
    fs: int,
) -> dict[str, float]:
    results = {}
    results["stoi"] = stoi(target, distant, fs, extended=False)
    results["estoi"] = stoi(target, distant, fs, extended=True)
    results["pesq"] = pesq(fs, target, distant, 'wb')
    return results

def _metrics_for_one_scenario(
    scenfile: Path,
    distant_dir: Path,
    enhanced_dir: Path,
    algos: list[str],
    debug_dir: Path | None,
    dspconf: config.DspCfg,
) -> tuple[str, dict[str, dict[str, float]]]:
    """
    Computes metrics for one scenario file
    Returns:
    (
      scenario_hash,
      {algorithm_name: {metric_name: float, ...}, ...}
    )
    """
    scen = load_scenario(scenfile)
    scen_hash = make_pydantic_cache_key(scen)
    results = {}

    for algo in algos:
        algo_enh_dir = enhanced_dir / algo
        target = load_wav(distant_dir / f"{scen_hash}_target.wav", expected_fs=dspconf.enhancement_samplerate, expected_ndim=1)
        distant = load_wav(algo_enh_dir / f"{scen_hash}.wav", expected_fs=dspconf.enhancement_samplerate, expected_ndim=1)
        distant = distant[:len(target)] # make distant same length as target. They have same offset from their RIRs.
        results[algo] = _metrics(target, distant, dspconf.enhancement_samplerate)
        if debug_dir:
            X = dsp.stft(distant, dspconf.enhancement_samplerate)
            if algo == "target":
                title = "Target Speech"
            elif algo == "distant":
                title = "Distant Noisy Speech"
            else:
                title = f"Enhanced Speech ({algo})"
            spectrogram(X, debug_dir/f"{scen_hash}_{algo}.png", title=title, x_tick_prop=dspconf.x_tick_prop, y_tick_prop=dspconf.y_tick_prop, c_range=dspconf.c_range)

    if debug_dir:
        import json
        with open(debug_dir / f"{scen_hash}.json", "w") as f:
            json.dump(results, f)

    return scen_hash, results

def write_metrics_csv(outdir: Path, results: dict[str, dict[str, dict[str, float]]]):
    preferred_alg_order = ["target", "distant"]
    def order_algorithms(seen):
        # keep preferred order first, then any extras sorted
        extras = sorted(set(seen) - set(preferred_alg_order))
        return [a for a in preferred_alg_order if a in seen] + extras

    records = []
    algorithms_seen = set()
    metrics_seen = set()
    for scen_hash, res in results.items():
        for alg, met_dict in res.items():
                    algorithms_seen.add(alg)
                    for metric, value in met_dict.items():
                        metrics_seen.add(metric)
                        records.append((scen_hash, alg, metric, value))

    alg_order = order_algorithms(algorithms_seen)
    # Build: metric -> scenario -> algorithm -> value
    metric_map = defaultdict(lambda: defaultdict(dict))
    for scen, alg, metric, value in records:
        metric_map[metric][scen][alg] = value

    # Write one CSV per metric
    for metric, scen_dict in metric_map.items():
        out_path = outdir / f"{metric}.csv"
        log.debug(f"Writing metric file {out_path}")
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scen", *alg_order])

            # stable row order
            for scen in sorted(scen_dict.keys()):
                row = [scen]
                alg_values = scen_dict[scen]
                for alg in alg_order:
                    row.append(alg_values.get(alg, ""))  # blank if missing
                w.writerow(row)

def compute_all_metrics(
    scenario_dir: Path,
    outdir: Path,
    distant_dir: Path,
    enhanced_dir: Path,
    debug_dir: Path | None,
    dspconf: config.DspCfg,
):
    if debug_dir:
        debug_dir.mkdir(exist_ok=True)
    # Enhancement algorithm names are stored as folder names
    # i.e. <algo-name>/<scenario_id>.wav
    enh_algo_names = next(os.walk(enhanced_dir))[1]
    log.debug(f"Found enhancement algorithm names {enh_algo_names} as the subdirs of {enhanced_dir}")
    log.info(f"Computing metrics for {len(enh_algo_names)} algorithms")
    if len(enh_algo_names) == 0:
        return

    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    num_processes = int(slurm_cpus) if slurm_cpus else os.cpu_count()
    scenario_files = list(files_in_path_recursive(scenario_dir, "*.scenario.json"))
    log.info(f"Computing metrics for {len(scenario_files)} scenarios with {num_processes} workers")

    worker_fn = partial(
        _metrics_for_one_scenario,
        distant_dir=distant_dir,
        enhanced_dir=enhanced_dir,
        algos=enh_algo_names,
        debug_dir=debug_dir,
        dspconf=dspconf,
    )

    results: dict[str, dict[str, dict[str, float]]] = {}
    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(scenario_files), desc="Computing metrics for scenarios") as pbar:
            # imap_unordered yields one result as each worker finishes
            for scen_hash, res in pool.imap_unordered(worker_fn, scenario_files):
                results[scen_hash] = res
                pbar.update()

    if debug_dir:
        import json
        with open(debug_dir / "all.json", "w") as f:
            json.dump(results, f)

    write_metrics_csv(outdir, results)

def main():
    cfg = config.get()
    compute_all_metrics(
        scenario_dir=cfg.paths.scenario_dir,
        outdir=cfg.paths.metrics_dir,
        distant_dir=cfg.paths.distant_dir,
        enhanced_dir=cfg.paths.enhanced_dir,
        debug_dir=cfg.paths.debug_dir/"metric" if cfg.debug else None,
        dspconf=cfg.dsp,
    )
