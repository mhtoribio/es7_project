from pathlib import Path
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import os

from seadge.utils.files import files_in_path_recursive
from seadge.utils.log import log
from seadge.utils.wavfiles import load_wav, write_wav
from seadge.utils.scenario import load_scenario
from seadge.utils.cache import make_pydantic_cache_key
from seadge import config
from seadge.enhancement import isclp

def _enhance_one_file(
    scenfile: Path,
    outdir: Path,
    room_dir: Path,
    distant_dir: Path,
    lap_div_file: Path,
    alpha_file: Path,
    debug_dir: Path | None,
    dspconf: config.DspCfg,
    psdmodel_path: Path,
    ):
    # Load scenario and room
    scen = load_scenario(scenfile)
    scen_hash = make_pydantic_cache_key(scen)
    log.debug(f"Enhancing scenario {scen_hash}")
    room = config.load_room(room_dir / f"{scen.room_id}.room.json")

    micpos = [np.asarray(x, dtype=np.float32) for x in room.mic_pos]

    # Build source position list
    target_source_idx = scen.target_speaker.sourceloc
    target_loc = room.sources[target_source_idx].location_history[0].location_m
    sourcepos = [np.asarray(target_loc, dtype=np.float32)]
    for i, src in enumerate(room.sources):
        if i == target_source_idx:
            continue # already added the target speaker at index 0 of sourcepos list
        loc = src.location_history[0].location_m # first position
        sourcepos.append(np.asarray(loc, dtype=np.float32))

    # Build ISCLP config class
    isclp_conf = isclp.ISCLPConfig(
        fs = dspconf.enhancement_samplerate,
        M = len(micpos),
        N = len(sourcepos),
        L = dspconf.isclpconf.L,
        N_STFT = dspconf.window_len,
        R_STFT = dspconf.hop_size,
        micpos = micpos,
        sourcepos = sourcepos,
        alpha_ISCLP_exp_db = dspconf.isclpconf.alpha_ISCLP_exp_db,
        psi_wLP_db = dspconf.isclpconf.psi_wLP_db,
        psi_wSC_db_range = dspconf.isclpconf.psi_wSC_db_range,
        beta_ISCLP_db = dspconf.isclpconf.beta_ISCLP_db,
        retf_thres_db = dspconf.isclpconf.retf_thres_db,
        x_tick_prop = dspconf.x_tick_prop,
        y_tick_prop = dspconf.y_tick_prop,
        c_range = dspconf.c_range,
    )

    # Load distant and clean speech
    _, y = load_wav(distant_dir / f"{scen_hash}.wav", expected_fs=dspconf.enhancement_samplerate, expected_ndim=2)
    _, s = load_wav(distant_dir / f"{scen_hash}_target.wav", expected_fs=dspconf.enhancement_samplerate, expected_ndim=1)

    e_vanilla, e_smooth_vanilla, e_dnn, e_smooth_dnn, e_oracle, e_smooth_oracle = isclp.enhance_isclp_kf(
        y = y,
        s = s,
        isclp_conf = isclp_conf,
        debug_dir = debug_dir / "isclp" / f"{scen_hash}" if debug_dir else None,
        lap_div_file = lap_div_file,
        alpha_file = alpha_file,
        modelpath = psdmodel_path,
    )
    # Write results
    write_wav(outdir / "isclp-kf" / f"{scen_hash}.wav", e_vanilla, fs=dspconf.enhancement_samplerate)
    write_wav(outdir / "isclp-kf-smooth" / f"{scen_hash}.wav", e_smooth_vanilla, fs=dspconf.enhancement_samplerate)
    write_wav(outdir / "deep-isclp-kf" / f"{scen_hash}.wav", e_dnn, fs=dspconf.enhancement_samplerate)
    write_wav(outdir / "deep-isclp-kf-smooth" / f"{scen_hash}.wav", e_smooth_dnn, fs=dspconf.enhancement_samplerate)
    write_wav(outdir / "oracle-isclp-kf" / f"{scen_hash}.wav", e_oracle, fs=dspconf.enhancement_samplerate)
    write_wav(outdir / "oracle-isclp-kf-smooth" / f"{scen_hash}.wav", e_smooth_oracle, fs=dspconf.enhancement_samplerate)
    # also write target and distant for metric computation
    write_wav(outdir / "target" / f"{scen_hash}.wav", s, fs=dspconf.enhancement_samplerate)
    write_wav(outdir / "distant" / f"{scen_hash}.wav", y[:,0], fs=dspconf.enhancement_samplerate) # mic 0 as ref mic

def enhance_scenarios(
    scenario_dir: Path,
    outdir: Path,
    room_dir: Path,
    distant_dir: Path,
    matfile_dir: Path,
    debug_dir: Path | None,
    dspconf: config.DspCfg,
    psdmodel_path: Path,
):
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    num_processes = int(slurm_cpus) if slurm_cpus else os.cpu_count()
    scenario_files = list(files_in_path_recursive(scenario_dir, "*.scenario.json"))
    log.info(f"Enhancing {len(scenario_files)} scenarios with {num_processes} workers")

    worker_fn = partial(
        _enhance_one_file,
        outdir=outdir,
        room_dir=room_dir,
        distant_dir=distant_dir,
        lap_div_file=matfile_dir / "lap_div.mat",
        alpha_file=matfile_dir / "alpha_sqrtMP.mat",
        debug_dir=debug_dir,
        dspconf=dspconf,
        psdmodel_path=psdmodel_path
    )

    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(scenario_files), desc="Enhancing scenarios") as pbar:
            # imap_unordered yields one result as each worker finishes
            for _ in pool.imap_unordered(worker_fn, scenario_files):
                pbar.update()

def main():
    cfg = config.get()
    enhance_scenarios(
        scenario_dir=cfg.paths.scenario_dir,
        outdir=cfg.paths.enhanced_dir,
        room_dir=cfg.paths.room_dir,
        distant_dir=cfg.paths.distant_dir,
        matfile_dir=cfg.paths.matfile_dir,
        debug_dir=cfg.paths.debug_dir if cfg.debug else None,
        dspconf=cfg.dsp,
        psdmodel_path=cfg.paths.psd_model,
    )
