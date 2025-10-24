from scipy.io import wavfile

from seadge import config
from seadge import room_modelling
from seadge.utils.distant_sim import animate_rir_time, animate_freqresp
from seadge.utils.log import log

def gen_dynamic_rir_animation(fps=30, duration_s=2, src_idx=0, *, mic=0, nfft=None, db=True):
    cfg = config.get()
    N = int(duration_s * cfg.dsp.samplerate)

    # Time-domain instantaneous RIR animation
    animate_rir_time(
        cfg.room.sources[src_idx],
        N=N, fs=cfg.dsp.samplerate,
        room_cfg=cfg.room, cache_root=cfg.paths.rir_cache_dir,
        xfade_ms=cfg.dsp.rirconv_xfade_ms, normalize=cfg.dsp.rirconv_normalize,
        fps=fps, duration_s=duration_s,
        outpath=cfg.paths.debug_dir / "anim" / "rir_time.mp4",
    )

    # Frequency-response |H(f,t)| animation for one mic
    animate_freqresp(
        cfg.room.sources[src_idx],
        N=N, fs=cfg.dsp.samplerate,
        room_cfg=cfg.room, cache_root=cfg.paths.rir_cache_dir,
        xfade_ms=cfg.dsp.rirconv_xfade_ms, normalize=cfg.dsp.rirconv_normalize,
        fps=fps, duration_s=duration_s,
        mic=mic, db=db, nfft=nfft,
        outpath=cfg.paths.debug_dir / "anim" / f"rir_freq_mic{mic}.mp4",
    )

def main():
    cfg = config.get()
    room_modelling.main()
    gen_dynamic_rir_animation(src_idx=-1) # DEBUG
