from scipy.io import wavfile

from seadge import config
from seadge import room_modelling
from seadge.utils.distant_sim import animate_rir_time, animate_freqresp, sim_distant_src
from seadge.utils.log import log

def gen_dynamic_rir_animation(room_cfg: config.RoomCfg, fps=30, duration_s=2, src_idx=0, *, mic=0, nfft=None, db=True):
    cfg = config.get()
    N = int(duration_s * cfg.dsp.samplerate)

    # Time-domain instantaneous RIR animation
    animate_rir_time(
        room_cfg.sources[src_idx],
        N=N, fs=cfg.dsp.samplerate,
        room_cfg=room_cfg, cache_root=cfg.paths.rir_cache_dir,
        xfade_ms=cfg.dsp.rirconv_xfade_ms, normalize=cfg.dsp.rirconv_normalize,
        fps=fps, duration_s=duration_s,
        outpath=cfg.paths.debug_dir / "anim" / "rir_time.mp4",
    )

    # Frequency-response |H(f,t)| animation for one mic
    animate_freqresp(
        room_cfg.sources[src_idx],
        N=N, fs=cfg.dsp.samplerate,
        room_cfg=room_cfg, cache_root=cfg.paths.rir_cache_dir,
        xfade_ms=cfg.dsp.rirconv_xfade_ms, normalize=cfg.dsp.rirconv_normalize,
        fps=fps, duration_s=duration_s,
        mic=mic, db=db, nfft=nfft,
        outpath=cfg.paths.debug_dir / "anim" / f"rir_freq_mic{mic}.mp4",
    )

def main():
    cfg = config.get()
    room = config.load_room("/home/markus/shit/seadge_output/rooms/064218cd7b9b8911f6dfb503588273cc7e4ef815.json") # mhtdebug
    room_modelling.main()
    # gen_dynamic_rir_animation(room, src_idx=-1) # DEBUG

    # mhtdebug
    from scipy.io import wavfile
    _, x = wavfile.read("/home/markus/shit/seadge_clean_data/datasets_fullband/clean_fullband/read_speech/book_00000_chp_0009_reader_06709_3_seg_2.wav")
    x = (0.99 / 32767) * x
    from scipy.signal import resample_poly
    x = resample_poly(x, 1, 3)
    import numpy as np
    wavfile.write("/home/markus/shit/isclp-debug/x.wav", cfg.dsp.samplerate, x)
    y = sim_distant_src(x, room.sources[0], fs=cfg.dsp.samplerate, room_cfg=room, cache_root=cfg.paths.rir_cache_dir, xfade_ms=cfg.dsp.rirconv_xfade_ms)
    y = (32737 / (np.max(np.abs(y)) + 1e-12)) * y
    wavfile.write("/home/markus/shit/isclp-debug/test0.wav", cfg.dsp.samplerate, y.astype(np.int16))
    y = sim_distant_src(x, room.sources[1], fs=cfg.dsp.samplerate, room_cfg=room, cache_root=cfg.paths.rir_cache_dir, xfade_ms=cfg.dsp.rirconv_xfade_ms)
    y = (32737 / (np.max(np.abs(y)) + 1e-12)) * y
    wavfile.write("/home/markus/shit/isclp-debug/test1.wav", cfg.dsp.samplerate, y.astype(np.int16))
