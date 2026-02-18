# Speech Enhancement Algorithm Data Generation and Evaluation Framework (SEADGE)

SEADGE is a framework for generating data and computing evaluation metrics for testing speech enhancement algorithms.

## Installation

Create a virtual environment.

```sh
python3 -m venv .venv
. .venv/bin/activate
```

Install the framework and its dependencies.

```sh
pip3 install git+https://github.com/mhtoribio/es7_project
```

## Configuration

SEADGE reads configuration from a file passed via `-c` in TOML, JSON, or YAML format.
Values are loaded with the following precedence (highest priority first): **explicit overrides** → **environment variables** → **config file** → **defaults**.

### Configuration file overview

The configuration is organized into a few top-level tables. In practice you will typically touch:

* `logging`: log verbosity and formatting
* `paths`: where inputs are read from and where outputs are written
* `dsp`: STFT / sample rates / RIR convolution settings (and ISCLP parameters)
* `roomgen`: random room generation settings (room sizes, RT60, mic array defaults, source constraints)
* `scenariogen`: scenario settings (duration, SNR range, scenarios per room, interference volume range)
* `deeplearning`: CNN training hyperparameters

Additionally:

* `debug` (bool): enables extra debug behavior across the pipeline
* `clean_zip_files` (int): number of DNS Challenge clean archives to download in fixed order; values larger than the available archive list are truncated
* `scenarios` (int | omitted): optional scenario cap used by distant simulation, enhancement, and metrics (`0` behaves like omitted, meaning no cap)

### `paths` and output directory layout

`paths.output_dir` is the root for generated artifacts. SEADGE creates/uses a set of derived subfolders under it, including:

* Rooms / scenarios / simulation: `room_dir`, `scenario_dir`, `distant_dir`, `rir_cache_dir`
* Enhancement outputs: `enhanced_dir`
* Metrics and aggregate stats: `metrics_dir`, `stats_dir`
* Debug artifacts: `debug_dir`
* ML artifacts: `ml_data_dir`, `models_dir`, `checkpoint_dir`

Other important path inputs:

* `paths.clean_dir`: directory containing clean wav files (speech/interference)
* `paths.noise_dir`: directory containing noise wav files (not downloaded by SEADGE; you must provide these files yourself)
* `paths.download_cache_dir`: cache for downloads
* `paths.matfile_dir`: directory for ISCLP `.mat` assets (`lap_div.mat`, `alpha_sqrtMP.mat`); only needed if you use the ISCLP-based enhancement algorithms
* `paths.psd_model`: path to a trained PSD model checkpoint (`.pt`) used by deep-ISCLP enhancement variants; not needed if deep-ISCLP is disabled

### Logging notes

* `logging.level`: standard level name (`DEBUG`, `INFO`, `WARNING`, `ERROR`, etc.)
* `logging.format`: Python `logging.Formatter` format string used by the CLI logger, for example `"[%(asctime)s %(levelname)s] %(message)s"`

### DSP notes

The `dsp` section controls STFT and signal processing settings, including:

* STFT: `window_len`, `hop_size`, `window_type`
  * `window_type` accepts `"sqrt_hann"` and any window name supported by `scipy.signal.get_window` (for example `"hann"`, `"hamming"`, `"blackman"`)
* Sample rates: `datagen_samplerate` (simulation side), `enhancement_samplerate` (enhancement/training side)
* Early/late split: `early_tmax_ms`, `early_offset_ms`
* RIR convolution: `rirconv_method` (`"oaconv"` or `"fft"`), `rirconv_normalize`, `rirconv_xfade_ms`
* Spectrogram plotting (debug/analysis figures): `x_tick_prop`, `y_tick_prop`, `c_range`
* ISCLP parameters live under `dsp.isclpconf`:
  * `L`: linear prediction length
  * `alpha_ISCLP_exp_db`: forgetting-factor exponent used in ISCLP-KF
  * `psi_wLP_db`: LP filter process-variance level
  * `psi_wSC_db_range`: speech-component variance range in dB (linearly spread over frequency bins)
  * `beta_ISCLP_db`: smoothing parameter used for smoothed ISCLP-KF outputs
  * `retf_thres_db`: threshold controlling RETF update decisions

For training, SEADGE also derives a maximum frame count (`L_max`) from the scenario duration + samplerate + STFT settings (and an assumed max RIR tail) to standardize tensor shapes.

### Room Generation Notes

`roomgen` controls random room and source-trajectory generation:

* `rt60_min`, `rt60_max`: random RT60 range in seconds used for each generated room
* `min_dimensions_m`, `max_dimensions_m`: per-axis room-size ranges `(x, y, z)` in meters
* `max_image_order`: maximum image-source reflection order for room acoustics / RIR modeling
* `num_generated_rooms`: number of random rooms to generate
* `enable_noise`: if `true`, add one fixed omni noise source per room; if `false`, no room noise source is added
* `min_num_source_locations`, `max_num_source_locations`: random range for number of speech source trajectories per room
* `min_source_distance_to_wall_m`: minimum allowed distance from source positions to walls
* `min_source_distance_to_mic_m`: minimum allowed distance from source positions to the mic array
* `min_source_inter_spacing`: minimum allowed spacing between different source trajectories
* `max_source_movement_m`: maximum XY displacement per movement step; large values can reduce transition realism because trajectory changes are blended with a windowed/crossfaded transition scheme
* `max_azimuth_rotation_deg`: maximum per-step change in source azimuth (orientation)
* `max_movement_steps`: maximum number of trajectory points per source; set to `1` to disable source movement entirely (no position or azimuth changes over time)
* if `max_source_movement_m = 0` and `max_movement_steps > 1`, position remains fixed but azimuth rotation can still change between steps
* `min_movement_step_duration`: minimum sample interval between trajectory points (`start_sample` spacing)
* `num_mics`: number of microphones in the generated linear array
* `mic_spacing`: spacing in meters between adjacent microphones
* `mic_wall_offset`: distance in meters from the front wall to the array center (y-axis offset)
* `mic_height`: microphone array height in meters

### Scenario Generation Notes

`scenariogen` controls scenario composition:

* `num_speakers`: optional number of active speech sources per scenario; if omitted (`null`/unset), SEADGE uses all available room sources
* `min_wavsource_duration_s`: optional minimum source clip duration in seconds; if omitted, it defaults to `scenario_duration_s` (full-length clips)
* `min_interference_volume`, `max_interference_volume`: uniform sampling range for interferent speaker source volumes
* volume semantics: the target speaker is forced to volume `1.0`; only interferent speakers use the sampled interference-volume range

### Deep Learning Notes

`deeplearning` controls PSD CNN training and tensor preparation:

* `epochs`: number of training epochs
* `batch_size`: mini-batch size for train/eval loaders
* `learning_rate`: Adam optimizer learning rate
* `weight_decay`: Adam weight decay (L2-style regularization)
* `num_max_npz_files`: maximum number of `.npz` scenario files used when building cached tensors (`0` means use all available files)

### Minimal example

This is not a complete config (many fields have defaults), but shows the expected structure.
A TOML example is provided below:

```toml
debug = false
# scenarios = 100  # optional cap; omit to process all

[logging]
level = "INFO"
# format = "%(levelname)s:%(name)s:%(message)s"

[paths]
clean_dir = "/data/clean_wav"
noise_dir = "/data/noise_wav"
output_dir = "/data/seadge_out"
download_cache_dir = "/data/seadge_cache"

[dsp]
window_len = 512
hop_size = 256
datagen_samplerate = 48000
enhancement_samplerate = 16000

[roomgen]
num_generated_rooms = 10

[scenariogen]
scenario_duration_s = 10.0
scenarios_per_room = 20
min_snr_db = 10.0
max_snr_db = 20.0

[deeplearning]
epochs = 50
batch_size = 16
learning_rate = 1e-3
```

### Environment variable overrides

All configuration keys can be overwritten by environment variables.
This can be useful when you want to override, for example, the logging level.

```
# Env override examples:
SEADGE_LOGGING__LEVEL=DEBUG
SEADGE_PATHS__OUTPUT_DIR=/home/seadge/output
SEADGE_PATHS__CLEAN_DIR=/data/clean_wav
```

Nested fields use `__` (double underscore) to separate levels (e.g. `paths.output_dir` → `SEADGE_PATHS__OUTPUT_DIR`).

## Usage

Always start by writing a configuration file (commonly `config.toml`).
Configuration is documented in the section above.

SEADGE is packaged as a command-line tool.
All subcommands support `-h` flag to get help with command-line parameters.

```sh
seadge -h
seadge -c config.toml datagen -h
# etc.
```

### Typical Workflow

1. Prepare your config file (`config.toml` / JSON / YAML) and set input/output paths.
2. Populate clean/noise data:
   - either download clean speech via `seadge -c config.toml datagen download`
   - or provide your own clean speech files in `paths.clean_dir`
   - provide noise WAV files manually in `paths.noise_dir`
3. Run data generation steps (`seadge -c config.toml datagen`, or individual `datagen` subcommands).
4. (Optional) Train the PSD model if you want to run deep-ISCLP enhancement (`seadge -c config.toml train [--gpus GPUS]`).
5. Run enhancement (`seadge -c config.toml enhance`).
6. Compute metrics (`seadge -c config.toml metric`).

### Data Generation

```sh
# Download clean speech from DNS challenge
seadge -c config.toml datagen download

# Run all steps in data generation (except data download)
seadge -c config.toml datagen

# Individual steps can also be run. For example:
seadge -c config.toml datagen room
seadge -c config.toml datagen scenario
seadge -c config.toml datagen rir
seadge -c config.toml datagen distant
seadge -c config.toml datagen tensor
```

`datagen download` downloads DNS clean-speech archives into `paths.download_cache_dir` and extracts them into `paths.clean_dir`.

`datagen room` generates random room definitions (`*.room.json`) and room plots in `paths.room_dir`.

`datagen scenario` generates scenario files (`*.scenario.json`) in `paths.scenario_dir` using clean/noise WAVs and generated rooms.

`datagen rir` precomputes/caches room impulse responses for generated rooms in `paths.rir_cache_dir`.

`datagen distant` simulates distant microphone mixtures from scenarios and writes:
- resampled distant/target wav files in `paths.distant_dir`
- per-scenario PSD-training `.npz` files in `paths.ml_data_dir`

`datagen tensor` converts the `.npz` files in `paths.ml_data_dir` into cached tensors (`X.npy`, `Y.npy`, `meta.json`) used by the training step.

### Enhancement

Enhance generated scenarios.

```sh
seadge -c config.toml enhance
```

Enhanced WAV files are written to the enhancement output directory (see "Configuration").

By default, enhancement includes ISCLP-based variants (including deep-ISCLP variants).
When using these defaults:
- `paths.matfile_dir` must contain `lap_div.mat` and `alpha_sqrtMP.mat`
- `paths.psd_model` must point to a trained checkpoint for deep-ISCLP variants

If you do not want to use ISCLP/deep-ISCLP, disable or comment out those algorithms in the enhancement runner (see "Adding new algorithms").

### Metrics

```sh
# Compute metrics for all enhanced audio files.
seadge -c config.toml metric
```

Computed metrics are written to CSV files.
Per-scenario metrics are written to `<METRICS_DIR>/<METRIC_NAME>.csv`.
Statistics (min, max, avg, median, etc.) are computed and stored in `<METRICS_DIR>/<METRIC_NAME>_boxstats.csv`.

### Utility Commands

Inspect resolved config and scenario-linked artifacts:

```sh
# Dump active config as JSON
seadge -c config.toml dump config

# Dump scenario-linked values
seadge -c config.toml dump scenario <SCENARIO_ID>
seadge -c config.toml dump scenario <SCENARIO_ID> --room
seadge -c config.toml dump scenario <SCENARIO_ID> --wav
```

`<SCENARIO_ID>` is the scenario hash (the filename stem of `*.scenario.json` in `paths.scenario_dir`).

Cleanup helper:

```sh
seadge -c config.toml clean
```

`clean` recursively removes `paths.output_dir` without a confirmation prompt.
Use it only when you want to delete generated artifacts and start from scratch.

### Training CNN

In the project, the [Integrated Sidelobe Cancelling and Linear Prediction Kalman Filter (ISCLP-KF)](https://github.com/tdietzen/ISCLP-KF) was investigated.
It was investigated whether a target speech power spectral density estimate based on a convolutional neural network could enhance this algorithm.
Training this CNN was integrated into the SEADGE framework.
Training requires at least one CUDA GPU and supports multi-GPU training.

```sh
# Train PSD CNN model
seadge -c config.toml train [--gpus GPUS]
```

Requirements:
- At least one CUDA-visible GPU is required (`train` exits if no GPU is available).
- `--gpus` is optional; if omitted, SEADGE uses all detected GPUs.
- Tensor cache must exist in `paths.ml_data_dir` (`X.npy`, `Y.npy`, `meta.json`), typically produced by:
  - `seadge -c config.toml datagen distant`
  - `seadge -c config.toml datagen tensor`

Training writes checkpoints to `paths.checkpoint_dir` and final model weights to `paths.models_dir/model.pt`.

If you are not using deep-ISCLP, you can skip training entirely.

In the future, it may be desirable to split this PSD training to a separate project, since it is highly specific to the project done by the original authors.
As a workaround, if you want to use the framework for evaluating your own algorithms, you can simply choose not to call the training module and comment out the specific ISCLP-KF algorithms in the enhancement runner module (see "Adding new algorithms").

## Adding new algorithms

New speech enhancement algorithms can be easily added for comparison across metrics.
To do so, they must be added to the [speech enhancement runner](seadge/enhance.py) by editing the `_enhance_one_file` function.
The enhanced wav file is written to `<ENHANCED_DIR>/<ALGORITHM_NAME>/<SCENARIO_HASH>.wav`.
When doing so, it is automatically picked up by the [performance metrics module](seadge/metric.py).

## Known Issues

An issue persists in the metrics evaluation module which results in incorrect/excessively low metric scores for intrusive metrics.
The original authors expect that the issue stems from a misalignment of the reference signal (either time alignment or level difference).

## Credits

Developed by Project Group 752 in the Autumn Semester 2025 in the study of Electronic Systems at Aalborg University.

Group members:
- Markus Heinrich Toribio
- Snorre Johnsen
- Mathias Majland Jørgensen
- Rasmus Mellergaard Christensen

Supervisor:
- Jesper Rindom Jensen, Aalborg University

External advisors:
- Pejman Mowlaee, GN Audio
- Peter Theilgaard Hounum, GN Audio
