from pathlib import Path
from typing import Optional
import random
import math

from seadge.utils.visualization import plot_room_topdown, plot_room_3d
from seadge.utils.log import log
from seadge.utils.cache import make_pydantic_cache_key
from seadge import config

def _dist_2d(a: tuple[float, float], b: tuple[float, float]):
    """Computes distance from a to b"""
    diff_x = b[0] - a[0]
    diff_y = b[1] - a[1]
    return math.sqrt(diff_x**2 + diff_y**2)

def allowed_y_intervals_at_x(
    x: float,
    mic_positions: list[tuple[float, float]],
    distance_min: float,
    *,
    clip: Optional[tuple[float, float]] = None,
    strict: bool = False,
    eps: float = 1e-12,
) -> list[tuple[float, float]]:
    """
    Solve distance(x, y; xi, yi) >= distance_min for all mics (xi, yi), at fixed x.
    Returns the (possibly multiple) valid y-intervals that satisfy ALL mic constraints.

    Each mic gives:
        (y - yi)^2 >= d^2 - (x - xi)^2
    If |x - xi| >= d: no restriction from that mic.
    If |x - xi| < d: forbidden interval (yi - r, yi + r), r = sqrt(d^2 - (x - xi)^2).
    Valid set = complement of the union of all forbidden intervals (optionally clipped).

    Args:
        x: fixed x-coordinate.
        mic_positions: iterable of (xi, yi).
        distance_min: required minimum distance d.
        clip: optional (ymin, ymax) to bound the output.
        strict: if True, enforce distance > d (exclude boundary slightly).
        eps: small numeric cushion.

    Returns:
        Sorted list of disjoint (y_start, y_end) intervals where placement is valid.
    """
    if distance_min < 0:
        raise ValueError("distance_min must be non-negative")

    import math

    ymin = -math.inf if clip is None else clip[0]
    ymax =  math.inf if clip is None else clip[1]
    if ymin >= ymax:
        return []

    d2 = distance_min * distance_min

    # Collect forbidden intervals from each mic
    forbidden: list[tuple[float, float]] = []
    for xi, yi in mic_positions:
        dx = x - xi
        dx2 = dx * dx
        if dx2 < d2 - eps:
            r = math.sqrt(max(0.0, d2 - dx2))
            a, b = yi - r, yi + r
            if strict:
                a -= eps
                b += eps
            forbidden.append((a, b))
        elif not strict and abs(dx2 - d2) <= eps:
            # On the boundary in x: distance >= d still allows all y.
            continue
        else:
            # |dx| >= d ⇒ no restriction
            continue

    if not forbidden:
        return [] if ymin == ymax else [(ymin, ymax)]

    # Merge forbidden intervals
    forbidden.sort()
    merged: list[tuple[float, float]] = []
    ca, cb = forbidden[0]
    for a, b in forbidden[1:]:
        if a <= cb + eps:
            cb = max(cb, b)
        else:
            merged.append((ca, cb))
            ca, cb = a, b
    merged.append((ca, cb))

    # Complement within [ymin, ymax]
    valid: list[tuple[float, float]] = []
    cursor = ymin
    for a, b in merged:
        if b <= ymin or a >= ymax:
            continue
        a = max(a, ymin)
        b = min(b, ymax)
        if a > cursor:
            valid.append((cursor, a))
        cursor = max(cursor, b)
        if cursor >= ymax:
            break
    if cursor < ymax:
        valid.append((cursor, ymax))

    # Remove tiny/degenerate intervals
    return [(a, b) for a, b in valid if b - a > eps]

def gen_one_source_loc(
        room_dimensions: tuple[float, float, float],
        min_source_distance_to_wall_m: float,
        min_source_distance_to_mic_m: float,
        min_source_inter_spacing: float,
        max_source_movement_m: float,
        max_azimuth_rotation_deg: float,
        mic_locs: list[tuple[float, float, float]],
        max_movement_steps: int,
        min_movement_step_duration: int,
        *,
        current_sources: Optional[list[config.SourceSpec]] = None,
    ) -> Optional[config.SourceSpec]:
    """
    Generate a SourceSpec with a collision-free trajectory. Returns None on any failure.
    """
    rx, ry, rz = room_dimensions
    wall = float(min_source_distance_to_wall_m)
    mic_keepout = float(min_source_distance_to_mic_m)
    src_keepout = float(min_source_inter_spacing)
    step_max = float(max_source_movement_m)
    max_daz = float(max_azimuth_rotation_deg)
    max_steps = int(max_movement_steps)

    # Bounds respecting wall keep-out
    xmin, xmax = wall, max(wall, rx - wall)
    ymin, ymax = wall, max(wall, ry - wall)
    zmin, zmax = wall, max(wall, rz - wall)
    if not (xmin < xmax and ymin < ymax and zmin < zmax):
        return None  # room too small for the keep-out distances

    # Mic XY list
    mic_xy = [(mx, my) for (mx, my, _mz) in mic_locs]

    # Flatten existing sources' XY waypoints for inter-source spacing checks
    other_xy: list[tuple[float, float]] = []
    if current_sources:
        for s in current_sources:
            for loc in getattr(s, "location_history", []):
                x, y, _z = loc.location_m
                other_xy.append((float(x), float(y)))

    # --- helpers -------------------------------------------------------------
    def _far_from_all_sources(x: float, y: float) -> bool:
        if not other_xy:
            return True
        for ox, oy in other_xy:
            if _dist_2d((x, y), (ox, oy)) < src_keepout:
                return False
        return True

    def _far_from_mics(x: float, y: float) -> bool:
        for mx, my in mic_xy:
            if _dist_2d((x, y), (mx, my)) < mic_keepout:
                return False
        return True

    def _inside_walls(x: float, y: float, z: float) -> bool:
        return (xmin <= x <= xmax) and (ymin <= y <= ymax) and (zmin <= z <= zmax)

    def _choose_y_given_x(x: float) -> Optional[float]:
        intervals = allowed_y_intervals_at_x(
            x, mic_positions=mic_xy, distance_min=mic_keepout,
            clip=(ymin, ymax), strict=False
        )
        if not intervals:
            return None

        # length-weighted interval choice
        lengths = [b - a for (a, b) in intervals]
        total_len = sum(lengths)
        if total_len <= 0:
            return None
        rpick = random.random() * total_len
        acc = 0.0
        for (a, b), L in zip(intervals, lengths):
            if acc + L >= rpick:
                # try several times inside this interval to satisfy inter-source spacing
                for _ in range(8):
                    y_try = random.uniform(a, b)
                    if _far_from_all_sources(x, y_try):
                        return y_try
                # fall through if crowded: try next interval
            acc += L
        return None

    def _sample_initial_xyz(max_tries: int = 2000) -> Optional[tuple[float, float, float]]:
        for _ in range(max_tries):
            x = random.uniform(xmin, xmax)
            y = _choose_y_given_x(x)
            if y is None:
                continue
            z = random.uniform(zmin, zmax)
            if _inside_walls(x, y, z) and _far_from_mics(x, y) and _far_from_all_sources(x, y):
                return (x, y, z)
        return None

    def _sample_next_xy(x0: float, y0: float, max_tries: int = 2000) -> Optional[tuple[float, float]]:
        if step_max <= 0:
            return (x0, y0)
        for _ in range(max_tries):
            r = step_max * math.sqrt(random.random())   # area-uniform radius
            th = random.random() * 2.0 * math.pi
            x = x0 + r * math.cos(th)
            y = y0 + r * math.sin(th)
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                continue
            if not _far_from_mics(x, y):
                continue
            if not _far_from_all_sources(x, y):
                continue
            return (x, y)
        return None

    # --- build trajectory ----------------------------------------------------
    start = _sample_initial_xyz()
    if start is None:
        return None
    x0, y0, z0 = start

    az = random.gauss(-90, 20) # bias facing towards the mic array
    if az < -180.0:
        az = -180.0
    elif az > 180.0:
        az = 180.0
    col = 90.0

    locs: list[config.SourceSpec.LocationSpec] = []
    locs.append(config.SourceSpec.LocationSpec(
        pattern="cardioid",
        start_sample=0,
        azimuth_deg=round(az, 0),
        colatitude_deg=col,
        location_m=(round(x0, 4), round(y0, 4), round(z0, 4)),
    ))

    x_prev, y_prev = x0, y0
    for s in range(1, max_steps):
        nxt = _sample_next_xy(x_prev, y_prev)
        if nxt is None:
            break  # stop early if no valid continuation found
        x_new, y_new = nxt
        z_new = z0  # keep constant height

        d_az = random.gauss(0, max_daz/2)
        if d_az > max_daz:
            d_az = max_daz
        elif d_az < -max_daz:
            d_az = -max_daz
        az = (az + d_az + 360.0) % 360.0
        start_samp = s * int(min_movement_step_duration)

        locs.append(config.SourceSpec.LocationSpec(
            pattern="cardioid",
            start_sample=start_samp,
            azimuth_deg=round(az, 0),
            colatitude_deg=col,
            location_m=(round(x_new, 4), round(y_new, 4), round(z_new, 4)),
        ))

        x_prev, y_prev = x_new, y_new

    if not locs:
        return None

    return config.SourceSpec(location_history=locs)

def gen_source_locs(
        num_wanted_sources: int,
        room_dimensions: tuple[float, float, float],
        min_source_distance_to_wall_m: float,
        min_source_distance_to_mic_m: float,
        min_source_inter_spacing: float,
        max_source_movement_m: float,
        max_movement_steps: int,
        min_movement_step_duration: int,
        max_azimuth_rotation_deg: float,
        mic_locs: list[tuple[float, float, float]],
    ) -> list[config.SourceSpec]:
    room_x, room_y, _ = room_dimensions
    allowed_x_range = (min_source_distance_to_wall_m, room_x-min_source_distance_to_wall_m)
    wall_allowed_y_range = (min_source_distance_to_wall_m, room_y-min_source_distance_to_wall_m)
    sources = []
    for i in range(num_wanted_sources):
        src = gen_one_source_loc(
            room_dimensions=room_dimensions,
            min_source_distance_to_wall_m=min_source_distance_to_wall_m,
            min_source_distance_to_mic_m=min_source_distance_to_mic_m,
            min_source_inter_spacing=min_source_inter_spacing,
            max_source_movement_m=max_source_movement_m,
            max_azimuth_rotation_deg=max_azimuth_rotation_deg,
            mic_locs=mic_locs,
            max_movement_steps=max_movement_steps,
            min_movement_step_duration=min_movement_step_duration,
            current_sources=sources,
        )
        if src:
            sources.append(src)
        else:
            log.info(f"Room too cramped, generating only {i} source(s) instead of {num_wanted_sources}")
            break

    return sources

def gen_random_room(gencfg: config.RoomGenCfg) -> config.RoomCfg:
    room = config.RoomCfg()
    room.max_image_order = gencfg.max_image_order
    room.rt60 = round(random.uniform(gencfg.rt60_min, gencfg.rt60_max), 2)
    num_wanted_sources = random.randint(gencfg.min_num_source_locations, gencfg.max_num_source_locations)
    room.dimensions_m = tuple([round(random.uniform(xmin, xmax), 2) for xmin, xmax in zip(gencfg.min_dimensions_m, gencfg.max_dimensions_m)])

    # mic array
    mic_x = round(room.dimensions_m[0] / 2, 3)
    mic_y = round(gencfg.mic_wall_offset, 3)
    mic_z = round(gencfg.mic_height, 3)
    mic_locs = config.MicDesc(
            num_mics=gencfg.num_mics,
            spacing=gencfg.mic_spacing,
            origin=(mic_x, mic_y, mic_z),
            ).expand_positions()
    room.mic_pos = mic_locs

    room.sources = gen_source_locs(
        num_wanted_sources=num_wanted_sources,
        room_dimensions=room.dimensions_m,
        min_source_distance_to_wall_m=gencfg.min_source_distance_to_wall_m,
        min_source_distance_to_mic_m=gencfg.min_source_distance_to_mic_m,
        min_source_inter_spacing=gencfg.min_source_inter_spacing,
        max_source_movement_m=gencfg.max_source_movement_m,
        max_movement_steps=gencfg.max_movement_steps,
        min_movement_step_duration=gencfg.min_movement_step_duration,
        max_azimuth_rotation_deg=gencfg.max_azimuth_rotation_deg,
        mic_locs=mic_locs,
    )

    return room

def create_all_rooms(gencfg: config.RoomGenCfg, outpath: Path):
    """
    Generates rooms according to config and writes them to disk
    """
    for _ in range(gencfg.num_generated_rooms):
        room_cfg = gen_random_room(gencfg)
        room_hash = make_pydantic_cache_key(room_cfg)
        log.debug(f"Generated room {room_hash}")
        config.save_json(room_cfg, outpath / f"{room_hash}.room.json")
        plot_room_topdown(room_cfg, outpath / f"{room_hash}_topdown.png")
        plot_room_3d(room_cfg, outpath / f"{room_hash}_3d.png")

def main():
    cfg = config.get()
    log.info("Generating rooms")
    create_all_rooms(cfg.roomgen, cfg.paths.room_dir)
