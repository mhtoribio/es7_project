import sys
from contextlib import contextmanager
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.run import main as dist_main

@contextmanager
def _argv(new_argv):
    old = sys.argv[:]
    sys.argv = new_argv
    try:
        yield
    finally:
        sys.argv = old

def launch_ddp(module: str, nproc: int, extra_args: list[str] | None = None):
    """
    module: e.g. "seadge.train_psd_model" (the module that has if __name__ == '__main__': main())
    """
    argv = [
        "torchrun",
        "--standalone",                 # single-node convenience (omit if multi-node)
        "--nproc_per_node", str(nproc),
        "--module", module,            # same as: torchrun --module seadge.train_psd_model
    ]
    if extra_args:
        argv += extra_args

    with _argv(argv):
        try:
            dist_main()                # will sys.exit() on completion
        except SystemExit as e:
            if e.code not in (0, None):
                raise

def setup_distributed():
    """
    Initialize distributed training if launched with torchrun.

    Returns:
        use_ddp (bool),
        rank (int),
        world_size (int),
        device (torch.device)
    """
    if "RANK" not in os.environ:
        # Single-process (no torchrun)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, device

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    return True, rank, world_size, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

