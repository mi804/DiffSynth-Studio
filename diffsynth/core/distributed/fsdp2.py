from torch.distributed.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

from .context import parallel_context


def resolve_block_types(model):
    """Repeated block class names declared by the model (`_repeated_blocks`)."""
    return tuple(getattr(model, "_repeated_blocks", None) or ())


def find_blocks(model, block_types):
    # Match block class names across the MRO and drop nested matches.
    wanted = set(block_types)
    found = [
        (path, module)
        for path, module in model.named_modules()
        if path and wanted.intersection(base.__name__ for base in type(module).__mro__)
    ]
    return [
        module for path, module in found
        if not any(path.startswith(other + ".") for other, _ in found)
    ]


def apply_fsdp2(model):
    block_types = resolve_block_types(model)
    units = find_blocks(model, block_types)
    if not units:
        raise ValueError(
            f"{type(model).__name__}: no modules matched block types {tuple(block_types)}; "
            "the model declares no repeated blocks or the class names are stale"
        )
    mesh = parallel_context.get_fsdp_mesh()
    # Shard each repeated block, leaving the root module unsharded.
    for block in units:
        fully_shard(block, mesh=mesh, reshard_after_forward=True)


def load_sharded_state_dict(model, state_dict):
    set_model_state_dict(
        model,
        state_dict,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=True),
    )


def move_replicated_to_device(model, device):
    """Move non-sharded (root-level) parameters and buffers onto the compute device."""
    from torch.distributed.tensor import DTensor
    for p in model.parameters():
        if not isinstance(p, DTensor):
            p.data = p.data.to(device)
    for b in model.buffers():
        if not isinstance(b, DTensor):
            b.data = b.data.to(device)
