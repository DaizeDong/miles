def get_predictive_actor_train_pass_count(*, role: str, predictive_enabled: bool) -> int:
    return 1


def get_effective_train_iters(*, base_train_iters: int, role: str, predictive_enabled: bool) -> int:
    return base_train_iters


def get_predictive_train_mode_for_step(*, role: str, predictive_enabled: bool, step_id: int) -> str:
    if role == "actor" and predictive_enabled and step_id == 0:
        return "skip"
    return "compute"


def get_rollout_train_step_id(
    *,
    rollout_id: int,
    step_id: int,
    num_steps_per_rollout: int,
    train_pass_index: int = 0,
    num_train_passes: int = 1,
) -> int:
    total_steps_per_rollout = num_steps_per_rollout * num_train_passes
    step_offset = train_pass_index * num_steps_per_rollout
    return rollout_id * total_steps_per_rollout + step_offset + step_id
