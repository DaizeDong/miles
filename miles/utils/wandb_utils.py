import logging
import os
from copy import deepcopy

import wandb

logger = logging.getLogger(__name__)


def _is_offline_mode(args) -> bool:
    """Detect whether W&B should run in offline mode.

    Priority order:
    1) args.wandb_mode if provided
    2) WANDB_MODE environment variable
    """
    if args.wandb_mode:
        return args.wandb_mode == "offline"
    return os.environ.get("WANDB_MODE") == "offline"


def _resolve_wandb_key(args):
    if args.wandb_key:
        return args.wandb_key
    return os.environ.get("WANDB_API_KEY")


def _resolve_wandb_group(args):
    return args.wandb_group or args.wandb_project or "miles"


def _resolve_wandb_run_name(args, group):
    run_name = getattr(args, "wandb_run_name", None)
    if run_name:
        return run_name

    if args.wandb_random_suffix:
        return f"{group}_{wandb.util.generate_id()}-RANK_{args.rank}"

    return group


def init_wandb_primary(args):
    if not args.use_wandb:
        args.wandb_run_id = None
        return

    # Set W&B mode if specified (overrides WANDB_MODE env var)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_mode == "offline":
            logger.info("W&B offline mode enabled. Data will be saved locally.")
        elif args.wandb_mode == "disabled":
            logger.info("W&B disabled mode enabled. No data will be logged.")
        elif args.wandb_mode == "online":
            logger.info("W&B online mode enabled. Data will be uploaded to cloud.")

    offline = _is_offline_mode(args)

    # Only perform explicit login when NOT offline
    wandb_key = _resolve_wandb_key(args)
    if (not offline) and wandb_key is not None:
        wandb.login(key=wandb_key, host=args.wandb_host)

    # Keep the W&B group stable and only vary the run name.
    # This matches the documented behavior of --disable-wandb-random-suffix,
    # and allows multiple runs of the same model to aggregate under one group.
    group = _resolve_wandb_group(args)
    run_name = _resolve_wandb_run_name(args, group)
    args.wandb_group_resolved = group
    args.wandb_run_name = run_name

    # Prepare wandb init parameters
    init_kwargs = {
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group,
        "name": run_name,
        "config": _compute_config_for_logging(args),
    }

    # Configure settings based on offline/online mode
    if offline:
        init_kwargs["settings"] = wandb.Settings(mode="offline")
    else:
        init_kwargs["settings"] = wandb.Settings(mode="shared", x_primary=True)

    # Add custom directory if specified
    if args.wandb_dir:
        # Ensure directory exists to avoid backend crashes
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
        logger.info(f"W&B logs will be stored in: {args.wandb_dir}")

    wandb.init(**init_kwargs)
    logger.info(
        "Initialized primary W&B run: project=%s group=%s name=%s id=%s",
        args.wandb_project,
        group,
        run_name,
        wandb.run.id,
    )

    _init_wandb_common()

    # Set wandb_run_id in args for easy access throughout the training process
    args.wandb_run_id = wandb.run.id


def _compute_config_for_logging(args):
    output = deepcopy(args.__dict__)

    whitelist_env_vars = [
        "SLURM_JOB_ID",
        # We may insert more default values here, and may also allow users to configure a whitelist
    ]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}

    return output


# https://docs.wandb.ai/guides/track/log/distributed-training/#track-all-processes-to-a-single-run
def init_wandb_secondary(args, router_addr=None):
    wandb_run_id = getattr(args, "wandb_run_id", None)
    if wandb_run_id is None:
        return

    # Set W&B mode if specified (same as primary)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    offline = _is_offline_mode(args)

    wandb_key = _resolve_wandb_key(args)
    if (not offline) and wandb_key is not None:
        wandb.login(key=wandb_key, host=args.wandb_host)

    group = getattr(args, "wandb_group_resolved", None) or _resolve_wandb_group(args)
    run_name = _resolve_wandb_run_name(args, group)
    args.wandb_group_resolved = group
    args.wandb_run_name = run_name

    # Configure settings based on offline/online mode
    if offline:
        settings_kwargs = dict(mode="offline")
    else:
        settings_kwargs = dict(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
        )

    if args.sglang_enable_metrics and router_addr is not None:
        logger.info(f"Forward SGLang metrics at {router_addr} to WandB.")
        settings_kwargs |= dict(
            x_stats_open_metrics_endpoints={
                "sgl_engine": f"{router_addr}/engine_metrics",
            },
            x_stats_open_metrics_filters={
                "sgl_engine.*": {},
            },
        )

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group,
        "name": run_name,
        "config": args.__dict__,
        "resume": "allow",
        "reinit": True,
        "settings": wandb.Settings(**settings_kwargs),
    }

    # Add custom directory if specified
    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    wandb.init(**init_kwargs)
    logger.info(
        "Initialized secondary W&B run: project=%s group=%s name=%s id=%s",
        args.wandb_project,
        group,
        run_name,
        wandb_run_id,
    )

    _init_wandb_common()


def _init_wandb_common():
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("rollout/step")
    wandb.define_metric("rollout/*", step_metric="rollout/step")
    wandb.define_metric("multi_turn/*", step_metric="rollout/step")
    wandb.define_metric("passrate/*", step_metric="rollout/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("perf/*", step_metric="rollout/step")
