import torch
import resource
from typing import Optional, Set
import socket
import json
import wandb
import trainers
from omegaconf import OmegaConf, DictConfig
import torch.multiprocessing as mp
import hydra
import os
from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port
import transformers
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel
# from peft.optimizers import create_loraplus_optimizer
from transformers import Trainer
torch.backends.cuda.matmul.allow_tf32 = True

OmegaConf.register_new_resolver(
    "get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))


def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module, reference_model: Optional[nn.Module] = None):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    if 'FSDP' in config.trainer:
        init_distributed(rank, world_size, port=config.fsdp_port)

    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if rank == 0 and config.wandb.enabled:
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(config.local_dirs)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),
            name=config.exp_name,
        )

    TrainerClass = getattr(trainers, config.trainer)
    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(policy, config, config.seed, config.local_run_dir,
                           reference_model=reference_model, rank=rank, world_size=world_size)
    
    trainer.train()
    trainer.save()


@hydra.main(version_base=None, config_path="config", config_name="config_dialogue")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")
    
    if "is_lora" not in config or not config["is_lora"]:
        raise ValueError("Must set is_lora to true in config, or you should use train.py instead")

    if config.eval_every % config.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', config.eval_every -
              config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size

    if 'FSDP' in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print('no FSDP port specified; using open port for FSDP:', free_port)
        config.fsdp_port = free_port
        
    print("Flash attention enabled ", torch.backends.cuda.flash_sdp_enabled())
    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    print('=' * 80)
    print(f'Writing to {socket.gethostname()}:{config.local_run_dir}')
    print('=' * 80)

    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)
    print('building policy')
    model_kwargs = {'device_map': 'balanced'} if config.trainer == 'BasicTrainer' else {}
    policy_dtype = getattr(torch, config.model.policy_dtype)
    print(config.model.name_or_path)
    if config.model.name_or_path[0] != "/":
        config.model.name_or_path = "/home/data_91_d/ligr/model/" + config.model.name_or_path
    
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs,attn_implementation="flash_attention_2")
    if config.model.archive is not None:
        print(
            f'loading pre-trained lora weights')
        policy = PeftModel.from_pretrained(policy, config.model.archive, is_trainable=True)
        # policy.merge_and_unload()
        print('loaded pre-trained weights')
    else:
        print("Initializing Lora weights")
        lora_config = LoraConfig(
            r = 16,
            lora_alpha=32,
            init_lora_weights="gaussian"
        )
        policy = get_peft_model(policy, lora_config, autocast_adapter_dtype=False)
    policy.to(torch.bfloat16)
    policy.print_trainable_parameters()
    
    if config.loss.name == 'dpo':
        disable_dropout(policy)
        print('building reference model')
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=reference_model_dtype, **model_kwargs,attn_implementation="flash_attention_2")
        disable_dropout(reference_model) 
    else:
        reference_model = None

    if 'FSDP' in config.trainer:
        world_size = torch.cuda.device_count()
        print('starting', world_size, 'processes for FSDP training')
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
        mp.spawn(worker_main, nprocs=world_size, args=(
            world_size, config, policy, reference_model), join=True)
    else:
        print('starting single-process worker')
        worker_main(0, 1, config, policy, reference_model)


if __name__ == '__main__':
    main()
