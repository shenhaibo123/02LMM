import os
import sys
import math

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from pathlib import Path
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode,
    setup_seed, init_model, SkipBatchSampler, get_device_type, get_best_device,
)
from metrics.probes import ModelProbe
from metrics.tracker import TrainingTracker

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

            if is_main_process():
                tracker.update_grad_clip(total_norm.item() if torch.is_tensor(total_norm) else total_norm, args.grad_clip)
                tracker.tick_speed()
                tracker.step()

        current_loss = loss.item() * args.accumulation_steps
        current_ppl = math.exp(min(current_loss, 20))

        if is_main_process():
            tracker.update_loss(current_loss, "train")
            tracker.update_ppl(current_ppl, "train")
            tracker.update_lr(lr)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb:
                wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

            if is_main_process():
                tracker.log_step()

        # 每 probe_interval 步计算一次探针指标
        if is_main_process() and args.probe_interval > 0 and step % args.probe_interval == 0:
            probe.activate()
            with torch.no_grad(), autocast_ctx:
                probe_res = model(input_ids, labels=labels)
            dist_metrics = probe.compute_output_distribution(probe_res.logits, labels)
            repr_metrics = probe.compute_representation_diversity()
            tracker.log_probe_metrics(dist_metrics, repr_metrics)
            probe.deactivate()

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir=ckp_dir)
            model.train()
            del state_dict

        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K Pretraining")
    parser.add_argument("--save_dir", type=str, default=str(PROJECT_ROOT / "out"), help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="auto", help="训练设备（auto / cuda:0 / mps / cpu）")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument("--probe_interval", type=int, default=500, help="探针指标计算间隔（0=禁用）")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--num_attention_heads', default=8, type=int, help="注意力头数")
    parser.add_argument('--num_key_value_heads', default=2, type=int, help="KV头数（GQA）")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument("--data_path", type=str, default=str(PROJECT_ROOT / "dataset" / "pretrain_hq.jsonl"), help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="K-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速")
    parser.add_argument("--log_dir", type=str, default=str(PROJECT_ROOT / "logs" / "pretrain"), help="指标 JSONL 与曲线图输出目录")
    args = parser.parse_args()

    # ========== 1. 设备选择 ==========
    if args.device == "auto":
        args.device = get_best_device()
    device_type = get_device_type(args.device)
    Logger(f'Using device: {args.device} (type={device_type})')

    # ========== 2. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
        device_type = "cuda"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 3. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    ckp_dir = str(PROJECT_ROOT / "checkpoints")
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               num_attention_heads=args.num_attention_heads, num_key_value_heads=args.num_key_value_heads,
                               use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir=ckp_dir) if args.from_resume == 1 else None

    # ========== 4. 混合精度：CUDA 用 AMP，MPS/CPU 用 nullcontext ==========
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    if device_type == "cuda":
        autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
    elif device_type == "mps":
        autocast_ctx = torch.amp.autocast(device_type="cpu", dtype=dtype) if dtype == torch.bfloat16 else nullcontext()
    else:
        autocast_ctx = nullcontext()

    # ========== 5. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"K-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 6. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    use_cuda_scaler = (device_type == "cuda" and args.dtype == "float16")
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_scaler) if device_type == "cuda" else torch.cuda.amp.GradScaler(enabled=False)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 7. 初始化训练追踪器和模型探针 ==========
    tracker = TrainingTracker(log_dir=args.log_dir if is_main_process() else None)

    probe = ModelProbe()
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    n_layers = lm_config.num_hidden_layers
    probe_layers = [f"model.layers.{i}" for i in [0, n_layers // 2, n_layers - 1]]
    probe.attach(raw_model, probe_layers)
    Logger(f'ModelProbe attached to: {probe_layers}')

    # ========== 8. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        if 'scaler' in ckp_data:
            scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 9. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 10. 开始训练 ==========
    num_workers = 0 if device_type == "mps" else args.num_workers
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=(device_type == "cuda"))
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 11. 收尾 ==========
    tracker.close(plot=True)
    probe.detach()
    if dist.is_initialized():
        dist.destroy_process_group()
    Logger("训练完成！")
