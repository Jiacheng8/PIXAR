import argparse
import os
import shutil
import sys
import time
from functools import partial
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from model.SIDA import SIDAForCausalLM
from model.llava import conversation as conversation_lib
from utils.SID_Set import collate_fn, CustomDataset
from utils.batch_sampler import BatchSampler
import torch.distributed as dist
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)
import random
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

# 新增：用于保存图像 / 掩码
from PIL import Image
import torchvision.transforms as T

def parse_args(args):
    parser = argparse.ArgumentParser(description="SIDA Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--test_dataset", default="test", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="sida", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.00001, type=float)

    parser.add_argument("--num_classes", type=int, default=3,
                       help="Number of classes for classification in stage 1")
    parser.add_argument("--use_stage1_cls", action="store_true", default=True,
                   help="Whether to use Stage 1 CLS token in Stage 2")
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=1.0, type=float)
    parser.add_argument("--bce_loss_weight", default=1.0, type=float)
    parser.add_argument("--cls_loss_weight", default=1.0, type=float)
    parser.add_argument("--mask_loss_weight", default=1.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--no_test", action="store_true", default=False)
    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument("--mask_overlap_pct", type=float, default=10.0,help="视为有重叠的最小覆盖率阈值（交集/GT面积，百分比）。默认10%")
    parser.add_argument("--mask_dilate_px", type=int, default=0,help="在计算覆盖率前对GT做像素级膨胀（容忍位移抖动）。0=不膨胀")
    parser.add_argument("--exp-name", type=str, default="",help="实验名称")

    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )

    return parser.parse_args(args)

def main(args):
    args = parse_args(args)
    deepspeed.init_distributed()
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    num_added_token = tokenizer.add_tokens("[CLS]")
    num_added_token = tokenizer.add_tokens("[SEG]")
    args.cls_token_idx = tokenizer("[CLS]", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "cls_loss_weight": args.cls_loss_weight,
        "mask_loss_weight": args.mask_loss_weight,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "cls_token_idx": args.cls_token_idx,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = SIDAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    print("\nChecking specific components:")
    for component in [ "cls_head", "sida_fc1", "attention_layer", "text_hidden_fcs"]:
        matching_params = [n for n, _ in model.named_parameters() if component in n]
        if matching_params:
            print(f"Found {component} in parameters: {matching_params}")
        else:
            print(f"Component not found: {component}")
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    if not args.test_only:
        model.get_model().initialize_sida_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False

    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                                "cls_head",
                                "sida_fc1",
                                "attention_layer",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))
        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
                model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    model.resize_token_embeddings(len(tokenizer))

    for n, p in model.named_parameters():
        if "lm_head" in n:
            p.requires_grad = False

    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["embed_tokens", "mask_decoder", "text_hidden_fcs","cls_head", "sida_fc1","attention_layer"]
            ]
        ):
            p.requires_grad = True

    print("Checking trainable parameters:")
    total_params = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"Trainable: {n} with {p.numel()} parameters")
            total_params += p.numel()
    print(f"Total trainable parameters: {total_params}")

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    train_dataset = CustomDataset(
        base_image_dir=args.dataset_dir,  
        tokenizer=tokenizer,
        vision_tower=args.vision_tower,  
        split="train", 
        precision=args.precision,  
        image_size=args.image_size, 

    )
    print(f"\nInitializing datasets:")
    print(f"Training split size: {len(train_dataset)}")

    if args.no_test == False:
        test_dataset = CustomDataset(
            base_image_dir=args.dataset_dir, 
            tokenizer=tokenizer,
            vision_tower=args.vision_tower,  
            split="test", 
            precision=args.precision, 
            image_size=args.image_size, 
        )
        print(
            f"Training with {len(train_dataset)} examples and testing with {len(test_dataset)} examples."
        )
    else:
        test_dataset = None
        print(f"Training with {len(train_dataset)} examples.")
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
            "loss_scale": 0,  
            "initial_scale_power": 12,  
            "loss_scale_window": 1000,
            "min_loss_scale": 1,
            "hysteresis": 2
        },
        "gradient_clipping": 1.0,
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    batch_sampler = BatchSampler(
        dataset=train_dataset,
        batch_size=ds_config["train_micro_batch_size_per_gpu"],
        world_size=torch.cuda.device_count(),
        rank=args.local_rank
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
            cls_token_idx=args.cls_token_idx,
        ),
    )
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
        training_data=None, 
    )

    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir,  "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    if test_dataset is not None:
        test_sampler = BatchSampler(
            dataset=test_dataset,
            batch_size=args.test_batch_size,
            world_size=torch.cuda.device_count(),
            rank=args.local_rank
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=partial(
                 collate_fn,
                 tokenizer=tokenizer,
                 conv_type=args.conv_type,
                 use_mm_start_end=args.use_mm_start_end,
                 local_rank=args.local_rank,
             ),
        )

    train_iter = iter(train_loader)

    best_acc, best_score, cur_ciou = 0.0, 0.0, 0.0

    if args.test_only:
        acc, giou, ciou, _ = test(test_loader, model_engine, 0, writer, args) 
        exit()

    test_epochs = [1,3,5,7,10]
    if args.local_rank == 0:
        print(f"\nTraining Configuration:")
        print(f"Total epochs: {args.epochs}")
        print(f"test will be performed after epochs: {test_epochs}")
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )
        if (epoch + 1) in test_epochs: 
            if args.local_rank == 0:
                print(f"\nPerforming test after epoch {epoch + 1}")

            if args.no_test == False:
                acc, giou, ciou, _ = test(test_loader, model_engine, epoch, writer, args)
                best_score = max(giou, best_score)
                is_best_iou = giou > best_score
                cur_ciou = ciou if is_best_iou else cur_ciou
                is_best_acc = acc > best_acc
                best_acc = max(acc, best_acc)
                cur_acc = acc if is_best_acc else cur_acc
                is_best = is_best_iou or is_best_acc

            if args.local_rank == 0:
                print(f"Current accuracy: {acc:.2f}%, Best accuracy: {best_acc:.2f}%")
                print(f"Current iou: {cur_ciou:.2f}%, Best score: {best_score:.2f}%")
            if args.no_test or is_best:
                save_dir = os.path.join(args.log_dir, "ckpt_model")
                if args.local_rank == 0:
                    torch.save(
                                {"epoch": epoch},
                                os.path.join(
                                    args.log_dir,
                                    f"meta_log_acc{best_acc:.3f}_iou{best_score:.3f}.pth"
                                ),
                    )
                    if os.path.exists(save_dir):
                        shutil.rmtree(save_dir)
                torch.distributed.barrier()
                model_engine.save_checkpoint(save_dir)
        else:
            if args.local_rank == 0:
                print(f"Epoch {epoch + 1} completed. Skipping test.")

        if epoch == args.epochs - 1:
            save_dir = os.path.join(args.log_dir, "final_checkpoint")
            if args.local_rank == 0:
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)
            if args.local_rank == 0:
                print(f"\nTraining completed. Final checkpoint saved to {save_dir}")

def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    cls_losses = AverageMeter("ClsLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")
    progress = ProgressMeter(
        args.steps_per_epoch,
        [batch_time, losses, cls_losses, mask_bce_losses, mask_dice_losses, mask_losses],
        prefix="Epoch: [{}]".format(epoch),
    )
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        model.zero_grad()
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)
            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()
            output_dict = model(**input_dict)
            loss = output_dict["loss"]
            cls_loss = output_dict["cls_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]
            losses.update(loss.item(), input_dict["images"].size(0))
            cls_losses.update(cls_loss.item(), input_dict["images"].size(0))
            if input_dict['cls_labels'][0] == 2:
                mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
                mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
                mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
            model.backward(loss)
            model.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()
                losses.all_reduce()
                cls_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/cls_loss", cls_losses.avg, global_step)
                writer.add_scalar("train/mask_bce_loss", mask_bce_losses.avg, global_step)
                writer.add_scalar("train/mask_dice_loss", mask_dice_losses.avg, global_step)
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                writer.add_scalar("metrics/total_secs_per_batch", batch_time.avg, global_step)
                writer.add_scalar("metrics/data_secs_per_batch", data_time.avg, global_step)
            batch_time.reset()
            data_time.reset()
            losses.reset()
            cls_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter

def _safe_to_uint8_image_tensor(img_t: torch.Tensor) -> Image.Image:
    """
    将任意范围的[C,H,W]张量线性归一化为[0,255]并转 PIL.Image
    """
    if img_t.ndim != 3:
        raise ValueError("Expect image tensor with shape [C,H,W]")
    img_t = img_t.detach().cpu().to(torch.float32)
    min_v = float(img_t.min())
    max_v = float(img_t.max())
    if max_v - min_v < 1e-6:
        img_norm = img_t - min_v
    else:
        img_norm = (img_t - min_v) / (max_v - min_v)
    img_uint8 = (img_norm * 255.0).clamp(0, 255).to(torch.uint8)
    to_pil = T.ToPILImage()
    return to_pil(img_uint8)

def _save_mask_png(arr, path):
    """
    将掩码数组保存为二值PNG（>0 为 255）
    """
    try:
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        img = Image.fromarray(((arr > 0).astype('uint8')) * 255)
        img.save(path)
    except Exception as e:
        with open(path + ".err.txt", "w", encoding="utf-8") as f:
            f.write(str(e))

def test(test_loader, model_engine, epoch, writer, args, sample_ratio=None):
    import os
    model_engine.eval()
    correct = 0
    total = 0
    num_classes = 3
    confusion_matrix = torch.zeros(num_classes, num_classes, device='cuda')
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    # === 日志目录（只记录tampered预测正确样本）===
    log_dir = os.path.join(args.log_dir, args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    tp_tampered_log = []
    # === 新增：分类为tampered且mask与GT完全不重叠的样本记录 ===
    tp_tampered_mask_mismatch = []
    # === 新增：GT为tampered但预测为full synthetic 的误判样本 ===
    mis_tampered_as_full = []       # 仅文件名
    mis_tampered_as_full_rows = []  # CSV 行（含概率等）

    total_batches = len(test_loader)
    if sample_ratio is not None:
        num_batches = max(1, int(total_batches * sample_ratio))
        sample_indices = set(random.sample(range(total_batches), num_batches))
        print(f"\ntest on {num_batches}/{total_batches} randomly sampled batches...")

    for batch_idx, input_dict in enumerate(tqdm.tqdm(test_loader)):
        if sample_ratio is not None and batch_idx not in sample_indices:
            continue

        if batch_idx == 0:
            print("\nFirst test batch details:")
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key} shape: {value.shape}")
                elif isinstance(value, list):
                    print(f"{key} length: {len(value)}")

        torch.cuda.empty_cache()
        input_dict = dict_to_cuda(input_dict)

        if total == 0:
            print("\nProcessing first batch:")
            print("Input dict keys:", input_dict.keys())

        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        input_dict['inference'] = True
        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        logits = output_dict["logits"]
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        cls_labels = input_dict["cls_labels"]


        # === 新增：记录 tampered 真阳性文件名（健壮处理 paths） ===
        paths = input_dict.get("image_paths", [])
        for i in range(cls_labels.size(0)):
            gt_i = int(cls_labels[i].item()) if torch.is_tensor(cls_labels[i]) else int(cls_labels[i])
            pred_i = int(preds[i].item()) if torch.is_tensor(preds[i]) else int(preds[i])
            if gt_i == 2 and pred_i == 2:
                if isinstance(paths, list) and i < len(paths):
                    name = os.path.basename(paths[i]) if isinstance(paths[i], str) else str(paths[i])
                elif isinstance(paths, torch.Tensor) and paths.dim() > 0 and i < paths.shape[0]:
                    name = os.path.basename(str(paths[i]))
                else:
                    name = f"sample_{batch_idx}_{i}"
                tp_tampered_log.append(name)
                
                
        correct += (preds == cls_labels).sum().item()
        total += cls_labels.size(0)

        for t, p in zip(cls_labels, preds):
            confusion_matrix[t.long(), p.long()] += 1

        # === IoU统计部分（保持原样）===
        if cls_labels[0] == 2:
            pred_masks = output_dict["pred_masks"]              # 期望形如 [B, M, H, W] 或 list 长度 B
            masks_list = output_dict["gt_masks"][0].int()       # 取第0个样本的 GT mask 列表/张量
            # 兼容 pred_masks 既可能是张量也可能是 list 的情况
            pm0 = pred_masks[0] if isinstance(pred_masks, (list, tuple)) else pred_masks[0]
            output_list = (pm0 > 0).int()
            assert (isinstance(pred_masks, (list, tuple)) and len(pred_masks) == 1) or \
                (isinstance(pred_masks, torch.Tensor) and pred_masks.shape[0] == 1), \
                "expect batch_size=1 for mask evaluation"

            # 本样本 gt / pred（就地计算，避免 NameError）
            gt  = int(cls_labels[0].item() if torch.is_tensor(cls_labels[0]) else cls_labels[0])
            pred = int(preds[0].item() if torch.is_tensor(preds[0]) else preds[0])

            # 判定是否“完全无重叠”
            sample_has_any_overlap = False
            
            # 覆盖率阈值（交/GT）；支持可选膨胀以容忍小位移
            COV_TAU = float(getattr(args, "mask_overlap_pct", 10.0)) / 100.0
            DILATE_PX = int(getattr(args, "mask_dilate_px", 0))

            # 统一 [K,H,W], [M,H,W] 的 bool 张量（K: GT masks, M: Pred masks）
            if isinstance(masks_list, (list, tuple)):
                gt_stack = torch.stack([m.to(torch.bool) for m in masks_list], dim=0)
            else:
                gt_stack = masks_list.to(torch.bool)

            if isinstance(output_list, (list, tuple)):
                pred_stack = torch.stack([o.to(torch.bool) for o in output_list], dim=0)
            else:
                pred_stack = output_list.to(torch.bool)

            # （可选）对 GT 做膨胀，容忍边界/位移抖动
            if DILATE_PX > 0:
                k = 2 * DILATE_PX + 1
                gt_stack_f = gt_stack.float().unsqueeze(1)                          # [K,1,H,W]
                gt_stack_f = F.max_pool2d(gt_stack_f, kernel_size=k, stride=1, padding=DILATE_PX)
                gt_stack = (gt_stack_f.squeeze(1) > 0)

            # 计算 pairwise 交/GT 覆盖率
            # gt: [K,H,W] -> [K,1,H,W]; pred: [M,H,W] -> [1,M,H,W]
            gt_b   = gt_stack.unsqueeze(1)                                          # [K,1,H,W]
            pred_b = pred_stack.unsqueeze(0)                                        # [1,M,H,W]
            inter  = (gt_b & pred_b).sum(dim=(2,3)).float()                         # [K,M]

            gt_area = gt_stack.sum(dim=(1,2)).float().clamp_min(1.0)                # [K]
            coverage_mat = inter / gt_area.unsqueeze(1)                              # [K,M]

            # 只要任一 (gt_k, pred_m) 覆盖率 ≥ 阈值，就认为“有重叠”
            sample_has_any_overlap = bool((coverage_mat >= COV_TAU).any())


            intersection, union, acc_iou = 0.0, 0.0, 0.0
            for mask_i, output_i in zip(masks_list, output_list):
                # if torch.any((mask_i == 1) & (output_i == 1)):
                #     sample_has_any_overlap = True

                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
                )
                intersection += intersection_i
                union += union_i
                acc_iou += intersection_i / (union_i + 1e-5)
                acc_iou[union_i == 0] += 1.0

            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = acc_iou.cpu().numpy() / max(1, masks_list.shape[0])
            intersection_meter.update(intersection)
            union_meter.update(union)
            acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

            # 若分类正确(tampered)且mask完全不重叠，则记录文件名
            if gt == 2 and pred == 2 and not sample_has_any_overlap:
                if isinstance(paths, list) and len(paths) > 0:
                    name = os.path.basename(paths[0]) if isinstance(paths[0], str) else str(paths[0])
                else:
                    name = f"sample_{batch_idx}_0"
                tp_tampered_mask_mismatch.append(name)
            # === 新增：GT=Tampered(2) 但预测=Full Synthetic(1) 的误判记录 ===
            if gt == 2 and pred == 1:
                if isinstance(paths, list) and i < len(paths):
                    name = os.path.basename(paths[i]) if isinstance(paths[i], str) else str(paths[i])
                elif isinstance(paths, torch.Tensor) and paths.dim() > 0 and i < paths.shape[0]:
                    name = os.path.basename(str(paths[i]))
                else:
                    name = f"sample_{batch_idx}_{i}"
                mis_tampered_as_full.append(name)

    # === 写出日志（只包含tampered预测成功且mask有重叠的文件名）===
    # 去重排序
    tp_tampered_log = sorted(set(tp_tampered_log))
    tp_tampered_mask_mismatch = sorted(set(tp_tampered_mask_mismatch))

    # 从 tp_tampered_log 中移除不重叠样本
    tp_tampered_log_filtered = [name for name in tp_tampered_log if name not in tp_tampered_mask_mismatch]
    mis_tampered_as_full = sorted(set(mis_tampered_as_full))

    # 写出“预测正确且mask有重叠”的样本名单
    log_path = os.path.join(log_dir, f"tp_tampered.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        for name in tp_tampered_log_filtered:
            f.write(f"{name}\n")
        for name in mis_tampered_as_full:
            f.write(f"{name}\n")

    # === 写出“分类正确但mask完全不重叠”的样本名单 ===
    mismatch_log_path = os.path.join(log_dir, f"tp_tampered_mask_mismatch.txt")
    with open(mismatch_log_path, "w", encoding="utf-8") as f:
        for name in tp_tampered_mask_mismatch:
            f.write(f"{name}\n")

            
    # === 打印日志统计信息 ===
    num_total_tp = len(set(tp_tampered_log))                      # 总的预测正确（包括重叠 + 不重叠）
    num_mismatch = len(set(tp_tampered_mask_mismatch))            # mask完全不重叠的数量
    num_overlap = num_total_tp - num_mismatch                     # mask有重叠的数量
    ratio_overlap = num_overlap / num_total_tp * 100 if num_total_tp > 0 else 0.0
    ratio_mismatch = num_mismatch / num_total_tp * 100 if num_total_tp > 0 else 0.0

    summary_log_path = os.path.join(log_dir, f"summary.txt")
    with open(summary_log_path, "w", encoding="utf-8") as f:
        f.write("===== Tampered Prediction Summary =====\n")
        f.write(f"Total correctly predicted Tampered samples: {num_total_tp}\n")
        f.write(f" - With mask overlap:      {num_overlap} ({ratio_overlap:.2f}%)\n")
        f.write(f" - Mask completely mismatch: {num_mismatch} ({ratio_mismatch:.2f}%)\n")
        f.write(f"Saved logs:\n  - Overlap list: {log_path}\n  - Mismatch list: {mismatch_log_path}\n")
        f.write("=======================================\n")

    # 同时在控制台打印一份
    print("\n===== Tampered Prediction Summary =====")
    print(f"Total correctly predicted Tampered samples: {num_total_tp}")
    print(f" - With mask overlap:      {num_overlap} ({ratio_overlap:.2f}%)")
    print(f" - Mask completely mismatch: {num_mismatch} ({ratio_mismatch:.2f}%)")
    print("=======================================\n")


    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1] if len(iou_class) > 1 else 0.0
    giou = acc_iou_meter.avg[1] if len(acc_iou_meter.avg) > 1 else 0.0

    accuracy = correct / total * 100.0
    confusion_matrix = confusion_matrix.cpu()
    class_names = ['Real', 'Full Synthetic', 'Tampered']
    per_class_metrics = {}
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        tn = confusion_matrix.sum() - (tp + fp + fn)
        total_class_samples = confusion_matrix[i, :].sum()
        class_accuracy = float(tp / total_class_samples) if total_class_samples > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class_metrics[class_names[i]] = {
            'accuracy': class_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    pixel_correct = intersection_meter.sum[1]
    pixel_total = union_meter.sum[1]
    pixel_accuracy = pixel_correct / (pixel_total + 1e-10) * 100.0

    iou = ciou
    f1_score = 2 * (iou * accuracy / 100) / (iou + accuracy / 100 + 1e-10) if (iou + accuracy / 100) > 0 else 0.0
    avg_precision = np.mean([metrics['precision'] for metrics in per_class_metrics.values()])
    avg_recall = np.mean([metrics['recall'] for metrics in per_class_metrics.values()])
    auc_approx = avg_precision * avg_recall

    if args.local_rank == 0:
        writer.add_scalar("test/accuracy", accuracy, epoch)
        writer.add_scalar("test/giou", giou, epoch)
        writer.add_scalar("test/ciou", ciou, epoch)
        writer.add_scalar("test/pixel_accuracy", pixel_accuracy, epoch)
        writer.add_scalar("test/iou", iou, epoch)
        writer.add_scalar("test/f1_score", f1_score, epoch)
        writer.add_scalar("test/auc_approx", auc_approx, epoch)
        for class_name, metrics in per_class_metrics.items():
         for metric_name, value in metrics.items():
             writer.add_scalar(f"test/{class_name.lower().replace('/', '_')}_{metric_name}", value, epoch)

        test_type = "Full" if sample_ratio is None else f"Sampled ({sample_ratio*100}%)"
        print(f"\n{test_type} test Results:")
        print(f"giou: {giou:.4f}, ciou: {ciou:.4f}")
        print(f"Classification Accuracy: {accuracy:.4f}%")
        print(f"Pixel Accuracy: {pixel_accuracy:.4f}%")
        print(f"IoU: {iou:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        print(f"Approximate AUC: {auc_approx:.4f}")
        print(f"Total correct classifications: {correct}")
        print(f"Total classification samples: {total}")
        print("\nPer-Class Metrics:")
        for class_name, metrics in per_class_metrics.items():
            print(f"\n{class_name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")

        print("\nConfusion Matrix:")
        print("Predicted ")
        print("Actual ")
        print(f"{'':20}", end="")  
        for name in class_names:
            print(f"{name:>12}", end="") 
        print()  

        for i, class_name in enumerate(class_names):
            print(f"{class_name:20}", end="") 
            for j in range(num_classes):
                print(f"{confusion_matrix[i, j]:12.0f}", end="")
            print()  
            
    return accuracy, giou, ciou, per_class_metrics


if __name__ == "__main__":
    main(sys.argv[1:])
