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
# [ADD] pixel-level AUC 计算
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

def parse_args(args):
    parser = argparse.ArgumentParser(description="SIDA Model Testing (with OBJ token)")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument("--precision", default="fp16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--test_dataset", default="test", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="sida", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--grad_accumulation_steps", default=10, type=int)
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.00001, type=float)

    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--use_stage1_cls", action="store_true", default=True)
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
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])

    # === NEW: OBJ head arguments ===
    parser.add_argument("--num_obj_classes", type=int, default=81,
                        help="Number of object categories for <OBJ> image-level classification")
    parser.add_argument("--obj_loss_weight", type=float, default=1.0,
                        help="Loss weight for <OBJ> image-level classification head")
    parser.add_argument("--obj_threshold", type=float, default=0.5,
                        help="Threshold for multi-label prediction on OBJ head")
    parser.add_argument("--log_obj_prefix", type=str, default="obj",
                        help="TensorBoard tag prefix for OBJ multi-label metrics")

    # === NEW: Text generation arguments ===
    parser.add_argument("--generate_text", action="store_true", default=False,
                        help="Enable text generation during inference (slower)")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--save_generated_text", action="store_true", default=False,
                        help="Save generated text to a JSON file")
    parser.add_argument("--text_output_file", type=str, default="generated_texts.json",
                        help="Output file for generated texts")

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

    # ----- Tokenizer with [OBJ] -----
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[CLS]")
    tokenizer.add_tokens("[SEG]")
    tokenizer.add_tokens("[OBJ]")  # NEW
    tokenizer.add_tokens("[END]")  # NEW
    args.cls_token_idx = tokenizer("[CLS]", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.obj_token_idx = tokenizer("[OBJ]", add_special_tokens=False).input_ids[0]  # NEW
    args.end_token_idx = tokenizer("[END]", add_special_tokens=False).input_ids[0]  # NEW
    if args.use_mm_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    # ----- Model -----
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
        # NEW: pass OBJ settings
        "obj_token_idx": args.obj_token_idx,
        "num_obj_classes": args.num_obj_classes,
        "obj_loss_weight": args.obj_loss_weight,
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
    for component in ["cls_head", "sida_fc1", "attention_layer", "text_hidden_fcs"]:
        matching_params = [n for n, _ in model.named_parameters() if component in n]
        print(f"{'Found' if matching_params else 'Component not found'} {component}", flush=True)

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

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # ----- LoRA (exclude obj_head like reference) -----
    lora_r = args.lora_r
    if lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (isinstance(module, cls)
                    and all(x not in name for x in [
                        "visual_model", "vision_tower", "mm_projector",
                        "text_hidden_fcs", "cls_head", "sida_fc1",
                        "attention_layer", "obj_head"  # NEW exclude
                    ])
                    and any(x in name for x in lora_target_modules)):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=find_linear_layers(model, args.lora_target_modules.split(",")),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    # Freeze lm_head
    for n, p in model.named_parameters():
        if "lm_head" in n:
            p.requires_grad = False

    # Keep trainable parts aligned (even though we're testing, harmless)
    for n, p in model.named_parameters():
        if any(x in n for x in [
            "embed_tokens", "mask_decoder", "text_hidden_fcs",
            "cls_head", "sida_fc1", "attention_layer", "obj_head"  # NEW include obj_head
        ]):
            p.requires_grad = True

    print("Checking trainable parameters:")
    total_params = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            total_params += p.numel()
    print(f"Total trainable parameters: {total_params}")

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    # ----- Datasets & Loaders -----
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

    if not args.no_test:
        test_dataset = CustomDataset(
            base_image_dir=args.dataset_dir,
            tokenizer=tokenizer,
            vision_tower=args.vision_tower,
            split='validation',
            precision=args.precision,
            image_size=args.image_size,
        )
        print(f"Training with {len(train_dataset)} examples and testing with {len(test_dataset)} examples.")
    else:
        test_dataset = None
        print(f"Training with {len(train_dataset)} examples.")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {"type": "AdamW", "params": {"lr": args.lr, "weight_decay": 0.0, "betas": (args.beta1, args.beta2)}},
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0, "warmup_max_lr": args.lr,
                "warmup_num_steps": 100, "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
            "loss_scale": 0, "initial_scale_power": 12,
            "loss_scale_window": 1000, "min_loss_scale": 1, "hysteresis": 2
        },
        "bf16": {"enabled": args.precision == "bf16"},
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
            obj_token_idx=args.obj_token_idx,  # NEW
        ),
    )

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
        training_data=None,
    )

    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        print(f"resume training from {args.resume}, start from epoch {args.start_epoch}")

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
                obj_token_idx=args.obj_token_idx,  # NEW
            ),
        )

    train_iter = iter(train_loader)

    best_acc, best_score, cur_ciou = 0.0, 0.0, 0.0

    if args.test_only:
        acc, giou, ciou, _ = test(test_loader, model_engine, 0, writer, args, tokenizer)
        sys.exit(0)

    test_epochs = [1, 3, 5, 7, 10]
    if args.local_rank == 0:
        print(f"\nTraining Configuration:")
        print(f"Total epochs: {args.epochs}")
        print(f"test will be performed after epochs: {test_epochs}")

    for epoch in range(args.start_epoch, args.epochs):
        # 简化：只测试逻辑不展示训练实现（与原始一致）
        train_iter = train(train_loader, model_engine, epoch, scheduler, writer, train_iter, args)

        if (epoch + 1) in test_epochs:
            if args.local_rank == 0:
                print(f"\nPerforming test after epoch {epoch + 1}")
            if not args.no_test:
                acc, giou, ciou, _ = test(test_loader, model_engine, epoch, writer, args, tokenizer)
                best_score = max(giou, best_score)
                is_best_iou = giou > best_score
                cur_ciou = ciou if is_best_iou else cur_ciou
                is_best_acc = acc > best_acc
                best_acc = max(acc, best_acc)
                cur_acc = acc if is_best_acc else acc
                is_best = is_best_iou or is_best_acc

            if args.local_rank == 0:
                print(f"Current accuracy: {acc:.2f}%, Best accuracy: {best_acc:.2f}%")
                print(f"Current iou: {cur_ciou:.2f}%, Best score: {best_score:.2f}%")

            if args.no_test or is_best:
                save_dir = os.path.join(args.log_dir, "ckpt_model")
                if args.local_rank == 0:
                    torch.save({"epoch": epoch},
                               os.path.join(args.log_dir, f"meta_log_acc{best_acc:.3f}_iou{best_score:.3f}.pth"))
                    if os.path.exists(save_dir):
                        shutil.rmtree(save_dir)
                torch.distributed.barrier()
                model_engine.save_checkpoint(save_dir)
        else:
            if args.local_rank == 0:
                print(f"Epoch {epoch + 1} completed. Skipping test.")

        if epoch == args.epochs - 1:
            save_dir = os.path.join(args.log_dir, "final_checkpoint")
            if args.local_rank == 0 and os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)
            if args.local_rank == 0:
                print(f"\nTraining completed. Final checkpoint saved to {save_dir}")

def train(train_loader, model, epoch, scheduler, writer, train_iter, args):
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
        for _ in range(args.grad_accumulation_steps):
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
                batch_time.all_reduce(); data_time.all_reduce()
                losses.all_reduce(); cls_losses.all_reduce()
                mask_bce_losses.all_reduce(); mask_dice_losses.all_reduce(); mask_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/cls_loss", cls_losses.avg, global_step)
                writer.add_scalar("train/mask_bce_loss", mask_bce_losses.avg, global_step)
                writer.add_scalar("train/mask_dice_loss", mask_dice_losses.avg, global_step)
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                writer.add_scalar("metrics/total_secs_per_batch", batch_time.avg, global_step)
                writer.add_scalar("metrics/data_secs_per_batch", data_time.avg, global_step)
            batch_time.reset(); data_time.reset()
            losses.reset(); cls_losses.reset()
            mask_bce_losses.reset(); mask_dice_losses.reset(); mask_losses.reset()

        if global_step != 0 and args.local_rank == 0:
            curr_lr = scheduler.get_last_lr()
            writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter


def test(test_loader, model_engine, epoch, writer, args, tokenizer=None, sample_ratio=None):
    model_engine.eval()
    correct = 0; total = 0
    num_classes = 3
    # 放 CPU 足够了
    confusion_matrix = torch.zeros(num_classes, num_classes, device='cpu')
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    # --- OBJ 累加器 ---
    obj_tp_total = 0.0; obj_fp_total = 0.0; obj_fn_total = 0.0
    obj_exact_match_total = 0; obj_rows_total = 0
    obj_tp_per_class = None; obj_fp_per_class = None; obj_fn_per_class = None
    obj_hit1_total = 0; obj_hit5_total = 0; obj_hit_den_total = 0

    # --- 像素级 TP/FP/FN ---
    pix_TP = 0; pix_FP = 0; pix_FN = 0

    # --- 新增：AUC 直方图缓存（常数内存） ---
    BINS = 512
    pos_hist = torch.zeros(BINS, device='cuda', dtype=torch.float64)
    neg_hist = torch.zeros(BINS, device='cuda', dtype=torch.float64)

    # --- NEW: Text generation storage ---
    generated_texts = [] if args.generate_text else None

    # Check tokenizer is provided when text generation is enabled
    if args.generate_text and tokenizer is None:
        raise ValueError("Tokenizer must be provided when generate_text is enabled")

    if args.generate_text and args.local_rank == 0:
        print("\n🔤 Text generation enabled with max_new_tokens =", args.max_new_tokens)

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

        # 删掉 empty_cache（无益反慢）
        # torch.cuda.empty_cache()
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

        # ------ Text generation or standard inference ------
        if args.generate_text:
            # Use evaluate() method for text generation
            from model.llava.constants import IMAGE_TOKEN_INDEX

            with torch.no_grad():
                # Extract base model from DeepSpeed wrapper
                base_model = model_engine.module if hasattr(model_engine, 'module') else model_engine

                output_ids, pred_masks, obj_preds = base_model.evaluate(
                    input_dict["images_clip"],
                    input_dict["images"],
                    input_dict["input_ids"],
                    input_dict["resize_list"],
                    [input_dict["label_list"][i].shape for i in range(len(input_dict["label_list"]))],
                    max_new_tokens=args.max_new_tokens,
                    tokenizer=tokenizer,
                )

                # Decode generated text
                output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
                text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
                text_output = text_output.replace("\n", " ").replace("  ", " ").strip()

                # Store generated text
                generated_texts.append({
                    "image_path": input_dict["image_paths"][0],
                    "generated_text": text_output,
                    "ground_truth_label": cls_labels.item() if hasattr(cls_labels, 'item') else int(cls_labels),
                })

                if batch_idx < 5 and args.local_rank == 0:  # Print first 5 examples
                    print(f"\n[Generated Text {batch_idx}]")
                    print(f"Image: {input_dict['image_paths'][0]}")
                    print(f"Generated: {text_output}")

                # For text generation mode, we need to extract predictions from generated text
                # Parse the CLS token prediction from generated text
                if "[CLS]" in text_output:
                    # Simple heuristic: check which class is mentioned
                    text_lower = text_output.lower()
                    if "real" in text_lower and "synthetic" not in text_lower and "tampered" not in text_lower:
                        pred_class = 0
                    elif "full synthetic" in text_lower or "fully synthetic" in text_lower:
                        pred_class = 1
                    elif "tampered" in text_lower:
                        pred_class = 2
                    else:
                        pred_class = 0  # Default to real
                    preds = torch.tensor([pred_class], device='cuda')
                else:
                    preds = torch.tensor([0], device='cuda')  # Default

                # Create output_dict compatible with downstream code
                output_dict = {
                    "logits": F.one_hot(preds, num_classes=3).float(),
                    "pred_masks": pred_masks if pred_masks else [],
                    "gt_soft_masks": input_dict["soft_masks_list"],
                    "obj_logits": torch.empty((0, args.num_obj_classes), device='cuda'),  # Placeholder
                }
        else:
            # Standard inference without text generation
            input_dict['inference'] = True
            with torch.no_grad():
                output_dict = model_engine(**input_dict)

            # ------ classification ------
            logits = output_dict["logits"]
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        cls_labels = input_dict["cls_labels"]
        correct += (preds == cls_labels).sum().item()
        total += cls_labels.size(0)

        # 混淆矩阵在 CPU 上累计
        for t, p in zip(cls_labels.cpu(), preds.cpu()):
            confusion_matrix[t.long(), p.long()] += 1

        # ------ segmentation（Tampered 才评） ------
        if cls_labels[0] == 2:
            pred_masks = output_dict["pred_masks"]
            masks_list = output_dict["gt_soft_masks"][0].int()
            output_list = (pred_masks[0] > 0).int()
            assert len(pred_masks) == 1

            intersection, union, acc_iou = 0.0, 0.0, 0.0
            for mask_i, output_i in zip(masks_list, output_list):
                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
                )
                intersection += intersection_i
                union += union_i
                acc_iou += intersection_i / (union_i + 1e-5)
                acc_iou[union_i == 0] += 1.0  # no-object target
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
            intersection_meter.update(intersection)
            union_meter.update(union)
            acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

            # ---- 像素级指标 & AUC 直方图累积 ----
            with torch.no_grad():
                pm = pred_masks[0].float()
                if (pm.min() < 0) or (pm.max() > 1.0):
                    pred_scores = torch.sigmoid(pm)   # [num_masks, H, W]
                else:
                    pred_scores = pm.clamp(0, 1)

            pred_bin = (pred_scores >= 0.5).to(torch.int32)

            for mask_i, score_i, bin_i in zip(masks_list, pred_scores, pred_bin):
                m = mask_i.flatten().to(torch.uint8)        # GT
                p = bin_i.flatten().to(torch.uint8)         # Pred bin
                s = score_i.flatten().to(torch.float32)     # score ∈ [0,1]

                TP = (p.eq(1) & m.eq(1)).sum().item()
                FP = (p.eq(1) & m.eq(0)).sum().item()
                FN = (p.eq(0) & m.eq(1)).sum().item()
                pix_TP += TP; pix_FP += FP; pix_FN += FN

                # 直方图计数（常数内存）
                s_clamped = s.clamp_(0, 1)
                bins = torch.clamp((s_clamped * (BINS - 1)).long(), 0, BINS - 1)
                m_bool = (m > 0)
                if m_bool.any():
                    pos_hist.index_add_(0, bins[m_bool], torch.ones_like(bins[m_bool], dtype=torch.float64))
                if (~m_bool).any():
                    neg_hist.index_add_(0, bins[~m_bool], torch.ones_like(bins[~m_bool], dtype=torch.float64))

        # ------ OBJ Multi-label ------
        if ("obj_logits" in output_dict) and ("obj_labels" in input_dict):
            gt = input_dict["obj_labels"]
            if gt.numel() > 0:
                logits_obj = output_dict["obj_logits"]
                probs_obj = logits_obj.sigmoid()
                pred = (probs_obj >= args.obj_threshold).to(gt.dtype)

                gt_bool = (gt > 0).to(torch.bool)
                valid_rows = gt_bool.any(dim=1)
                n_valid = int(valid_rows.sum().item())
                if n_valid > 0:
                    K = gt.shape[1]; k5 = min(5, K)
                    topk_idx = probs_obj.topk(k5, dim=1).indices
                    top1_idx = topk_idx[:, :1]
                    hit1 = (gt_bool.gather(1, top1_idx)).any(dim=1)
                    topk_mask = torch.zeros_like(gt_bool)
                    topk_mask.scatter_(1, topk_idx, True)
                    hit5 = (topk_mask & gt_bool).any(dim=1)
                    obj_hit1_total += int(hit1[valid_rows].sum().item())
                    obj_hit5_total += int(hit5[valid_rows].sum().item())
                    obj_hit_den_total += n_valid

                if obj_tp_per_class is None:
                    K = gt.shape[1]; device = gt.device
                    obj_tp_per_class = torch.zeros(K, device=device, dtype=torch.float64)
                    obj_fp_per_class = torch.zeros(K, device=device, dtype=torch.float64)
                    obj_fn_per_class = torch.zeros(K, device=device, dtype=torch.float64)

                tp = (pred * gt).sum().double()
                fp = (pred * (1 - gt)).sum().double()
                fn = ((1 - pred) * gt).sum().double()
                obj_tp_total += tp.item(); obj_fp_total += fp.item(); obj_fn_total += fn.item()

                exact_match = (pred == gt).all(dim=1).sum().item()
                obj_exact_match_total += exact_match
                obj_rows_total += gt.shape[0]

                obj_tp_per_class += (pred * gt).sum(dim=0).double()
                obj_fp_per_class += (pred * (1 - gt)).sum(dim=0).double()
                obj_fn_per_class += ((1 - pred) * gt).sum(dim=0).double()

    # -------- reduce + metrics --------
    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    # 像素 TP/FP/FN 规约
    if args.distributed:
        _pix = torch.tensor([pix_TP, pix_FP, pix_FN], dtype=torch.float64, device='cuda')
        dist.all_reduce(_pix, op=dist.ReduceOp.SUM)
        pix_TP, pix_FP, pix_FN = _pix.tolist()

    # 派生像素 P/R/F1
    pixel_precision = pix_TP / (pix_TP + pix_FP + 1e-12) if (pix_TP + pix_FP) > 0 else 0.0
    pixel_recall    = pix_TP / (pix_TP + pix_FN + 1e-12) if (pix_TP + pix_FN) > 0 else 0.0
    pixel_f1        = (2 * pixel_precision * pixel_recall / (pixel_precision + pixel_recall + 1e-12)
                       if (pixel_precision + pixel_recall) > 0 else 0.0)

    # 直方图规约并计算 AUC
    if args.distributed:
        dist.all_reduce(pos_hist, op=dist.ReduceOp.SUM)
        dist.all_reduce(neg_hist, op=dist.ReduceOp.SUM)

    if args.local_rank == 0 and (pos_hist.sum() + neg_hist.sum()) > 0:
        pos_cum = torch.cumsum(pos_hist.flip(0), dim=0)
        neg_cum = torch.cumsum(neg_hist.flip(0), dim=0)
        tp = pos_cum; fp = neg_cum
        P = pos_cum[-1]; N = neg_cum[-1]
        fn = P - tp; tn = N - fp

        precision = tp / (tp + fp + 1e-12)
        recall    = tp / (tp + fn + 1e-12)

        dr = recall[:-1] - recall[1:]
        pixel_pr_auc = torch.sum(precision[1:] * dr).item()

        fpr = fp / (fp + tn + 1e-12)
        tpr = recall
        df = fpr[1:] - fpr[:-1]
        pixel_roc_auc = torch.sum((tpr[1:] + tpr[:-1]) * 0.5 * df).item()
    else:
        pixel_pr_auc = 0.0
        pixel_roc_auc = 0.0

    # OBJ 累加器规约
    if args.distributed:
        tot = torch.tensor(
            [obj_tp_total, obj_fp_total, obj_fn_total,
             obj_exact_match_total, obj_rows_total,
             obj_hit1_total, obj_hit5_total, obj_hit_den_total],
            dtype=torch.float64, device='cuda'
        )
        dist.all_reduce(tot, op=dist.ReduceOp.SUM)
        (obj_tp_total, obj_fp_total, obj_fn_total,
         obj_exact_match_total, obj_rows_total,
         obj_hit1_total, obj_hit5_total, obj_hit_den_total) = tot.tolist()

        if obj_tp_per_class is not None:
            for t in (obj_tp_per_class, obj_fp_per_class, obj_fn_per_class):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)

    # OBJ 指标
    obj_micro_prec = obj_tp_total / (obj_tp_total + obj_fp_total + 1e-12) if (obj_tp_total + obj_fp_total) > 0 else 0.0
    obj_micro_rec  = obj_tp_total / (obj_tp_total + obj_fn_total + 1e-12) if (obj_tp_total + obj_fn_total) > 0 else 0.0
    obj_micro_f1   = (2 * obj_micro_prec * obj_micro_rec / (obj_micro_prec + obj_micro_rec + 1e-12)) if (obj_micro_prec + obj_micro_rec) > 0 else 0.0
    obj_subset_acc = (obj_exact_match_total / obj_rows_total) if obj_rows_total > 0 else 0.0
    obj_top1 = (obj_hit1_total / obj_hit_den_total * 100.0) if obj_hit_den_total > 0 else 0.0
    obj_top5 = (obj_hit5_total / obj_hit_den_total * 100.0) if obj_hit_den_total > 0 else 0.0

    if obj_tp_per_class is not None:
        prec_c = obj_tp_per_class / (obj_tp_per_class + obj_fp_per_class + 1e-12)
        rec_c  = obj_tp_per_class / (obj_tp_per_class + obj_fn_per_class + 1e-12)
        f1_c   = (2 * prec_c * rec_c / (prec_c + rec_c + 1e-12))
        obj_macro_prec = float(prec_c.mean().item())
        obj_macro_rec  = float(rec_c.mean().item())
        obj_macro_f1   = float(f1_c.mean().item())
    else:
        obj_macro_prec = obj_macro_rec = obj_macro_f1 = 0.0

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1] if len(iou_class) > 1 else 0.0
    giou = acc_iou_meter.avg[1] if len(acc_iou_meter.avg) > 1 else 0.0

    accuracy = correct / total * 100.0
    class_names = ['Real', 'Full Synthetic', 'Tampered']
    per_class_metrics = {}
    cm = confusion_matrix  # CPU tensor
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        total_class_samples = cm[i, :].sum()
        class_accuracy = float(tp / total_class_samples) if total_class_samples > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class_metrics[class_names[i]] = {
            'accuracy': class_accuracy, 'precision': precision, 'recall': recall, 'f1': f1
        }

    pixel_correct = intersection_meter.sum[1]
    pixel_total = union_meter.sum[1]
    pixel_accuracy = pixel_correct / (pixel_total + 1e-10) * 100.0

    iou = ciou
    f1_score = 2 * (iou * accuracy / 100) / (iou + accuracy / 100 + 1e-10) if (iou + accuracy / 100) > 0 else 0.0
    avg_precision = np.mean([metrics['precision'] for metrics in per_class_metrics.values()])
    avg_recall = np.mean([metrics['recall'] for metrics in per_class_metrics.values()])
    auc_approx = avg_precision * avg_recall

    # --- NEW: Save generated texts if enabled ---
    if args.generate_text and args.save_generated_text and generated_texts and args.local_rank == 0:
        import json
        output_path = os.path.join(args.log_dir, args.text_output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(generated_texts, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Generated texts saved to: {output_path}")
        print(f"Total texts generated: {len(generated_texts)}")

    if args.local_rank == 0 and writer is not None:
        writer.add_scalar("test/accuracy", accuracy, epoch)
        writer.add_scalar("test/giou", giou, epoch)
        writer.add_scalar("test/ciou", ciou, epoch)
        writer.add_scalar("test/pixel_accuracy", pixel_accuracy, epoch)
        writer.add_scalar("test/iou", iou, epoch)
        writer.add_scalar("test/f1_score", f1_score, epoch)
        writer.add_scalar("test/auc_approx", auc_approx, epoch)
        writer.add_scalar("test/pixel_precision",  pixel_precision, epoch)
        writer.add_scalar("test/pixel_recall",     pixel_recall,    epoch)
        writer.add_scalar("test/pixel_f1",         pixel_f1,        epoch)
        writer.add_scalar("test/pixel_pr_auc",     pixel_pr_auc,    epoch)
        writer.add_scalar("test/pixel_roc_auc",    pixel_roc_auc,   epoch)

        pfx = args.log_obj_prefix
        writer.add_scalar(f"test/{pfx}_micro_precision", obj_micro_prec, epoch)
        writer.add_scalar(f"test/{pfx}_micro_recall",    obj_micro_rec, epoch)
        writer.add_scalar(f"test/{pfx}_micro_f1",        obj_micro_f1, epoch)
        writer.add_scalar(f"test/{pfx}_subset_acc",      obj_subset_acc, epoch)
        writer.add_scalar(f"test/{pfx}_macro_precision", obj_macro_prec, epoch)
        writer.add_scalar(f"test/{pfx}_macro_recall",    obj_macro_rec, epoch)
        writer.add_scalar(f"test/{pfx}_macro_f1",        obj_macro_f1, epoch)
        writer.add_scalar(f"test/{pfx}_top1_acc", obj_top1, epoch)
        writer.add_scalar(f"test/{pfx}_top5_acc", obj_top5, epoch)

        test_type = "Full" if sample_ratio is None else f"Sampled ({sample_ratio*100}%)"
        print(f"\n{test_type} test Results:")
        print(f"giou: {giou:.4f}, ciou: {ciou:.4f}")
        print(f"Pixel Precision: {pixel_precision:.4f}")
        print(f"Pixel Recall:    {pixel_recall:.4f}")
        print(f"Pixel F1:        {pixel_f1:.4f}")
        # print(f"Pixel PR-AUC:    {pixel_pr_auc:.4f}")
        print(f"Pixel ROC-AUC:   {pixel_roc_auc:.4f}")
        print(f"Classification Accuracy: {accuracy:.4f}%")
        print("\n[OBJ] Multi-Label Metrics:")
        print(f"  threshold: {args.obj_threshold:.2f}")
        print(f"  micro  - P: {obj_micro_prec:.4f}, R: {obj_micro_rec:.4f}, F1: {obj_micro_f1:.4f}")
        print(f"  macro  - P: {obj_macro_prec:.4f}, R: {obj_macro_rec:.4f}, F1: {obj_macro_f1:.4f}")
        print(f"  subset - Acc: {obj_subset_acc:.4f}")
        print(f"[OBJ] Top-1 Acc (Hit@1): {obj_top1:.4f}%")
        print(f"[OBJ] Top-5 Acc (Hit@5): {obj_top5:.4f}%")
        print(f"Pixel Accuracy: {pixel_accuracy:.4f}%")
        print(f"IoU: {iou:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        print(f"Approximate AUC: {auc_approx:.4f}")
        # print(f"Total correct classifications: {correct}")
        # print(f"Total classification samples: {total}\n")
        # print("Per-Class Metrics:")
        # for class_name, metrics in per_class_metrics.items():
        #     print(f"\n{class_name}:")
        #     print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        #     print(f"  Precision: {metrics['precision']:.4f}")
        #     print(f"  Recall:    {metrics['recall']:.4f}")
        #     print(f"  F1 Score:  {metrics['f1']:.4f}")
        # print("\nConfusion Matrix:\nPredicted \nActual ")
        # print(f"{'':20}", end="")
        # for name in class_names:
        #     print(f"{name:>12}", end="")
        # print()
        # for i, class_name in enumerate(class_names):
        #     print(f"{class_name:20}", end="")
        #     for j in range(num_classes):
        #         print(f"{cm[i, j]:12.0f}", end="")
        #     print()

    return accuracy, giou, ciou, per_class_metrics



if __name__ == "__main__":
    main(sys.argv[1:])