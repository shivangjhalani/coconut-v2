# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.distributed
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from coconut import Coconut
from dataset import (
    get_dataset,
    get_question_latent_dataset,
    get_cot_latent_dataset,
    MyCollator,
)

from tqdm import tqdm
from copy import copy
import itertools
import os, sys
import yaml
import json
import gc
import argparse
import functools
from utils import Config, set_seed
import time
import psutil
import numpy as np


def main():

    parser = argparse.ArgumentParser(description="coconut")
    parser.add_argument("config_file")
    args = parser.parse_args()

    # init distributed environment
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    # load the configuration file
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    if rank == 0:
        print("Config:", config_dict)

    configs = Config(config_dict)
    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)

    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    torch.distributed.barrier()
    cur_ckpts = os.listdir(save_dir)

    # check if the job is preempted and resumed.

    if len(cur_ckpts) > 0 and not configs.only_eval:
        # if there are previous checkpoints, and only_eval is False
        # it means the previous run was preempted and the program is restarted.
        # need to find the latest checkpoint and resume from that.

        if rank == 0:
            print(
                f"Warning: found previous run and gonna resume from that. the inputted `resume` argument is ignored!"
            )

        checkpoints = [f for f in cur_ckpts if f.startswith("checkpoint_")]
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))

        # Get the last item in the sorted list
        latest_checkpoint = checkpoints[-1] if checkpoints else None
        configs.resume = int(latest_checkpoint.split("_")[1])
        load_dir = os.path.join(configs.save_path, configs.name, latest_checkpoint)

        configs.load_model_path = load_dir
        print(f"Loading from previous run epoch_{configs.resume}!")

    elif configs.resume != 0:
        # by setting `resume`, we can skip a few epoches at the beginning.
        if configs.load_model_path == "None":
            print(
                f"Warning: you want to skip the first {configs.resume} but you are not loading any existing checkpoint!"
            )
            # not an intended use case at this point
        print(
            f"Loading from {configs.load_model_path} and skip the first {configs.resume} epochs"
        )

    model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    loaded = False

    if configs.load_model_path != "None":
        saved_weights = torch.load(
            configs.load_model_path, map_location=torch.device(rank)
        )

        if configs.coconut and not any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            # we are loading a base model into coconut model
            # e.g., for GSM8k, we used a SFTed model to skip the stage 0
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

        elif not configs.coconut and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            raise ValueError("Cannot load coconut model weights into a causallm model")

        elif configs.coconut and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            # loading from preempted run
            # will handle later
            pass

        else:
            # resume or evaluate sft model
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        # if we need new tokens, initialize their embeddings and lm heads
        model.resize_token_embeddings(len(tokenizer))
        embeddings = model.get_input_embeddings()
        target_id = tokenizer.convert_tokens_to_ids("<<")
        # initialize the new token embeddings with a known token
        # it helps stablize the training
        for token_id in [latent_id, start_id, end_id]:
            target_embedding = embeddings.weight.data[token_id]
            embeddings.weight.data[token_id] = target_embedding
            # The input embeddings and lm heads are tied in GPT2. So the code below is not necessary
            lm_head = model.lm_head
            lm_head.weight.data[token_id] = lm_head.weight.data[target_id]

    if configs.no_thoughts:
        configs.c_thought = 0
        configs.coconut = False

    if configs.coconut:
        model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    if configs.load_model_path != "None" and not loaded:
        print(model.load_state_dict(saved_weights, strict=False))

    print(f"Running FSDP on rank = {rank}, world size = {world_size}")
    model = model.to(rank)

    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            # GPT2Block,       # for GPT2, we don't need to shard layers (it becomes DDP)
            LlamaDecoderLayer  # only shard llama's layers.
        },
    )

    if configs.bf16:
        model.to(torch.bfloat16)

    # if only eval, use ddp (to avoid bugs in fsdp)
    if configs.only_eval:
        parallel_model = DDP(model, device_ids=[rank])

    else:
        parallel_model = FSDP(
            model, auto_wrap_policy=llama_auto_wrap_policy, device_id=rank
        )

    del model

    if rank == 0:
        print(parallel_model)

    # prepare the ground truth answer and cot for evaluation
    question_val = [d["question"] for d in json.load(open(configs.val_path))]
    answers_val = [
        d["answer"].replace(",", "").strip() for d in json.load(open(configs.val_path))
    ]
    cot_val = ["\n".join(d["steps"]) for d in json.load(open(configs.val_path))]

    base_dataset_valid = get_dataset(
        configs.val_path, tokenizer, max_size=32 if configs.debug else 100000000
    )

    if not configs.only_eval:
        base_dataset_train = get_dataset(
            configs.train_path, tokenizer, max_size=5000 if configs.debug else 100000000
        )

    if "gsm" in configs.val_path:
        max_new_tokens = 64
    else:
        max_new_tokens = 128

    total_train_steps = 0

    if not configs.debug and not configs.only_eval and rank == 0:
        wandb_run = wandb.init(project=configs.project, name=configs.name)
        wandb_run.config.update(configs, allow_val_change=True)
        text_table = wandb.Table(columns=["step", "text"])

    else:
        wandb_run = None

    if configs.reset_optimizer:
        optimizer = None

    else:
        optimizer = optim.AdamW(
            parallel_model.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )

    best_acc = 0

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    for epoch in range(configs.resume, configs.num_epochs):

        scheduled_stage = (
            0 if (configs.cot or configs.no_cot) else epoch // configs.epochs_per_stage
        )
        dataset_gen_val = get_question_latent_dataset(
            scheduled_stage,
            base_dataset_valid,
            configs,
            start_id,
            latent_id,
            end_id,
            no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
        )

        valid_gen_dataloader = torch.utils.data.DataLoader(
            dataset_gen_val,
            num_workers=1,
            pin_memory=True,
            batch_size=1,
            collate_fn=collator,
            sampler=DistributedSampler(dataset_gen_val, shuffle=False),
        )

        if not configs.only_eval:

            dataset_train = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_train,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
                shuffle=True,
            )

            train_dataloader = torch.utils.data.DataLoader(
                dataset_train,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_train, shuffle=True),
            )

            # the sampler is deterministic even if shuffle is set to True
            # so we have shuffled the dataset when it's constructed (at every epoch).

            dataset_loss_val = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_valid,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
            )

            valid_loss_dataloader = torch.utils.data.DataLoader(
                dataset_loss_val,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_loss_val, shuffle=False),
            )

            if configs.reset_optimizer:
                del optimizer

                optimizer = optim.AdamW(
                    parallel_model.parameters(),
                    lr=configs.lr,
                    weight_decay=configs.weight_decay,
                )

            parallel_model.module.train()

            total_length = len(train_dataloader) // configs.gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )

            # Add to training loop, collect losses per epoch
            # Remove unused variable that was never used
            # epoch_losses = []

            for step, batch in enumerate(train_dataloader):

                if step == 0 and wandb_run and rank == 0:
                    print("logging training data")
                    cur_bs = len(batch["input_ids"])
                    text_str = ""
                    for data_idx in range(cur_bs):
                        for token_idx in range(len(batch["input_ids"][data_idx])):
                            text_str += (
                                str(batch["input_ids"][data_idx][token_idx].item())
                                + " "
                                + str(batch["labels"][data_idx][token_idx].item())
                                + " "
                                + tokenizer.decode(
                                    batch["input_ids"][data_idx][token_idx]
                                )
                                + "\n"
                            )
                        text_str += "====" * 10 + "\n"
                    text_table.add_data(total_train_steps, text_str)
                    # copy the table due to a bug in wandb
                    # https://github.com/wandb/wandb/issues/2981

                    wandb_run.log({"data_table": copy(text_table)})

                total_train_steps += 1
                batch = {
                    key: batch[key].to(rank) for key in batch.keys() if key != "idx"
                }

                outputs = parallel_model(**batch)

                loss = outputs.loss / configs.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(
                    train_dataloader
                ) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(1)

                if wandb_run and rank == 0:
                    # Existing logging
                    log_dict = {
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": (loss.detach().float() * configs.gradient_accumulation_steps).item(),
                    }
                    
                    # NEW: Track loss spikes at stage transitions
                    current_stage = epoch // configs.epochs_per_stage if not (configs.cot or configs.no_cot) else 0
                    if hasattr(configs, 'prev_stage') and current_stage > configs.prev_stage:
                        # Stage transition detected
                        if hasattr(configs, 'prev_epoch_loss'):
                            loss_spike = log_dict["train/loss"] - configs.prev_epoch_loss
                            log_dict["train/loss_spike_at_transition"] = loss_spike
                        configs.prev_stage = current_stage
                    
                    # Store loss for next comparison
                    configs.prev_epoch_loss = log_dict["train/loss"]
                    
                    try:
                        wandb_run.log(log_dict)
                    except Exception as e:
                        print(f"Warning: Failed to log to wandb: {e}")

                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                    f"completed (loss: {round(float(loss.detach().float() * configs.gradient_accumulation_steps), 4)}"
                )
            pbar.close()
            dist.barrier()

            if (
                not configs.save_only_improve
                and not configs.debug
                and not configs.only_eval
            ):
                states = parallel_model.state_dict()
                if rank == 0:
                    torch.save(
                        states, os.path.join(save_dir, f"checkpoint_{epoch + 1}")
                    )
                    print("saving model.")

                dist.barrier()
                del states
                gc.collect()
                torch.cuda.empty_cache()

            # val loss
            total_loss = 0

            with torch.no_grad():
                parallel_model.module.eval()
                for step, batch in enumerate(valid_loss_dataloader):

                    batch = {
                        key: batch[key].to(rank) for key in batch.keys() if key != "idx"
                    }

                    outputs = parallel_model(**batch)
                    loss = outputs.loss
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    total_loss += loss.item() / world_size

                if wandb_run and rank == 0:
                    try:
                        log_dict = {
                            "eval/loss": total_loss / len(valid_loss_dataloader),
                        }
                        wandb_run.log(log_dict)
                        print("eval loss", total_loss / len(valid_loss_dataloader))
                    except Exception as e:
                        print(f"Warning: Failed to log eval loss to wandb: {e}")

        # val generation accuracy
        total_length = len(valid_gen_dataloader)

        pbar = tqdm(
            colour="blue", desc=f"Test Accuracy", total=total_length, dynamic_ncols=True
        )
        cor, cor_cot, total = (
            torch.tensor(0, device=rank),
            torch.tensor(0, device=rank),
            torch.tensor(0, device=rank),
        )

        # Add before the generation loop
        problem_times = []

        # Track reasoning path lengths during evaluation
        path_lengths = []

        with torch.no_grad():
            parallel_model.module.eval()
            for idx, batch in enumerate(valid_gen_dataloader):
                test_idx = batch["idx"][0]

                batch = {
                    k: v.to(rank)
                    for k, v in batch.items()
                    if v != None and k not in ["idx", "position_ids"]
                }
                # https://github.com/huggingface/transformers/issues/32492

                assert len(batch["input_ids"]) == 1
                answer = answers_val[test_idx.cpu().item()]
                answer_cot = cot_val[test_idx.cpu().item()]
                question = question_val[test_idx.cpu().item()]

                total += 1

                # NEW: Time each inference
                start_time = time.time()
                
                # synced_gpus=True in FSDP mode, as we need to keep # forward pass the same on each device
                outputs = parallel_model.module.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    synced_gpus=not configs.only_eval,
                )
                
                # NEW: Record inference time
                inference_time = time.time() - start_time
                problem_times.append(inference_time)

                text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer_output = text_output.split("#")[-1].replace(",", "").strip()
                cot_output = (
                    ("\n".join(text_output.split("\n")[1:])).split("#")[0].strip()
                )

                # Track reasoning path lengths during evaluation
                reasoning_steps = text_output.split('\n')
                reasoning_steps = [step.strip() for step in reasoning_steps if step.strip()]
                path_lengths.append(len(reasoning_steps))

                if idx < 5 and rank == 0:
                    # Existing debug output
                    print(f"Question {test_idx}: Answer = '{answer}' CoT = '{answer_cot}'")
                    print(f"Full output: '{tokenizer.decode(outputs[0])}'")
                    print(f"Extracted Output: '{answer_output}'")

                # NEW: Extract reasoning confidence
                if configs.coconut and wandb_run and rank == 0:
                    try:
                        # Get the probability distribution for each generated token
                        with torch.no_grad():
                            input_ids = outputs[0].unsqueeze(0)
                            model_outputs = parallel_model.module(**{"input_ids": input_ids})
                            logits = model_outputs.logits[0]  # [seq_len, vocab_size]
                            
                            # Calculate confidence as max probability at each step
                            probs = torch.softmax(logits, dim=-1)
                            max_probs = torch.max(probs, dim=-1)[0]
                            
                            # Log average reasoning confidence
                            if len(max_probs) > 1:
                                avg_confidence = max_probs.mean().item()
                                wandb_run.log({"eval/reasoning_confidence_at_each_step": avg_confidence})
                    except Exception as e:
                        print(f"Warning: Failed to log reasoning confidence to wandb: {e}")

                cor += answer_output == answer
                cor_cot += cot_output == answer_cot

                pbar.update(1)
                pbar.set_description(
                    f"Test accuracy: {round(float(cor.detach().float() / total.detach().float()), 2)}"
                )

            pbar.close()
            print(f"Device {rank}: Cor={cor}, CoT={cor_cot}, Total={total}")

        dist.all_reduce(cor_cot, op=dist.ReduceOp.SUM)
        dist.all_reduce(cor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)

        cor_cot = cor_cot.item()
        cor = cor.item()
        total = total.item()
        if rank == 0:
            print(f"Accuracy on validation set: {cor} / {total} = {cor/total}")
            print(f"CoT match on validation set: {cor_cot} / {total} = {cor_cot/total}")
        sys.stdout.flush()

        if wandb_run:
            # Existing logging
            log_dict = {"eval/acc": cor / total, "eval/cot_em": cor_cot / total}
            
            # NEW: Track fixed latent steps inefficiency
            scheduled_stage = (0 if (configs.cot or configs.no_cot) else epoch // configs.epochs_per_stage)
            if configs.coconut:
                fixed_latent_steps = scheduled_stage * configs.c_thought
                log_dict["eval/fixed_latent_steps_used"] = fixed_latent_steps
            
            try:
                wandb_run.log(log_dict)
            except Exception as e:
                print(f"Warning: Failed to log evaluation metrics to wandb: {e}")

        # NEW: Log average inference time
        if wandb_run and rank == 0:
            try:
                if len(problem_times) > 0:
                    avg_inference_time = sum(problem_times) / len(problem_times)
                    wandb_run.log({"eval/inference_time_per_problem": avg_inference_time})
                else:
                    print("Warning: No inference times recorded")
            except Exception as e:
                print(f"Warning: Failed to log inference time to wandb: {e}")

        # After evaluation - log reasoning path lengths
        if wandb_run and rank == 0 and len(path_lengths) > 0:
            try:
                wandb_run.log({
                    "eval/reasoning_path_length_mean": np.mean(path_lengths),
                    "eval/reasoning_path_length_std": np.std(path_lengths),
                    "eval/reasoning_path_length_min": np.min(path_lengths),
                    "eval/reasoning_path_length_max": np.max(path_lengths),
                })
            except Exception as e:
                print(f"Warning: Failed to log reasoning path lengths to wandb: {e}")

        # # Log difficulty-based computation efficiency
        # if wandb_run and rank == 0:
        #     for diff in ["easy", "medium", "hard"]:
        #         if difficulty_stats[diff]["total"] > 0:
        #             accuracy = difficulty_stats[diff]["correct"] / difficulty_stats[diff]["total"]
        #             avg_time = difficulty_stats[diff]["time"] / difficulty_stats[diff]["total"]
        #             efficiency = accuracy / avg_time if avg_time > 0 else 0
        #             wandb_run.log({
        #                 f"eval/accuracy_{diff}": accuracy,
        #                 f"eval/avg_time_{diff}": avg_time,
        #                 f"eval/computation_efficiency_{diff}": efficiency
        #             })

        # COMMENTED OUT: Incomplete functions that depend on missing features
        # These would need proper implementation with access to stage_0_data and full evaluation logic
        
        # def evaluate_stage_retention(model, tokenizer, stage_0_data, device, rank):
        #     """Evaluate retention of stage 0 performance"""
        #     if len(stage_0_data) == 0:
        #         return 0.0
        #     
        #     correct = 0
        #     total = 0
        #     
        #     with torch.no_grad():
        #         model.eval()
        #         for sample in stage_0_data[:50]:  # Test on subset for speed
        #             # Process sample same as main evaluation
        #             # ... (similar to existing evaluation code)
        #             total += 1
        #             if prediction_correct:  # Your logic here
        #                 correct += 1
        #     
        #     return correct / total if total > 0 else 0.0

        # if not configs.only_eval and epoch > 0:
        #     current_stage = epoch // configs.epochs_per_stage if not (configs.cot or configs.no_cot) else 0
        #     if hasattr(configs, 'prev_stage') and current_stage > configs.prev_stage:
        #         # Stage transition occurred, test retention
        #         if hasattr(configs, 'stage_0_data'):
        #             retention_score = evaluate_stage_retention(
        #                 parallel_model.module, tokenizer, configs.stage_0_data, rank, rank
        #             )
        #             if wandb_run and rank == 0:
        #                 wandb_run.log({f"eval/stage_0_retention_after_stage_{current_stage}": retention_score})

        # def evaluate_consistency(model, dataloader, num_runs=3):
        #     """Evaluate consistency across multiple runs"""
        #     all_predictions = []
        #     
        #     for run in range(num_runs):
        #         predictions = []
        #         # ... run evaluation and collect predictions ...
        #         all_predictions.append(predictions)
        #     
        #     # Calculate consistency
        #     consistent_predictions = 0
        #     total_predictions = len(all_predictions[0])
        #     
        #     for i in range(total_predictions):
        #         if all(pred[i] == all_predictions[0][i] for pred in all_predictions):
        #             consistent_predictions += 1
        #     
        #     return consistent_predictions / total_predictions

        # if not configs.only_eval and epoch % 5 == 0:  # Every 5 epochs
        #     consistency_score = evaluate_consistency(parallel_model.module, valid_gen_dataloader)
        #     if wandb_run and rank == 0:
        #         wandb_run.log({"eval/consistency_across_runs": consistency_score})

        if configs.only_eval:
            break

        dist.barrier()
        if (
            cor / total > best_acc
            and configs.save_only_improve
            and not configs.debug
            and not configs.only_eval
        ):
            states = parallel_model.state_dict()

            if rank == 0:
                torch.save(states, os.path.join(save_dir, f"checkpoint_{epoch + 1}"))
                print("saving model.")

            best_acc = cor / total

            dist.barrier()
            del states
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
