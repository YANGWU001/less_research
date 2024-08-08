#!/usr/bin/env python
# coding=utf-8
import logging
import os
import random
import sys
import time
import functools
import datasets
import torch
import torch.distributed as dist
import transformers
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, HfArgumentParser, Trainer,
                          set_seed)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap,transformer_auto_wrap_policy

from data_formatter import get_training_dataset
from arguments import DataArguments, get_data_statistics, ModelArguments, add_padding_to_tokenizer, TrainingArguments

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
gpus = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
def get_block_class_from_model(model: torch.nn.Module, block_class_name: str) -> torch.nn.Module:
    """Get the class of a block from a model, using the block's class name."""
    for module in model.modules():
        if module.__class__.__name__ == block_class_name:
            return module.__class__
    raise ValueError(f"Could not find block class {block_class_name} in model {model}")
def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args._n_gpu=len(gpus.split(","))
    print(training_args)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")

    # Initialize distributed training
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # Load training dataset
    train_dataset = get_training_dataset(data_args.train_files,
                                         tokenizer=tokenizer,
                                         max_seq_length=data_args.max_seq_length,
                                         sample_percentage=data_args.percentage,
                                         seed=data_args.sample_data_seed)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, torch_dtype=model_args.torch_dtype).to(device)
    add_padding_to_tokenizer(tokenizer)

    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        # if you load lora model and resize the token embeddings, the requires_grad flag is set to True for embeddings
        if isinstance(model, PeftModel):
            model.get_input_embeddings().weight.requires_grad = False
            model.get_output_embeddings().weight.requires_grad = False

    if not isinstance(model, PeftModel) and model_args.lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        logger.info(
            f"Applied LoRA to model."
        )
        model.print_trainable_parameters()

        # for checkpointing
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    get_data_statistics(train_dataset)

    if "dataset" in train_dataset.features:
        train_dataset = train_dataset.remove_columns(
            ["dataset", "id", "messages"])
            

    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(
            f"Sample {index} of the training set: {train_dataset[index]}.")

    model_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable model_params: {model_params}")

    analysis_dataset = None
    if training_args.analysis_mode:
        from data_formatter import get_dataset
        analysis_dataset = get_dataset(training_args.analysis_dataset,
                                       data_dir=data_args.data_dir,
                                       tokenizer=tokenizer,
                                       max_length=data_args.max_seq_length)

    # Wrap model with FSDP
    # fsdp_config = {
    #     'fsdp_transformer_layer_cls_to_wrap': ['LlamaDecoderLayer'],
    #     'fsdp_backward_prefetch': 'backward_pre',
    #     'limit_all_gathers': 'true',
    #     'use_orig_params': 'true',
    #     'min_num_params': 0,
    #     'xla': False,
    #     'xla_fsdp_grad_ckpt': False,
    # }

    wrap_class = get_block_class_from_model(model, training_args.fsdp_config['fsdp_transformer_layer_cls_to_wrap'][0])
    model_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_class},)

    fsdp_config= {
        "auto_wrap_policy": model_auto_wrap_policy,
        "backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    }


    model = FSDP(model, **fsdp_config)

    if dist.is_initialized() and dist.get_rank() == 0:
        print(model)
    elif not dist.is_initialized():
        print(model)
    #print(training_args)
    training_args.remove_unused_columns=False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=analysis_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest")
    )

    # Training
    train_result = trainer.train()
    # trainer.save_model()  # Saves the tokenizer too for easy upload

    # metrics = train_result.metrics

    # metrics["train_samples"] = len(train_dataset)

    # trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)
    # trainer.save_state()
    # print("model saving is done")

    # # remove the full model in the end to save space, only adapter is needed
    # if isinstance(model, PeftModel):
    #     pytorch_model_path = os.path.join(
    #         training_args.output_dir, "pytorch_model_fsdp.bin")
    #     os.remove(pytorch_model_path) if os.path.exists(
    #         pytorch_model_path) else None


if __name__ == "__main__":
    main()
