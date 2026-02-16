import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers.trainer_utils import get_last_checkpoint



print("PYTHON:", os.sys.executable)
print("TORCH:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU: None (CPU)")

try:
    torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass



MODEL_NAME = "t5-small"

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "training_data.jsonl"

MAX_INPUT_LEN = 312
MAX_TARGET_LEN = 128

OUTPUT_DIR = BASE_DIR / "t5_chess_commentary"

PER_DEVICE_TRAIN_BS = 12
GRAD_ACCUM_STEPS = 1
PER_DEVICE_EVAL_BS = 8

LOGGING_STEPS = 200
SAVE_STEPS = 10000
SAVE_TOTAL_LIMIT = 3

LABEL_SMOOTHING = 0.03
NUM_EPOCHS = 1
VAL_FRACTION = 0.001   # 0.1%



def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"DATA_PATH not found: {DATA_PATH}")

    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    # load dataset
    ds_all = load_dataset("json", data_files={"train": str(DATA_PATH)})["train"]
    ds_split = ds_all.train_test_split(test_size=VAL_FRACTION, seed=42)
    train_ds = ds_split["train"]
    val_ds = ds_split["test"]

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["input"],
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_ds.map(
        preprocess,
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train",
    )
    tokenized_val = val_ds.map(
        preprocess,
        batched=True,
        remove_columns=val_ds.column_names,
        desc="Tokenizing val",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        overwrite_output_dir=False,   # allow resume

        # core training
        per_device_train_batch_size=PER_DEVICE_TRAIN_BS,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BS,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,

        learning_rate=3e-4,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=0.05,
        weight_decay=0.01,

        group_by_length=True,
        label_smoothing_factor=LABEL_SMOOTHING,

        # eval/log/save cadence
        eval_strategy="no",
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        save_safetensors=True,
        predict_with_generate=False,
        ignore_data_skip=True,
        logging_first_step=True,

        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none",

        load_best_model_at_end=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # auto resume if checkpoint exists
    last_ckpt = get_last_checkpoint(str(OUTPUT_DIR))
    if last_ckpt is not None:
        print(f"\nResuming from checkpoint: {last_ckpt}\n")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        print("\nStarting fresh training (no checkpoint found).\n")
        trainer.train()

    # save final
    final_dir = OUTPUT_DIR / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nDone. Saved to {final_dir}\n")


if __name__ == "__main__":
    main()
