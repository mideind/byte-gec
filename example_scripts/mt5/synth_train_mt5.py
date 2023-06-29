import random

import datasets
import numpy as np
import pandas as pd
from transformers import (
    DataCollatorForSeq2Seq,
    MT5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5Tokenizer,
)


def show_random_elements(dataset, num_examples=5):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])


def remove_none(data_row):
    if data_row[source_lang] is None or data_row[target_lang] is None:
        return False
    if data_row[source_lang] == "" or data_row[target_lang] == "":
        return False
    return True


def convert_to_features(example_batch):
    model_inputs = tokenizer(example_batch["is_err"], truncation=True, padding=False, max_length=max_len)
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch["is_corr"], truncation=True, padding=False, max_length=max_len)
    model_inputs["labels"] = target_encodings["input_ids"]
    return model_inputs


model_checkpoint = "google/mt5-base"
train_file = ""
valid_file = ""
model = MT5ForConditionalGeneration.from_pretrained(model_checkpoint)

tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
source_lang = "is_err"
target_lang = "is_corr"
max_length = 512
max_len = max_length

dataset = datasets.load_dataset(
    "csv", sep="\t", on_bad_lines="skip", data_files={"train": [train_file], "valid": valid_file}
)

dataset = dataset.filter(remove_none)

dataset = dataset.map(convert_to_features, batched=True, num_proc=32, remove_columns=["is_err", "is_corr"])

metric = datasets.load_metric("sacrebleu")
batch_size = 4
model_name = model_checkpoint.split("/")[-1]
num_epochs = 3
learning_rate = 2e-5
gradient_accumulation_steps = 4
steps = 10000

is_model_name = f"{model_name}-synth-{source_lang}-to-{target_lang}-max-len-{max_length}-bz-{batch_size}"
args = Seq2SeqTrainingArguments(
    output_dir=is_model_name,
    overwrite_output_dir=True,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    eval_accumulation_steps=gradient_accumulation_steps,
    weight_decay=0.00,
    log_level="info",
    seed=0,
    dataloader_drop_last=True,
    dataloader_num_workers=16,
    remove_unused_columns=False,
    lr_scheduler_type="linear",
    evaluation_strategy="steps",
    eval_steps=steps,
    do_eval=True,
    warmup_steps=1000,
    predict_with_generate=True,
    generation_max_length=max_length,
    generation_num_beams=1,
    report_to="all",
    save_strategy="steps",
    save_steps=steps,
    save_total_limit=5,
    logging_strategy="steps",
    logging_steps=10,
    logging_first_step=True,
    num_train_epochs=num_epochs,
    push_to_hub=False,
)
#    fp16=True,

#    group_by_length=True,
#    adafactor=True,


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    reference_lens = [np.count_nonzero(label != tokenizer.pad_token_id) for label in labels]

    result["gen_len_ratio"] = np.mean(prediction_lens) / np.mean(reference_lens)

    result = {k: round(v, 4) for k, v in result.items()}
    return result


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained(is_model_name)
