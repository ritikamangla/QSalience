import torch
import json
import re
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import krippendorff
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM
from sklearn.metrics import f1_score, mean_absolute_error, confusion_matrix
from scipy.stats import spearmanr
import argparse
from typing import List, Tuple
from nltk.tokenize import sent_tokenize
from datasets import Dataset, load_dataset
from typing import List, Tuple
from huggingface_hub import HfFolder
import os
DIR_PATH = os.path.dirname(__file__)
print("Current Working Directory:", os.getcwd())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.set_device(2)

MAX_LENTH = 4096
MAX_LENTH_T5 = 512


def save_scores(predictions, filename):
    with open(filename, 'w') as file:
        for score in predictions:
            file.write(f"{score}\n")

def load_gold_annotations(filename):
    with open(filename, 'r') as file:
        annotations = [int(line.strip()) for line in file]
    return annotations

def load_dataset_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def compute_k_alpha(preds, trues):
    df = pd.DataFrame({'rater1': preds,
                'rater2': trues})

    data = df.T.values.tolist()
    data_tuple = tuple(' '.join(map(str, row)) for row in data)
    reliability_data = [[val for val in coder.split()] for coder in data_tuple]
    unique_values = list(set([val for sublist in reliability_data for val in sublist]))
    unique_values.sort()
    print("krippendorff's alpha is ", krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='ordinal', value_domain=unique_values))

def compute_confusions(preds, trues, class_names=None, model_name=""):
    cm = confusion_matrix(trues, preds)
    print(cm)
    plt.figure(figsize=(10, 7))
    if class_names is not None:
        sns.heatmap(cm, annot=True, fmt='g', xticklabels=class_names, yticklabels=class_names)
    else:
        sns.heatmap(cm, annot=True, fmt='g', xticklabels=sorted(set(trues)), yticklabels=sorted(set(trues)))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    save_name = model_name.split("/")[-1]
    print(f"write to {save_name}_confusion_matrix.png")
    plt.savefig(f"{save_name}_confusion_matrix.png")
  

def evaluate_model(dataset, model, tokenizer, max_length=4096, model_name=""):
    numeric_preds = []
    numeric_labels = []


    for item in dataset:
        system_prompt = "<s>### System:\nBelow is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        input_text = """### Instruction: {instruction}\n\n### Input: {input}\n\n### Response: """.format(instruction=item['instruction'], input=item['input'])
        inputs = tokenizer.encode(system_prompt + input_text, return_tensors='pt', padding=True, truncation=True).to(device)
        outputs = model.generate(inputs, max_length=max_length, do_sample=False)

        for output in outputs:
            output_text = tokenizer.decode(output, skip_special_tokens=True)
            match = re.search(r'Response:[\s\S]*?(\d+)', output_text)
            if match:
                prediction = int(match.group(1))
            else:
                prediction = -1  # Default value for cases where the pattern is not found

            numeric_preds.append(prediction)
            actual = int(item['output'])
            numeric_labels.append(actual)

    macro_f1 = f1_score(numeric_labels, numeric_preds, average='macro')
    mae = mean_absolute_error(numeric_labels, numeric_preds)
    spearman_corr, _ = spearmanr(numeric_labels, numeric_preds)
    compute_confusions(numeric_preds, numeric_labels, model_name=model_name)
    k_alpha = compute_k_alpha(numeric_preds, numeric_labels)
    result = {
        "mae": round(mae, 4),
        "spearman_corr": round(spearman_corr, 4),
        "macro_f1": round(macro_f1, 4),
    }

    print("result", result)

    df = pd.DataFrame({"prediction": numeric_preds, "label": numeric_labels})
    save_name = model_name.split("/")[-1]
    df.to_csv(f"{save_name}_predictions.csv", index=False)
    return result



def merge_qlora_model(model_name, qlora_model):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(qlora_model)
    model = PeftModel.from_pretrained(base_model, qlora_model)
    return model, tokenizer



def evaluate_T5(model_name = "lingchensanwen/t5_model_1st", max_length=512):
    def load_data(model_type: str = "") -> Dataset:

        data = load_dataset('json', data_files={'train': os.path.join(DIR_PATH, '../data/train_val_test/upsampled_train.json'),
                                                'val':  os.path.join(DIR_PATH,'../data/train_val_test/val.json'),
                                                'test':  os.path.join(DIR_PATH,'../data/train_val_test/test.json')} 
                            )
        def process_question_data(examples):
            texts = []
            for input_text in examples['input']:
                text = f"### Question: {input_text['question']}\n\n### Article: {input_text['article']}\n\n"
                texts.append(text)

            return {"text": texts, "label": examples['output']}

        dataset = data.map(process_question_data, batched=True, remove_columns=["input","instruction", "output"])

        return dataset

    dataset = load_data()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    training_args = Seq2SeqTrainingArguments(
        output_dir="t5_model",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        fp16=False, 
        learning_rate=3e-4,
        num_train_epochs=3,
        evaluation_strategy="no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="tensorboard",
        push_to_hub=False,
        hub_token=HfFolder.get_token(),
    )


    def preprocess_function(sample: Dataset, padding: str = "max_length") -> dict:
        inputs = [item for item in sample["text"]]

        model_inputs = tokenizer(
            inputs, max_length=max_length, padding=padding, truncation=True
        )
        labels = tokenizer(
            text_target=sample["label"],
            max_length=max_length,
            padding=padding,
            truncation=True,
        )

        if padding == "max_length":
            labels["input_ids"] = [
                [(la if la != tokenizer.pad_token_id else -100) for la in label]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    def postprocess_text(
        preds: List[str], labels: List[str]
    ) -> Tuple[List[str], List[str]]:
        """helper function to postprocess text"""
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_confusions(preds, trues, class_names=None):
        cm = confusion_matrix(trues, preds)
        plt.figure(figsize=(10, 7))
        if class_names is not None:
            sns.heatmap(cm, annot=True, fmt='g', xticklabels=class_names, yticklabels=class_names)
        else:
            sns.heatmap(cm, annot=True, fmt='g', xticklabels=sorted(set(trues)), yticklabels=sorted(set(trues)))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(f"T5_confusion_matrix.png")


    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        numeric_preds = np.array([int(pred) for pred in decoded_preds])
        numeric_labels = np.array([int(label) for label in decoded_labels])

        macro_f1 = f1_score(numeric_labels, numeric_preds, average='macro')
        mae = mean_absolute_error(numeric_labels, numeric_preds)
        spearman_corr, _ = spearmanr(numeric_labels, numeric_preds)
        compute_confusions(numeric_preds, numeric_labels)
        compute_k_alpha(numeric_preds, numeric_labels)
        result = {
        "mae": round(mae, 3),
        "spearman_corr": round(spearman_corr, 3),
        "macro_f1": round(macro_f1, 3),
    }

        print("result", result)

        df = pd.DataFrame({"prediction": numeric_preds, "label": numeric_labels})
        df.to_csv(f"T5_predictions.csv", index=False)
        return result

    tokenized_dataset = dataset.map(
        preprocess_function, batched=True, remove_columns=["text", "label"]
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        compute_metrics=compute_metrics,
    )

    trainer.evaluate(eval_dataset = tokenized_dataset["test"])

def main():
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--model_name', type=str, help='The model name to evaluate')

    args = parser.parse_args()


    if args.model_name is not None:
        print("loading model from", args.model_name)
        model_name = args.model_name 
        test_dataset = load_dataset_from_json(os.path.join(DIR_PATH,"../data/train_val_test/test.json"))
        if model_name == "mistral-ins":
            model, tokenizer = merge_qlora_model("mistralai/Mistral-7B-Instruct-v0.2", "lingchensanwen/mistral-ins-generation-best-balanced")
            evaluate_model(test_dataset, model, tokenizer, max_length=4096, model_name=model_name)

        elif model_name == "llama2-chat":
            model, tokenizer = merge_qlora_model("meta-llama/Llama-2-7b-chat-hf", "lingchensanwen/llama2-chat-generation-best-balanced")
            evaluate_model(test_dataset, model, tokenizer, max_length=4096,model_name=model_name)
        
        elif model_name == "tiny-llama":
            model, tokenizer = merge_qlora_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "lingchensanwen/tiny-llama-generation-best-balanced-new")
            evaluate_model(test_dataset, model, tokenizer, max_length=4096,model_name=model_name)

        elif model_name == "t5":
            evaluate_T5("lingchensanwen/t5_model_1st", max_length=512)
        else:
            print("Model not found, Please check model name")

if __name__=="__main__": 
    main()
