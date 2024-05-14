# QSalience



## :star2: Introduction
QSalience introduces the first application-agnostic salience predictor for inquisitive questions. This project is fine-tuned over our newly annotated dataset of 1,766 (context, question) pairs with salience scores. We explore the usefulness of question salience in downstream applications such as TL;DR expansion.

To read more details, please check our paper - [Which questions should I answer? Salience Prediction of Inquisitive Questions](https://arxiv.org/abs/2404.10917)

**Authors:** [Yating Wu*](http://lingchensanwen.github.io), [Ritika Mangla*](https://ritikamangla01.netlify.app), [Alexandros G. Dimakis](https://users.ece.utexas.edu/~dimakis/), [Greg Durrett](https://www.cs.utexas.edu/~gdurrett/), and [Junyi Jessy Li](https://jessyli.com)

If you find our work useful or relevant to your research, please consider citing our paper.

```
@article{wu2024questions,
  title={Which questions should I answer? Salience Prediction of Inquisitive Questions},
  author={Wu, Yating and Mangla, Ritika and Dimakis, Alexandros G and Durrett, Greg and Li, Junyi Jessy},
  journal={arXiv preprint arXiv:2404.10917},
  year={2024}
}
```

## Salience

### Data Availability
Datasets are organized as follows:
- General data: [salience](./data/salience)
- Training, validation, and testing sets: [train_val_test](./data/train_val_test)

### Installation and Usage
- We provide a [quick colab running code](https://colab.research.google.com/drive/1MmZ_M7FOBcotf22j98Ov5ADsqCFaQEYz?usp=sharing) for usage.

- Go to QSalience/code folder 

- Please login into huggingface before running the code as ```mistralai/Mistral-7B-Instruct-v0.2``` now requires you to approve their policy.

1. Install necessary packages:
###### Depends on your dependency package version, you may expect very minor difference on the generation.

```
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
pip install -q datasets scipy evaluate peft scikit-learn torch transformers wandb
pip install -q trl
pip install krippendorff
```

- Go to code/preprocess folder

2. Provide your text file then predict salience of your question:
   
Provide your question_csv file and article_txt file and replace the path in following commend. Make sure question_csv has question and sentence_id columns.

```
python preprocess.py --question_csv_path="example_question.csv" --article_txt_path="example_article.txt"
```

- Then you will generate a example.json file

- Go back to code folder and run this to predict salience score

```
CUDA_VISIBLE_DEVICES="" python predict_salience.py --model_name="MODEL_NAME" --input_file="preprocess/example.json" 
```

Replace `MODEL_NAME` with one of the following:
  - `mistral-ins`
  - `llama2-chat`
  - `t5`
  - `tiny-llama`

3. Your prediction will be on MODEL_NAME_prediction.csv

4. (Optional) To run the evaluation script and reproduce our results:
   
```
CUDA_VISIBLE_DEVICES="" python evaluate_score_final.py --model_name="MODEL_NAME"
```
   
   
### Models
Fine-tuned models are available at the following links:
- [Salience Predict Mistral-Instruct](https://huggingface.co/lingchensanwen/mistral-ins-generation-best-balanced)
- [Salience Predict Llama2-chat](https://huggingface.co/lingchensanwen/llama2-chat-generation-best-balanced)
- [Salience Predict Flan T5-base](https://huggingface.co/lingchensanwen/t5_model_1st)
- [Salience Predict Tinyllama-chat](https://huggingface.co/lingchensanwen/tiny-llama-generation-best-balanced-new)

## Answerability
Relevant data can be found here: [answerability](./data/answerability)

## Repository Status 🚧
This repository is under construction! We will frequently maintain and update it. Please let us know if you have any questions by emailing us (yating.wu@utexas.edu) or creating issues in the repo.
