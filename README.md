# QSalience

## :star2: Introduction
This work introduces QSALIENCE, the first application-agnostic salience predictor of inquisitive questions, instruction fine-tuned over our newly annotated dataset of 1766 (context, question) pairs for salience scores, and investigates the usefulness of question salience in a downstream application of TL;DR expansion


### Salience

#### Data
Data can be found under [salience](./data/salience)
Upsampled train, default val, default test can be found in [train_val_test](./data/train_val_test)

#### Code
(Under construction)
- evaluation code coming soon!

#### Models
The fine tuned models can be found in:
- [Salience Predict Mistral-Instruct](https://huggingface.co/lingchensanwen/mistral-ins-generation-best-balanced)
- [Salience Predict Llama2-chat](https://huggingface.co/lingchensanwen/llama2-chat-generation-best-balanced)
- [Salience Predict Tinyllama-chat](https://huggingface.co/lingchensanwen/tiny-llama-generation-best-balanced)
- [Salience Predict Flan T5-base](https://huggingface.co/lingchensanwen/t5_model_1st)


### Answerability
Data can be found under [answerability](./data/answerability)



The repository is under construction! We will soon upload the scripts required to implement the paper
