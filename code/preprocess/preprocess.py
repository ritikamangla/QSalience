import pandas as pd
import numpy as np
import re
import json
import argparse


def format_data(question_df, articles_path):

    data = []
    def get_context(article_id):
        context = []
        path = articles_path
        with open(path, "r", encoding='utf-8') as r:
            for line in r:
                if line and line[0].isdigit():
                    line = re.sub(r"^\d+\s*", "", line).strip()
                    context.append(line)
        return context

    for _, row in question_df.iterrows():
        context = get_context(row['article_id'])
        current_sentence_id = row['sentence_id']
        current_sentence = context[current_sentence_id-1]

        data_object = {
        "instruction": '''
    Give a score from 1 to 5 for how important it is for the question to be answered later in the article.
    Score = 1 means the question is completely unrelated to the topic of the article.
    Score = 2 means the question is related to the article but answering it is not useful in making the article feel complete.
    Score = 3 means the question is related to the article but answering it might not enhance the understanding of the article.
    Score = 4 means the question is related to the article and answering it is somewhat useful in enhancing the understanding of the article.
    Score = 5 means the question is related to the article and should definitely be answered because it might provide explanation for some new concepts.
    ''',
        "input": {
        "article": " ".join(context[:current_sentence_id]),
        "question": row['question'],
        },
        "output": "-1"
    }
        data.append(data_object)
    with open('example.json', 'w') as file:
        json.dump(data, file, indent=4)
        print("processed data dumped successfully into example.json")
    return data

def main():
	parser = argparse.ArgumentParser(description='Evaluate the model')
	parser.add_argument('--question_csv_path', type=str, help='The question file path, in csv format')
	parser.add_argument('--article_txt_path', type=str, help='The article file path, in txt format')

	args = parser.parse_args()

	question_df = pd.read_csv(args.question_csv_path)
	format_data(question_df, args.article_txt_path)


if __name__=="__main__": 
    main()


