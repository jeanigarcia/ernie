from ernie import SentenceClassifier, helper
import sys

file_list=['config.json', 'tf_model.h5', 'tokenizer.json', 'vocab.txt', 'special_tokens_map.json', 'tokenizer_config.json']

# this downloads the model to a directory
for f in file_list:
    try:
        helper.download_from_hub(repo_id='jeang/bert-finetuned-sentence-classification-toy', filename=f, cache_dir='model/toy2/')
    except Exception as exp:
        print(exp)
        break

# this loads the model from a path
classifier = SentenceClassifier(model_path='model/toy2/', max_length=128, labels_no=2)


sentence = "Oh, that's great!"
probability = classifier.predict_one(sentence)[1]
print(
    f"\"{sentence}\": {probability} "
    f"[{'positive' if probability >= 0.5 else 'negative'}]"
)
