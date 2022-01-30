from ernie import SentenceClassifier, helper

# this downloads the model to the cache

file_list=['config.json', 'tf_model.h5', 'tokenizer.json', 'vocab.txt', 'special_tokens_map.json', 'tokenizer_config.json']

for f in file_list:
    helper.download_from_hub(repo_id='jeang/bert-finetuned-sentence-classification-toy', filename=f)

# this loads the fine tuned model from the cache
classifier = SentenceClassifier(model_name='jeang/bert-finetuned-sentence-classification-toy', max_length=128, labels_no=2)

sentence = "Oh, that's great!"
probability = classifier.predict_one(sentence)[1]
print(
    f"\"{sentence}\": {probability} "
    f"[{'positive' if probability >= 0.5 else 'negative'}]"
)
