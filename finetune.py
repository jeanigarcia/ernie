from ernie import SentenceClassifier, Models, helper
import pandas as pd
from csv import reader


tuples = [
    ("This is a positive example. I'm very happy today.", 1),
    ("This is a negative sentence. Everything was wrong today at work.", 0)
]

#tuples = []
#for f in ['data/amazon_cells_labelled.txt', 'data/imdb_labelled.txt', 'data/yelp_labelled.txt']:

#    myfile = open(f, 'r')
#    myreader = reader(myfile, delimiter='\t')
#    for row in myreader:
#        tuples.append((row[0], int(row[1])))
#    myfile.close()

df = pd.DataFrame(tuples)


classifier = SentenceClassifier(
    model_name=Models.BertBaseUncased, max_length=128, labels_no=2)
classifier.load_dataset(df, validation_split=0.2)
classifier.fine_tune(epochs=4, learning_rate=2e-5,
                     training_batch_size=32, validation_batch_size=64)

classifier.dump('model/toy/')



