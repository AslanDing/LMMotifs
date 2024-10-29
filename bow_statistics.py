from sklearn.feature_extraction.text import CountVectorizer
import os
import stanza
import pickle
import sys
import random
import numpy as np
from tqdm import tqdm
import json

dataset = 'pubmed_gpt-4-1'#sys.argv[1]
# load json
path = f'../data/{dataset}.raw_data.json'
with open(path) as fp:
    data = json.load(fp)

original = data['original']
sampled = data['sampled']

# Sample documents
documents = []
documents.extend(original)
documents.extend(sampled)

# Create the Bag of Words model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

# Convert to array and get feature names
bow_array = X.toarray()
feature_names = vectorizer.get_feature_names_out()

# Display the results
print("Feature Names:", feature_names)
print("Bag of Words Array:\n", bow_array)

part1 = bow_array[:150,:].sum(axis=0)/150
part2 = bow_array[150:,:].sum(axis=0)/150
# diff = np.abs(part1-part2)
diff = part1-part2
print(part1)
print(part2)
print(diff)
print("")

