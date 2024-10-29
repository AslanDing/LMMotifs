import nltk
from collections import Counter
import json

# Download required resources if you haven't already
nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

dataset = 'pubmed_gpt-4-1'#sys.argv[1]
# load json
path = f'../data/{dataset}.raw_data.json'
with open(path) as fp:
    data = json.load(fp)

original = data['original']
sampled = data['sampled']

# Sample texts divided into two parts
texts_part1 = original
texts_part2 = sampled
def calculate_pos_statistics(texts):
    total_pos_counter = Counter()
    for text in texts:
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        total_pos_counter.update(tag for word, tag in pos_tags)
    return total_pos_counter

# Calculate POS statistics for both parts
pos_part1 = calculate_pos_statistics(texts_part1)
pos_part2 = calculate_pos_statistics(texts_part2)

pos_difference = pos_part1 - pos_part2
print("POS Statistics for Part 1:")
for tag, count in pos_part1.items():
    print(f"{tag}: {count/150}")

print("\nPOS Statistics for Part 2:")
for tag, count in pos_part2.items():
    print(f"{tag}: {count/150}")

print("\nDifference (Part 1 - Part 2):")
for tag, count in pos_difference.items():
    print(f"{tag}: {count/150}")

print("\nDifference (Part 2 - Part 1):")
pos_difference = pos_part2 - pos_part1
for tag, count in pos_difference.items():
    print(f"{tag}: {count/150}")
print("")
