import nltk
from nltk import word_tokenize, pos_tag, ne_chunk, sent_tokenize

input = "Barack Obama went as a prime minister of USA in the year of 2015 . PM MODI is the prime minister of INDIA."
ner = ne_chunk(pos_tag(word_tokenize(input)))
# print(ner)

from nltk.tree import Tree
named_entity = []
for subtree in ner:
    if isinstance(subtree, Tree):
        entity = "".join([token for token, pos in subtree.leaves()])
        named_entity.append(entity)
print(named_entity)