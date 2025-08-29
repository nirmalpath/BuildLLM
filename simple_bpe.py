#Notes:
#Tokenization: Breaking text into smaller units and then assigning unique integer tokens
#Word tokenization: Assigning tags to each word / massive vocabs n out of vocab problems / morphological variants
#Char tokenization: Solves out of vocab / destroys semantic concept of word
#Sub-word tokenziation: Middle ground / common words remain single while rare words are broken / 
#       choose size of token / no out of vocab / efficient
#Byte Pair Encoding: starts with smallest possible unit and merges to build vocab
#           prepare data / init vocab: every unique char / 
#           loop: freq of adjacent pairs, find the most frequent n merge, replace all occurences of the pair with new token
#Other algos: WordPiece (BERT) and SentencePiece (google)           



import re
import collections

#Get stats on adjacent pairs
def get_stats(vocab):
    """
    Returns a Counter with counts of all adjacent word pairs (bigrams) in the text.
    """
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

#Merge the most freq pair
def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))    
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

# Our toy corpus
corpus = {
    'low </w>': 5,
    'lower </w>': 2,
    'newest </w>': 6,
    'widest </w>': 3
}

#Initialize vocab with characters
vocab = {' '.join(word): freq for word, freq in corpus.items()}
print(f"Initial vocab(char based): {vocab}")

#Learn Merges
num_merges = 10
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    #Find the most frequent pair
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(f"Merge{i + 1}: Merged{best} -> {''.join(best)}")
    print(f"Current Vocab:{vocab}\n")

print("---Final Result---")
print(f"Final Merged Vocab: {list(vocab.keys())}")

# The learned vocabulary now contains meaningful subwords like 'est</w>' and 'low' 
# A new word like "lowest</w>" would be tokenized into 'low' and 'est</w>'.