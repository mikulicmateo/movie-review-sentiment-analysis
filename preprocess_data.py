from string import punctuation
import os
from collections import Counter
from nltk.corpus import stopwords


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def load_vocab(filename):
    # open the file as read only
    file = open(os.path.join("Preprocessed_Data/Updated_Dataset", filename), 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# turn a doc into clean tokens
def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
    # load doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # update counts
    vocab.update(tokens)


# load all docs in a directory
def add_docs_to_vocab(directory, vocab):
    # walk through all files in the folder
    for filename in os.listdir(directory):
        # skip files that do not have the right extension
        if not filename.endswith(".txt"):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # add doc to vocab
        add_doc_to_vocab(path, vocab)


# save list to file
def save_list(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def create_vocab_file(min_word_occurrence, vocab_filename):
    # define vocab
    vocab = Counter()
    # add all docs to vocab
    add_docs_to_vocab('Datasets/Updated_Dataset/neg', vocab)
    add_docs_to_vocab('Datasets/Updated_Dataset/pos', vocab)
    # keep tokens with > 5 occurrence
    tokens = [k for k, c in vocab.items() if c >= min_word_occurrence]
    print(f"Words in vocab: {len(tokens)}")
    # save tokens to a vocabulary file
    save_list(tokens, os.path.join("Preprocessed_Data/Updated_Dataset", vocab_filename))


# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
    # load the doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)


def get_all_docs_in_dir(directory, vocab):
    lines = list()
    # walk through all files in the folder
    for filename in os.listdir(directory):
        # skip files that do not have the right extension
        if not filename.endswith(".txt"):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load and clean the doc
        line = doc_to_line(path, vocab)
        # add to list
        lines.append(line)
    return lines


def preprocess_data(vocab_filename, min_word_occurrence):
    create_vocab_file(min_word_occurrence, vocab_filename)
    vocab = load_vocab(vocab_filename)
    vocab = vocab.split()
    vocab = set(vocab)
    # prepare negative reviews
    negative_lines = get_all_docs_in_dir('Datasets/Updated_Dataset/neg', vocab)
    save_list(negative_lines, os.path.join("Preprocessed_Data/Updated_Dataset", "negative.txt"))
    # prepare positive reviews
    positive_lines = get_all_docs_in_dir('Datasets/Updated_Dataset/pos', vocab)
    save_list(positive_lines, os.path.join("Preprocessed_Data/Updated_Dataset", "positive.txt"))


def main():
    min_word_occurrence = 5
    vocab_filename = "vocab.txt"
    preprocess_data(vocab_filename, min_word_occurrence)


if __name__ == "__main__":
    main()
