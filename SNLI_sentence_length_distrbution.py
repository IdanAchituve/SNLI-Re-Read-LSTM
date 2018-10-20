import json
import re
import numpy as np
import matplotlib.pyplot as plt
import collections

# Read datasets
def read_data(fname):
    data = []  # list of lists. each list is [[sentence1],[sentence2],label]
    max_len = 0
    unique_words = set() # in order to get relevant words from the embedding file

    with open(fname, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            # get relevant data
            sentence1 = str(json_obj["sentence1"])
            sentence2 = str(json_obj["sentence2"])
            gold_label = str(json_obj["gold_label"])
            if gold_label != "-":
                # get all words/tokens in the sentences
                sentence1_list = [token.strip().lower() for token in re.split('(\W+)?', sentence1) if token.strip()]
                sentence2_list = [token.strip().lower() for token in re.split('(\W+)?', sentence2) if token.strip()]
                # create a record to add to the dataset
                record = [sentence1_list] + [sentence2_list] + [gold_label]
                data.append(record)
                # save unique words
                for word in sentence1_list:
                    unique_words.add(word)
                for word in sentence2_list:
                    unique_words.add(word)
                # save the max length sentence
                if len(sentence1_list) > max_len:
                    max_len = len(sentence1_list)

    return data, unique_words, max_len


def bar_plot(train, dev, test):

    count_premise_dict = {}
    count_hypothsis_dict = {}
    for example in train:
        premise_len = len(example[0])
        hypo_len = len(example[1])
        if count_premise_dict.has_key(premise_len):
            count_premise_dict[premise_len] += 1
        else:
            count_premise_dict[premise_len] = 1

        if count_hypothsis_dict.has_key(hypo_len):
            count_hypothsis_dict[hypo_len] += 1
        else:
            count_hypothsis_dict[hypo_len] = 1

    count_premise_dict = collections.OrderedDict(sorted(count_premise_dict.items()))
    count_hypothsis_dict = collections.OrderedDict(sorted(count_hypothsis_dict.items()))
    train_prem_x = np.fromiter(iter(count_premise_dict.keys()), dtype=int)
    train_hypo_x = np.fromiter(iter(count_hypothsis_dict.keys()), dtype=int)
    train_prem_y = np.fromiter(iter(count_premise_dict.values()), dtype=int)
    train_hypo_y = np.fromiter(iter(count_hypothsis_dict.values()), dtype=int)
    print("set train")


    count_premise_dict = {}
    count_hypothsis_dict = {}
    for example in dev:
        premise_len = len(example[0])
        hypo_len = len(example[1])
        if count_premise_dict.has_key(premise_len):
            count_premise_dict[premise_len] += 1
        else:
            count_premise_dict[premise_len] = 1

        if count_hypothsis_dict.has_key(hypo_len):
            count_hypothsis_dict[hypo_len] += 1
        else:
            count_hypothsis_dict[hypo_len] = 1

    count_premise_dict = collections.OrderedDict(sorted(count_premise_dict.items()))
    count_hypothsis_dict = collections.OrderedDict(sorted(count_hypothsis_dict.items()))
    dev_prem_x = np.fromiter(iter(count_premise_dict.keys()), dtype=int)
    dev_hypo_x = np.fromiter(iter(count_hypothsis_dict.keys()), dtype=int)
    dev_prem_y = np.fromiter(iter(count_premise_dict.values()), dtype=int)
    dev_hypo_y = np.fromiter(iter(count_hypothsis_dict.values()), dtype=int)
    print("set dev")

    count_premise_dict = {}
    count_hypothsis_dict = {}
    for example in test:
        premise_len = len(example[0])
        hypo_len = len(example[1])
        if count_premise_dict.has_key(premise_len):
            count_premise_dict[premise_len] += 1
        else:
            count_premise_dict[premise_len] = 1

        if count_hypothsis_dict.has_key(hypo_len):
            count_hypothsis_dict[hypo_len] += 1
        else:
            count_hypothsis_dict[hypo_len] = 1

    count_premise_dict = collections.OrderedDict(sorted(count_premise_dict.items()))
    count_hypothsis_dict = collections.OrderedDict(sorted(count_hypothsis_dict.items()))
    test_prem_x = np.fromiter(iter(count_premise_dict.keys()), dtype=int)
    test_hypo_x = np.fromiter(iter(count_hypothsis_dict.keys()), dtype=int)
    test_prem_y = np.fromiter(iter(count_premise_dict.values()), dtype=int)
    test_hypo_y = np.fromiter(iter(count_hypothsis_dict.values()), dtype=int)
    print("set test")

    plt.subplot(3, 2, 1)
    plt.plot(train_prem_x, train_prem_y, '.-')
    plt.title('Train Premise')
    plt.xlabel('Sentence Length')
    plt.ylabel('#Sentences')
    plt.yticks(np.arange(0, 50000, 10000))

    plt.subplot(3, 2, 2)
    plt.plot(train_hypo_x, train_hypo_y, '.-')
    plt.title('Train Hypothesis')
    plt.xlabel('Sentence Length')
    plt.ylabel('#Sentences')
    plt.yticks(np.arange(0, 100000, 20000))

    plt.subplot(3, 2, 3)
    plt.plot(dev_prem_x, dev_prem_y, '.-')
    plt.title('Dev Premise')
    plt.xlabel('Sentence Length')
    plt.ylabel('#Sentences')
    plt.yticks(np.arange(0, 1000, 200))

    plt.subplot(3, 2, 4)
    plt.plot(dev_hypo_x, dev_hypo_y, '.-')
    plt.title('Dev Hypothesis')
    plt.xlabel('Sentence Length')
    plt.ylabel('#Sentences')
    plt.yticks(np.arange(0, 1600, 400))

    plt.subplot(3, 2, 5)
    plt.plot(test_prem_x, test_prem_y, '.-')
    plt.title('Test Premise')
    plt.xlabel('Sentence Length')
    plt.ylabel('#Sentences')
    plt.yticks(np.arange(0, 1000, 200))

    plt.subplot(3, 2, 6)
    plt.plot(test_hypo_x, test_hypo_y, '.-')
    plt.title('Test Hypothesis')
    plt.xlabel('Sentence Length')
    plt.ylabel('#Sentences')
    plt.yticks(np.arange(0, 1600, 400))

    plt.tight_layout()

    plt.show()


if __name__ == '__main__':

    import sys
    train_file = sys.argv[1]
    dev_file = sys.argv[2]
    test_file = sys.argv[3]

    # read train and dev data sets
    train, train_words, max_len_train = read_data(
        train_file)  # read train data to list. each list item is a sentence. each sentence is a tuple
    dev, dev_words, max_len_dev = read_data(
        dev_file)  # read train data to list. each list item is a sentence. each sentence is a tuple
    test, test_words, max_len_test = read_data(
        test_file)  # read train data to list. each list item is a sentence. each sentence is a tuple

    print("read data")
    bar_plot(train, dev, test)