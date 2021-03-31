import numpy as np
import math
import csv
import ast

def read_data(file):
    with open(file) as f:
        lines = csv.reader(f)
        next(lines)
        label_seq = []
        sentences = []
        for line in lines:
            label = ast.literal_eval(line[2])
            words = line[0].split()
            label_seq.append(label)
            sentences.append(words)
    return sentences, label_seq

def read_test(file):
    with open(file) as f:
        lines = csv.reader(f)
        next(lines)
        sentences = []
        for line in lines:
            words = line[0].split()
            sentences.append(words)
    return sentences

def preprocess(sentences, label_seq):
    tag_unigram = {}
    for label in label_seq:
        for i in range(len(label)):
            if label[i] not in tag_unigram:
                tag_unigram[label[i]] = 1
            else:
                tag_unigram[label[i]] += 1
    
    tag_bigram = {}
    for label in label_seq:
        for i in range(len(label) - 1):
            tag1 = label[i]
            tag2 = label[i + 1]
            if (tag1, tag2) not in tag_bigram:
                tag_bigram[(tag1, tag2)] = 1
            else:
                tag_bigram[(tag1, tag2)] += 1
    
    word_dict = {}
    for i in range(len(sentences)):
        words = sentences[i]
        labels = label_seq[i]
        for j in range(len(words)):
            if (words[j], labels[j]) not in word_dict:
                word_dict[(words[j], labels[j])] = 1
            else:
                word_dict[(words[j], labels[j])] += 1
    
    return tag_unigram, tag_bigram, word_dict

def viterbi_algo(tag_unigram, tag_bigram, word_dict, lmbda, k, test):
    score = np.zeros((2, len(test)))
    bptr = np.zeros((2, len(test)))
    
    if (test[0], 0) not in word_dict:
        word_dict[(test[0], 0)] = k
    if (test[0], 1) not in word_dict:
        word_dict[(test[0], 1)] = k
        
    score[0][0] = math.exp(lmbda * math.log((tag_unigram[0] / (tag_unigram[0] + tag_unigram[1]))) + math.log((word_dict[(test[0], 0)] / tag_unigram[0])))
    score[1][0] = math.exp(lmbda * math.log((tag_unigram[1] / (tag_unigram[0] + tag_unigram[1]))) + math.log((word_dict[(test[0], 1)]/ tag_unigram[1])))
    bptr[0][0] = 0
    bptr[1][0] = 0
    
    for t in range(1, len(test)):
        if (test[t], 0) not in word_dict:
            word_dict[(test[t], 0)] = k
        if (test[t], 1) not in word_dict:
            word_dict[(test[t], 1)] = k
        
        score[0][t] = max(score[0][t - 1] * math.exp(lmbda * math.log((tag_bigram[(0, 0)] / tag_unigram[0])) + math.log((word_dict[(test[t], 0)] / tag_unigram[0]))),
                          score[1][t - 1] * math.exp(lmbda * math.log((tag_bigram[(1, 0)] / tag_unigram[1])) + math.log((word_dict[(test[t], 0)] / tag_unigram[0]))))
        bptr[0][t] = np.argmax((score[0][t - 1] * math.exp(lmbda * math.log((tag_bigram[(0, 0)] / tag_unigram[0])) + math.log((word_dict[(test[t], 0)] / tag_unigram[0]))),
                          score[1][t - 1] * math.exp(lmbda * math.log((tag_bigram[(1, 0)] / tag_unigram[1])) + math.log((word_dict[(test[t], 0)] / tag_unigram[0])))))
        
        score[1][t] = max(score[0][t - 1] * math.exp(lmbda * math.log((tag_bigram[(0, 1)] / tag_unigram[0])) + math.log((word_dict[(test[t], 1)] / tag_unigram[1]))),
                          score[1][t - 1] * math.exp(lmbda * math.log((tag_bigram[(1, 1)] / tag_unigram[1])) + math.log((word_dict[(test[t], 1)] / tag_unigram[1]))))
        bptr[1][t] = np.argmax((score[0][t - 1] * math.exp(lmbda * math.log((tag_bigram[(0, 1)] / tag_unigram[0])) + math.log((word_dict[(test[t], 1)] / tag_unigram[1]))),
                          score[1][t - 1] * math.exp(lmbda * math.log((tag_bigram[(1, 1)] / tag_unigram[1])) + math.log((word_dict[(test[t], 1)] / tag_unigram[1])))))
    
    T = np.zeros(len(test))
    T[len(test) - 1] = np.argmax((score[0][len(test) - 1], score[1][len(test) - 1]))
    for i in range(len(test) - 2, -1, -1):
        T[i] = bptr[int(T[i + 1])][i + 1]
    
    return T

def predict(tag_unigram, tag_bigram, word_dict, lmbda, k, test):
    with open("predc.csv", "w", newline='') as f:
        f_writer = csv.writer(f, delimiter=",")
        f_writer.writerow(["idx", "label"])
        
        count = 1
        for sentence in test:
            labels = viterbi_algo(tag_unigram, tag_bigram, word_dict, lmbda, k, sentence)
            for i in range(len(labels)):
                f_writer.writerow([count, int(labels[i])])
                count += 1

train_sen, train_label = read_data("train.csv")
tag_unigram, tag_bigram, word_dict = preprocess(train_sen, train_label)
test = read_test("test_no_label.csv")
predict(tag_unigram, tag_bigram, word_dict, 0.5, 0.1, test)