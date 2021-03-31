import numpy as np
import math
import csv
import ast
from sklearn.naive_bayes import MultinomialNB

def read_data(file):
    with open(file) as f:
        lines = csv.reader(f)
        next(lines)
        label_seq = []
        sentences = []
        POS_tags = []
        for line in lines:
            label = ast.literal_eval(line[2])
            words = line[0].split()
            pos = line[1].split()
            POS_tags.append(pos)
            label_seq.append(label)
            sentences.append(words)
    return sentences, label_seq, POS_tags

def read_test(file):
    with open(file) as f:
        lines = csv.reader(f)
        next(lines)
        sentences = []
        pos_tags = []
        for line in lines:
            words = line[0].split()
            pos = line[1].split()
            sentences.append(words)
            pos_tags.append(pos)
    return sentences, pos_tagsa

def preprocess(sentences, label_seq, POS_tags):    
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
    pos_dict = {}
    for i in range(len(sentences)):
        words = sentences[i]
        labels = label_seq[i]
        pos = POS_tags[i]
        for j in range(len(words)):
            if (words[j], labels[j]) not in word_dict:
                word_dict[(words[j], labels[j])] = 1
            else:
                word_dict[(words[j], labels[j])] += 1
            if (pos[j], 1) not in pos_dict:
                pos_dict[(pos[j], 1)] = 1
            else:
                pos_dict[(pos[j], 1)] += 1
                
    feature = []
    output = []
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            if (POS_tags[i][j], 1) not in pos_dict:
                pos_dict[(POS_tags[i][j], 1)] = 0
            if j != (len(sentences[i]) - 1) and (POS_tags[i][j + 1], 1) not in pos_dict:
                pos_dict[(POS_tags[i][j + 1], 1)] = 0
            vector = [j]  # index of current word
            vector.append(pos_dict[(POS_tags[i][j], 1)] / tag_unigram[1])  # emission probability for POS tags
            if j != 0 and j != (len(sentences[i]) - 1):
                vector.append(pos_dict[(POS_tags[i][j - 1], 1)] / tag_unigram[1])  # emission probability for previous word's POS tag
                vector.append(pos_dict[(POS_tags[i][j + 1], 1)] / tag_unigram[1])  # emission probability for next word's POS tag
            else:
                vector.append(0)
                vector.append(0)
            if (sentences[i][j], 1) not in word_dict:
                word_dict[(sentences[i][j], 1)] = 0
            vector.append(word_dict[(sentences[i][j], 1)] / tag_unigram[1])  # emission probability for current word
            feature.append(vector)
            output.append(label_seq[i][j])
    
    return tag_unigram, tag_bigram, word_dict, pos_dict, feature, output

def viterbi_algo(tag_unigram, tag_bigram, word_dict, pos_dict, feature, output, lmbda, k, test_words, test_pos):
    score = np.zeros((2, len(test_words)))
    bptr = np.zeros((2, len(test_words)))
    
    clf = MultinomialNB()
    clf.fit(feature, output)
   
    X = []
    for i in range(len(test_words)):
        if (test_pos[i], 1) not in pos_dict:
            pos_dict[(test_pos[i], 1)] = k
        if i != (len(test_words) - 1) and (test_pos[i + 1], 1) not in pos_dict:
            pos_dict[(test_pos[i + 1], 1)] = k
        vector = [i]
        vector.append(pos_dict[(test_pos[i], 1)] / tag_unigram[1])
        if i != 0 and i != (len(test_words) - 1):
            vector.append(pos_dict[(test_pos[i - 1], 1)] / tag_unigram[1])
            vector.append(pos_dict[(test_pos[i + 1], 1)] / tag_unigram[1])
        else:
            vector.append(0)
            vector.append(0)
        if (test_words[i], 1) not in word_dict:
            word_dict[(test_words[i], 1)] = k
        vector.append(word_dict[(test_words[i], 1)] / tag_unigram[1])
        X.append(vector)
    
    probs = clf.predict_proba(X)
    
    if (test_words[0], 0) not in word_dict:
        word_dict[(test_words[0], 0)] = k
    if (test_words[0], 1) not in word_dict:
        word_dict[(test_words[0], 1)] = k
        
    score[0][0] = math.exp(lmbda * math.log((tag_unigram[0] / (tag_unigram[0] + tag_unigram[1]))) + math.log(probs[0][0]))
    score[1][0] = math.exp(lmbda * math.log((tag_unigram[1] / (tag_unigram[0] + tag_unigram[1]))) + math.log(probs[0][1]))
    bptr[0][0] = 0
    bptr[1][0] = 0
    
    for t in range(1, len(test_words)):
        if (test_words[t], 0) not in word_dict:
            word_dict[(test_words[t], 0)] = k
        if (test_words[t], 1) not in word_dict:
            word_dict[(test_words[t], 1)] = k
        
        score[0][t] = max(score[0][t - 1] * math.exp(lmbda * math.log((tag_bigram[(0, 0)] / tag_unigram[0])) + math.log(probs[t][0])),
                          score[1][t - 1] * math.exp(lmbda * math.log((tag_bigram[(1, 0)] / tag_unigram[1])) + math.log(probs[t][0])))
        bptr[0][t] = np.argmax((score[0][t - 1] * math.exp(lmbda * math.log((tag_bigram[(0, 0)] / tag_unigram[0])) + math.log(probs[t][0])),
                          score[1][t - 1] * math.exp(lmbda * math.log((tag_bigram[(1, 0)] / tag_unigram[1])) + math.log(probs[t][0]))))
        
        score[1][t] = max(score[0][t - 1] * math.exp(lmbda * math.log((tag_bigram[(0, 1)] / tag_unigram[0])) + math.log(probs[t][1])),
                          score[1][t - 1] * math.exp(lmbda * math.log((tag_bigram[(1, 1)] / tag_unigram[1])) + math.log(probs[t][1])))
        bptr[1][t] = np.argmax((score[0][t - 1] * math.exp(lmbda * math.log((tag_bigram[(0, 1)] / tag_unigram[0])) + math.log(probs[t][1])),
                          score[1][t - 1] * math.exp(lmbda * math.log((tag_bigram[(1, 1)] / tag_unigram[1])) + math.log(probs[t][1]))))
    
    T = np.zeros(len(test_words))
    T[len(test_words) - 1] = np.argmax((score[0][len(test_words) - 1], score[1][len(test_words) - 1]))
    for i in range(len(test_words) - 2, -1, -1):
        T[i] = bptr[int(T[i + 1])][i + 1]
    
    return T

def predict(tag_unigram, tag_bigram, word_dict, pos_dict, feature, output, lmbda, k, test_words, test_pos):
    with open("pred.csv", "w", newline='') as f:
        f_writer = csv.writer(f, delimiter=",")
        f_writer.writerow(["idx", "label"])
        
        count = 1
        for i in range(len(test_words)):
            labels = viterbi_algo(tag_unigram, tag_bigram, word_dict, pos_dict, feature, output, lmbda, k, test_words[i], test_pos[i])
            for i in range(len(labels)):
                f_writer.writerow([count, int(labels[i])])
                count += 1
                
sentences, label_seq, POS_tags = read_data("train.csv")
tag_unigram, tag_bigram, word_dict, pos_dict, feature, output = preprocess(sentences, label_seq, POS_tags)
test_words, test_pos = read_test("test_no_label.csv")
predict(tag_unigram, tag_bigram, word_dict, pos_dict, feature, output, 0.5, 0.1, test_words, test_pos)