#!/usr/bin/python

import itertools
import collections
import sys
import math
import string

from collections import defaultdict

target = "B-MISC"
N = 1

def ngram_idf(handle, size):
    data = defaultdict(long)
    ngram = []
    exclude = set(string.punctuation + " " + string.digits)
    shift = 0

    for n, row in enumerate(handle):
        term = row.strip().split(" ")
        if len(term) > 1:
            term[0] = string.lower(term[0])
            stripped = ''.join(ch for ch in term[0] if ch not in exclude)
            if stripped == term[0]:
                ngram.append(term[0])
                if len(ngram) > N:
                    ngram = ngram[1:]
                if len(ngram) == N:
                    data[" ".join(ngram)] += 1
            else:
                ngram = []
        else:
            ngram = []
    all_sum = sum([data[key] for key in data])
    idf = dict([[key, math.log(1.0 * all_sum / data[key], 2)] for key in data])
    return idf


def rank_ngrams(handle, size, idf):
    data = defaultdict(long)
    ngram = []
    exclude = set(string.punctuation + string.digits + " ")
    shift = 0
    print "excluded chars : ", exclude
    
    for n, row in enumerate(handle):
        term = row.strip().split(" ")
        if len(term) > 1:
            term[0] = string.lower(term[0])
            if (len(ngram) == N) and (term[2] == target):       
                data[" ".join(ngram)] += 1
            stripped = ''.join(ch for ch in term[0] if ch not in exclude)
            if stripped == term[0]:
                ngram.append(term[0])
                if len(ngram) > N:
                    ngram = ngram[1:]                
            else:
                ngram = []
        else:
            ngram = []
    sorted_rank = sorted([[data[key] * idf[key], key] for key in data])
    return sorted_rank[-50:]

        
def fancy_list(list):
    for element in list:
        sys.stdout.write(str(element) + "\t")
    sys.stdout.write("\n")


def main():
    with open("dutch.train.txt") as handle:
        idf = ngram_idf(handle, N)
    with open("dutch.train.txt") as handle:
        rank = rank_ngrams(handle, N, idf)
        map(fancy_list, rank)
        for el in rank:
            print "'" + el[1] + "' ,",

if __name__ == "__main__":
    main() 