#!/usr/bin/python

import itertools
import collections
import sys
import math
import string

from collections import defaultdict

target = "B-ORG"
N = 5

def ngram_idf(handle, size):
	data = defaultdict(long)
	ngram = []
	exclude = set(string.punctuation + " ")
	shift = 0

	for n, row in enumerate(handle):
		term = row.strip().split(" ")
		stripped = ''.join(ch for ch in term[0] if ch not in exclude)
		if stripped != "":
			if n >= size + shift:
				if len(term) > 1:
					ngram_str = " ".join(ngram)
					data[ngram_str] += 1
				ngram = ngram[1:]
			ngram.append(term[0])
		else:
			shift += 1

	all_sum = sum([data[key] for key in data])
	idf = dict([[key, math.log(1.0 * all_sum / data[key], 2)] for key in data])
	return idf


def rank_ngrams(handle, size, idf):
	data = defaultdict(long)
	ngram = []
	exclude = set(string.punctuation)
	shift = 0
	print "excluded chars : ", exclude
	
	for n, row in enumerate(handle):
		term = row.strip().split(" ")
		stripped = ''.join(ch for ch in term[0] if ch not in exclude)
		if stripped != "":
			if n >= size + shift:
				if len(term) > 1 and term[2].endswith(target):
					ngram_str = " ".join(ngram)
					data[ngram_str] += 1
				ngram = ngram[1:]
			ngram.append(term[0])
		else:
			shift += 1
	sorted_rank = sorted([[data[key] * idf[key], key] for key in data])
	return sorted_rank[-50:]

		
def fancy_list(list):
	for element in list:
		sys.stdout.write(str(element) + "\t")
	sys.stdout.write("\n")


def main():
	with open("spanish.train.txt") as handle:
		idf = ngram_idf(handle, N)
	with open("spanish.train.txt") as handle:
		rank = rank_ngrams(handle, N, idf)
		map(fancy_list, rank)
		

if __name__ == "__main__":
	main() 