#!/usr/bin/python
# vim: set file-encoding=utf-8:

import sys
import math
import itertools
import operator
import cPickle
import string

import maxent

from collections import defaultdict
try:
    from collections import Counter
except:
    from counterfile import Counter
from maxent import MaxentModel
from optparse import OptionParser

unigrams = dict()


# |iterable| should yield lines.
def read_sentences(iterable):
    sentence = []
    for line in iterable:
        columns = line.rstrip().split()
        if len(columns) == 0 and len(sentence) > 0:
            yield sentence
            sentence = []
        if len(columns) > 0:
            sentence.append(columns)
    if len(sentence) > 0:
        yield sentence

# Computes (local) features for word at position |i| given that label for word
# at position |i - 1| is |previous_label|. You can pass any additional data
# via |data| argument.

# very important parameters! it's worth to play with them!
MIN_WORD_FREQUENCY = 5
MIN_LABEL_FREQUENCY = 5

def compute_features(data, words, poses, i, previous_label):
    # Condition on previous label.
    # if previous_label != "O":
        # yield "label-previous={0}".format(previous_label) 

    if data["word_frequencies"].get(string.lower(words[i]), 0) >= MIN_WORD_FREQUENCY:
        yield "word-current={0}".format(string.lower(words[i]))

    labels = data["labelled_words"].get(string.lower(words[i]), dict())
    labels = filter(lambda item: item[1] > MIN_LABEL_FREQUENCY, labels.items())
    for label in labels:
        yield "was-labelled-as={0}".format(label) 
    
    if not (words[i][0].isupper()):
        yield "small_letter"
    
    if (previous_label != 'O') and (previous_label != '^') and (words[i][0].isupper()):
        yield "After.{0}".format(previous_label)
    
    # if (previous_label == 'O') and (words[i][0].isupper()):
    #     yield "PrevPose.{0}".format(poses[i - 1])
    #     yield "CurPose.{0}".format(poses[i])
    #     yield "PrevPose-CurPose.{0}-{1}".format(poses[i-1], poses[i])

    if (previous_label != "^") and (string.lower(words[i - 1]) in data["unigrams"]["B-ORG"]) and (words[i][0].isupper()):
        # yield "UNI-ORG"
        yield "UNI-ORG={0}".format(string.lower(words[i - 1]))
    
    if (previous_label != "^") and (string.lower(words[i - 1]) in data["unigrams"]["B-LOC"]) and (words[i][0].isupper()):
        # yield "UNI-LOC"
        yield "UNI-LOC={0}".format(string.lower(words[i - 1]))
        
    if (previous_label != "^") and (string.lower(words[i - 1]) in data["unigrams"]["B-PER"]) and (words[i][0].isupper()):
        # yield "UNI-PER"
        yield "UNI-PER={0}".format(string.lower(words[i - 1]))
    
    if (previous_label != "^") and (string.lower(words[i - 1]) in data["unigrams"]["B-MISC"]) and (words[i][0].isupper()):
        # yield "UNI-MISC"
        yield "UNI-MISC={0}".format(string.lower(words[i - 1]))

        
        
    # if (i + 1 < len(words)) and (string.lower(words[i + 1]) in data["post_unigrams"]["ORG"]) and (words[i][0].isupper()):
        # # yield "UNI-ORG"
        # yield "POST-UNI-ORG={0}".format(string.lower(words[i + 1]))
    
    # if (i + 1 < len(words)) and (string.lower(words[i + 1]) in data["post_unigrams"]["LOC"]) and (words[i][0].isupper()):
        # # yield "UNI-LOC"
        # yield "POST-UNI-LOC={0}".format(string.lower(words[i + 1]))
        
    # if (i + 1 < len(words)) and (string.lower(words[i + 1]) in data["post_unigrams"]["PER"]) and (words[i][0].isupper()):
        # # yield "UNI-PER"
        # yield "POST-UNI-PER={0}".format(string.lower(words[i + 1]))
    
    # if (i + 1 < len(words)) and (string.lower(words[i + 1]) in data["post_unigrams"]["MISC"]) and (words[i][0].isupper()):
        # # yield "UNI-MISC"
        # yield "POST-UNI-MISC={0}".format(string.lower(words[i + 1]))
            
# |iterable| should yield sentences.
# |iterable| should support multiple passes.
def train_model(options, iterable):
    model = MaxentModel()
    data = {}
    dumb_counter = 6

    data["feature_set"] = set()
    data["word_frequencies"] = defaultdict(long)
    # XXX(sandello): defaultdict(lambda: defaultdict(long)) would be
    # a better choice here (for |labelled_words|) but it could not be pickled.
    # C'est la vie.
    data["labelled_words"] = dict()
    data["unigrams"] = dict()
    data["post_unigrams"] = dict()
    
    print >>sys.stderr, "*** Training options are:"
    print >>sys.stderr, "   ", options

    print >>sys.stderr, "*** First pass: Computing statistics..."
    
    unigrams = dict()
    unigrams["B-ORG"] = defaultdict(long)
    unigrams["B-MISC"] = defaultdict(long)
    unigrams["B-LOC"] = defaultdict(long)
    unigrams["B-PER"] = defaultdict(long)
    
    post_unigrams = dict()
    post_unigrams["ORG"] = defaultdict(long)
    post_unigrams["MISC"] = defaultdict(long)
    post_unigrams["LOC"] = defaultdict(long)
    post_unigrams["PER"] = defaultdict(long)
    
    for n, sentence in enumerate(iterable):
        if (n % 1000) == 0:
            print >>sys.stderr, "   {0:6d} sentences...".format(n)
        previous_word = "^"
        previous_label = "^"
        for word, pos, label in sentence:
            data["word_frequencies"][string.lower(word)] += 1
            if label.startswith("B-") or label.startswith("I-"):
                if word in data["labelled_words"]:
                    data["labelled_words"][string.lower(word)][label] += 1
                else:
                    data["labelled_words"][string.lower(word)] = defaultdict(long)
                    data["labelled_words"][string.lower(word)][label] = 1
            if label.startswith("B-") and (previous_word != "^"):
                unigrams[label][string.lower(previous_word)] += 1
                
            if (previous_label != 'O') and (previous_label != '^'):
                post_unigrams[previous_label[2:]][string.lower(word)] += 1
            previous_label = label
            previous_word = word
    

    unigram_counters = [Counter(unigrams[key]) for key in unigrams]
    total_count = Counter()
    for counter in unigram_counters:
         total_count += counter

    total_count = dict(total_count)
    inv_total_freq  = dict([[key, (math.log(sum(total_count.values()) /  total_count[key]) ** 3)] for key in total_count])
    
    for label in unigrams:
        all_sum = sum([unigrams[label][word] for word in unigrams[label]])
        uni = sorted([[(1.0 * unigrams[label][word] * inv_total_freq[word] / all_sum ), word] for word in unigrams[label]])
        uni = [word[1] for word in uni]
        data["unigrams"][label] = uni[-25:]
        print >>sys.stderr, "*** Collected {0} unigrams for {1}".format(len(data["unigrams"][label]), label)
    
    

    print data["unigrams"]
    
    # for label in post_unigrams:
        # all_sum = sum(post_unigrams[label][word] for word in post_unigrams[label])
        # data["post_unigrams"][label] = [word for word in post_unigrams[label] if 1.0 * post_unigrams[label][word] / all_sum > 0.005]
        # print >>sys.stderr, "*** Collected {0} post_unigrams for {1}".format(len(data["post_unigrams"][label]), label)
    
    print >>sys.stderr, "*** Second pass: Collecting features..."
    model.begin_add_event()
    for n, sentence in enumerate(iterable):
        if (n % 1000) == 0:
            print >>sys.stderr, "   {0:6d} sentences...".format(n)
        words, poses, labels = map(list, zip(*sentence))
        for i in xrange(len(labels)):
            features = compute_features(data, words, poses, i, labels[i - 1] if i >= 1 else "^")
            features = list(features)
            model.add_event(features, labels[i])
            for feature in features:
                data["feature_set"].add(feature)
    model.end_add_event(options.cutoff)
    print >>sys.stderr, "*** Collected {0} features.".format(len(data["feature_set"]))

    print >>sys.stderr, "*** Training..."
    maxent.set_verbose(1)
    model.train(options.iterations, options.technique, options.gaussian)
    maxent.set_verbose(0)

    print >>sys.stderr, "*** Saving..."
    model.save(options.model + ".maxent")
    with open(options.model + ".data", "w") as handle:
        cPickle.dump(data, handle)

# |iterable| should yield sentences.
def eval_model(options, iterable):
    model = MaxentModel()
    data = {}

    print >>sys.stderr, "*** Loading..."
    model.load(options.model + ".maxent")
    with open(options.model + ".data", "r") as handle:
        data = cPickle.load(handle)

    print >>sys.stderr, "*** Evaluating..."
    for n, sentence in enumerate(iterable):
        if (n % 100) == 0:
            print >>sys.stderr, "   {0:6d} sentences...".format(n)
        words, poses = map(list, zip(*sentence))
        labels = eval_model_sentence(options, data, model, words, poses)

        ## some post-proccessing for remove sequences: O I-ORG O
        previous_label = '^'
        for word, pos, label in zip(words, poses, labels):
            if (label.startswith('I-')) and ((previous_label == 'O') or (previous_label == '^')):
                label = 'B' + label[1:]
            print label
            previous_label = label
        print

# This is a helper method for |eval_model_sentence| and, actually,
# an implementation of Viterbi algorithm.
def eval_model_sentence(options, data, model, words, poses):
    viterbi_layers = [ None for i in xrange(len(words)) ]
    viterbi_backpointers = [ None for i in xrange(len(words) + 1) ]

    # Compute first layer directly.
    viterbi_layers[0] = model.eval_all(list(compute_features(data, words, poses, 0, "^")))
    viterbi_layers[0] = dict( (k, math.log(v)) for k, v in viterbi_layers[0] )
    viterbi_backpointers[0] = dict( (k, None) for k, v in viterbi_layers[0].iteritems() )

    # Compute intermediate layers.
    for i in xrange(1, len(words)):
        viterbi_layers[i] = defaultdict(lambda: float("-inf"))
        viterbi_backpointers[i] = defaultdict(lambda: None)
        for prev_label, prev_logprob in viterbi_layers[i - 1].iteritems():
            features = compute_features(data, words, poses, i, prev_label)
            features = list(features)
            for label, prob in model.eval_all(features):
                logprob = math.log(prob)
                if prev_logprob + logprob > viterbi_layers[i][label]:
                    viterbi_layers[i][label] = prev_logprob + logprob
                    viterbi_backpointers[i][label] = prev_label

    # Most probable endpoint.
    max_logprob = float("-inf")
    max_label = None
    for label, logprob in viterbi_layers[len(words) - 1].iteritems():
        if logprob > max_logprob:
            max_logprob = logprob
            max_label = label

    # Most probable sequence.
    path = []
    label = max_label
    for i in reversed(xrange(len(words))):
        path.insert(0, label)
        label = viterbi_backpointers[i][label]

    return path

################################################################################

def main():
    parser = OptionParser("A sample MEMM model for NER")
    parser.add_option("-T", "--train", action="store_true", dest="train",
        help="Do the training, if specified; do the evaluation otherwise")
    parser.add_option("-f", "--file", type="string", dest="filename",
        metavar="FILE", help="File with the training data")
    parser.add_option("-m", "--model", type="string", dest="model",
        metavar="FILE", help="File with the model")
    parser.add_option("-c", "--cutoff", type="int", default=5, dest="cutoff",
        metavar="C", help="Event frequency cutoff during training")
    parser.add_option("-i", "--iterations", type="int", default=100, dest="iterations",
        metavar="N", help="Number of training iterations")
    parser.add_option("-g", "--gaussian", type="float", default=0.0, dest="gaussian",
        metavar="G", help="Gaussian smoothing penalty (sigma)")
    parser.add_option("-t", "--technique", type="string", default="gis", dest="technique",
        metavar="T", help="Training algorithm (either 'gis' or 'lbfgs')")
    (options, args) = parser.parse_args()

    if not options.filename:
        parser.print_help()
        sys.exit(1)

    with open(options.filename, "r") as handle:
        data = list(read_sentences(handle))

    if options.train:
        print >>sys.stderr, "*** Training model..."
        train_model(options, data)
    else:
        print >>sys.stderr, "*** Evaluating model..."
        eval_model(options, data)

    print >>sys.stderr, "*** Done!"

if __name__ == "__main__":
    main()

