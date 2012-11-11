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
    # I have python 2.6 in cygwin
    # counterfile is file with implementation of Counter
    from counterfile import Counter
from maxent import MaxentModel
from optparse import OptionParser

def is_word(s):
    exclude = string.punctuation + " " + string.digits
    for ch in exclude:
        if ch in s:
            return False
    return True
 
def is_mixed_word(s):
    if (len(s) > 2):
        for ch in s[1:]:
            if ch.isupper():
                return True
    return False

def is_number(s):
    try:
        x = float(s)
        return True
    except ValueError:
        return False

def is_url(s):
    dig = False
    chars = False
    punc = False
    for ch in s:
        if ch in string.punctuation:
            punc = True
        if ch in string.digits:
            dig = True
        if (ch in string.lowercase) or (ch in string.lowercase):
            chars = True
    return (dig and chars and punc)

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
MIN_LABEL_FREQUENCY = 2

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

    # here need some magic, I think.
    # There are a lot of urls in text. But what to do with them?
    # This doesn't work. Perhaps, my implemetation is just bad :-)
    # Someone, fix is_url().
    # if (is_url(words[i])):
    #     yield "MayBeItsURL?"
    if (is_number(words[i])):
        yield "is_float"
    elif (len(words[i]) > 1) and (is_number(words[i][1:])):
        yield "almost_is_float"
    elif not (is_word(words[i])):
        yield "not_word"
    elif (len(words[i]) > 1) and (is_word(words[i][:-1])):
        yield "almost_word"

    # if (is_mixed_word(words[i])) and (words[i][0].isupper()):
        # yield "initCaps_and_is_mixed_word"

    # if (is_mixed_word(words[i])) and (not words[i][0].isupper()):
    #     yield "NotinitCaps_and_is_mixed_word"

    if (not words[i][0].isupper()):
        yield "small_letter"
  
        if (previous_label != '^') and (i + 1 < len(words)) and (words[i - 1][0].isupper()) and (words[i + 1][0].isupper()):
            yield "small_letter_in_sequence.{0}.{1}.{2}".format(words[i], previous_label, poses[i - 1])
        if (previous_label != '^') and (i + 2 < len(words)) and (words[i - 1][0].isupper()) and (not words[i + 1][0].isupper()) and (words[i + 2][0].isupper()):
            yield "double_small_letter_in_sequence1.{0}.{1}.{2}.{3}".format(words[i], words[i + 1], previous_label, poses[i])
        if (i > 1) and (i + 2 < len(words)) and (words[i - 2][0].isupper()) and (not words[i - 1][0].isupper()) and (words[i + 1][0].isupper()):
            yield "double_small_letter_in_sequence2.{0}.{1}.{2}".format(words[i - 1], words[i], previous_label)
    
    if (previous_label == '^'):
        if (i + 1 < len(words)) and (words[i + 1][0].isupper()):
            yield "FirstWord_NextWordIsUpper.{0}.{1}".format(poses[i], poses[i+1])
            yield "NextBigWord".format(string.lower(words[i + 1]))
        if (i + 3 < len(words)) and (not words[i + 1][0].isupper()) and (not words[i + 2][0].isupper()) and (words[i + 3][0].isupper()):
            yield "very_long_sequence.{0}.{1}".format(words[i + 1], words[i + 2])
        elif (i + 2 < len(words)) and (not words[i + 1][0].isupper()) and (words[i + 2][0].isupper()):
            yield "long_sequence.{0}".format(words[i + 1])
        elif (i + 1 < len(words)):
            yield "PosesC.{0}".format(poses[i])
            yield "PosesN.{0}".format(poses[i + 1])
            yield "WordsN.{0}".format(words[i + 1])
        else:
            yield "PosesC.{0}".format(poses[i])
               
    flag = 0
    if (previous_label == "O") and (string.lower(words[i - 1]) in data["unigrams"]["B-ORG"]) and (words[i][0].isupper()):
        # yield "UNI-ORG"
        flag = 1
        yield "UNI-ORG={0}".format(string.lower(words[i - 1]))
    
    if (previous_label == "O") and (string.lower(words[i - 1]) in data["unigrams"]["B-LOC"]) and (words[i][0].isupper()):
        # yield "UNI-LOC"
        flag = 1
        yield "UNI-LOC={0}".format(string.lower(words[i - 1]))
        
    if (previous_label == "O") and (string.lower(words[i - 1]) in data["unigrams"]["B-PER"]) and (words[i][0].isupper()):
        # yield "UNI-PER"
        flag = 1
        yield "UNI-PER={0}".format(string.lower(words[i - 1]))
    
    if (previous_label == "O") and (string.lower(words[i - 1]) in data["unigrams"]["B-MISC"]) and (words[i][0].isupper()):
        # yield "UNI-MISC"
        flag = 1
        yield "UNI-MISC={0}".format(string.lower(words[i - 1]))
        
    if (previous_label != 'O') and (previous_label != '^') and (words[i][0].isupper()):
        yield "After.{0}".format(previous_label)
        if (i > 1) and (not words[i - 1][0].isupper()) and (words[i - 2][0].isupper()):
            yield "prev_long_sequence.{0}".format(words[i - 1])
        elif (i > 2) and (not words[i - 1][0].isupper()) and (not words[i - 2][0].isupper()) and (words[i - 3][0].isupper()):
            yield "prev_very_long_sequence.{0}.{1}".format(words[i - 2], words[i - 1])
    
    if (flag == 0) and (previous_label == 'O') and (words[i][0].isupper()):    
        if (i + 1 < len(words)) and (words[i + 1][0].isupper()):
            # if (is_mixed_word(words[i + 1])):
            #     yield "NextWordIsBigAndMixed"
            yield "NextWordInit.CapAfterO.{0}".format(poses[i - 1])
            yield "NextBigWord".format(string.lower(words[i + 1]))
        else:
            yield "AfterPosO.{0}".format(poses[i - 1]) 
            #  yield "WordsN.{0}".format(words[i + 1])  
        yield "Previous_poses.{0}".format(string.lower(poses[i - 1]))
        yield "Previous_word.{0}".format(string.lower(words[i - 1])) 

        if (i > 1):
            yield "PPrevious_word.{0}".format(string.lower(words[i - 2]))
            yield "PPrevious_poses.{0}".format(string.lower(poses[i - 2]))
        if (i > 2):
            yield "PPPrevious_word.{0}".format(string.lower(words[i - 3]))
            yield "PPPrevious_poses.{0}".format(string.lower(poses[i - 3]))
        if (i > 3):
            # it's right, don't worry! It's just magic :-)
            yield "PPPrevious_word.{0}".format(string.lower(words[i - 4]))
            yield "PPPrevious_poses.{0}".format(string.lower(poses[i - 4]))
             
        if (i > 1) and (is_number(words[i - 1])):
            yield "prev_word_is_number!"
        # elif (i > 1) and (len(words[i - 1]) > 1) and (is_number(words[i - 1][1:])):
        #     yield "prev_word_is_almost_number!"
         
            
# |iterable| should yield sentences.
# |iterable| should support multiple passes.
def train_model(options, iterable):
    model = MaxentModel()
    data = {}
    

    data["feature_set"] = set()
    data["word_frequencies"] = defaultdict(long)
    # XXX(sandello): defaultdict(lambda: defaultdict(long)) would be
    # a better choice here (for |labelled_words|) but it could not be pickled.
    # C'est la vie.
    data["labelled_words"] = dict()
    data["unigrams"] = dict()

    
    print >>sys.stderr, "*** Training options are:"
    print >>sys.stderr, "   ", options

    print >>sys.stderr, "*** First pass: Computing statistics..."
    
    unigrams = dict()
    unigrams["B-ORG"] = defaultdict(long)
    unigrams["B-MISC"] = defaultdict(long)
    unigrams["B-LOC"] = defaultdict(long)
    unigrams["B-PER"] = defaultdict(long)

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
        data["unigrams"][label] = uni[-50:]
        # print >>sys.stderr, "*** Collected {0} unigrams for {1}".format(len(data["unigrams"][label]), label)

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

        for i in xrange(0, len(words)):
            label = labels[i]            
            if (label.startswith('I-')) and ((previous_label == 'O') or (previous_label == '^')):
                label = 'B' + label[1:]
            # if (i + 1 < len(words)) and (labels[i + 1] != 'O') and (labels[i] != 'O') and (labels[i + 1][0] != 'B') and (labels[i + 1][2:] != labels[i][2:]):
                # label = labels[i][:1] + labels[i + 1][2:]
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

