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
from maxent import MaxentModel
from optparse import OptionParser

unigrams = dict()
unigrams["B-ORG"] = ['sindicatos' , 'propia' , 'gubernamental' , 'nuevo' , 'extremeña' , 'sobre' , 'club' , 'peruana' , 'o' , 'británico' , 'móvil' , 'juvenil' , 'su' , 'francés' , 'estadounidense' , 'donde' , 'terrorista' , 'equipo' , 'una' , 'filial' , 'revista' , 'e' , 'un' , 'para' , 'sindicato' , 'acceso' , 'británica' , 'española' , 'compañía' , 'alemán' , 'como' , 'empresa' , 'ante' , 'grupo' , 'diario' , 'contra' , 'los' , 'entre' , 'en' , 'por' , 'las' , 'que' , 'con', 'y' , 'a' , 'al' , 'de' , 'el' , 'del' , 'la']
unigrams["B-LOC"] = ['junta' , 'ni' , 'central' , 'actual' , 'guerra' , 'riesgo' , 'las' , 'calles' , 'madrileña' , 'hotel' , 'vecina' , 'visitó' , 'pabellón' , 'grass' , 'cantabrico' , 'controlaba' , 'gaber' , 'masegosa' , 'privatizacion' , 'este' , 'estadio' , 'efe' , 'vecino' , 'antiguo' , 'o' , 'contra' , 'sobre' , 'hasta' , 'los' , 'lluvioso' , 'toda' , 'como' , 'que' , 'para' , 'por' , 'con' , 'al' , 'hacia' , 'entre' , 'cantabria' , 'desde' , 'despejado' , 'nuboso', 'la' , 'del' , 'el' , 'y' , 'a' , 'de' , 'en']
unigrams["B-PER"] = ['dominicano' , 'uruguayo' , 'argentino' , 'ruso' , 'destacó' , 'manifestó' , 'profesor' , 'rusa' , 'centrocampista' , 'propio' , 'colombiano' , 'opositor' , 'socialista' , 'comandante' , 'efe' , 'añadió' , 'alemán' , 'holandés' , 'la' , 'periodista' , 'e' , 'británico' , 'brasileño' , 'indicó' , 'afirmó' , 'explicó' , 'o' , 'cardenal' , 'pp' , 'declaró' , 'estadounidense' , 'español' , 'señaló' , 'francés' , 'general' , 'mexicano' , 'el' , 'presidente' , 'italiano' , 'como' , 'los' , 'para' , 'con' , 'dijo' , 'que' , 'por' , 'según' , 'a' , 'de' , 'y']
unigrams["B-MISC"] = ['actual' , 'conferencia' , 'segundo' , 'río' , 'origen' , 'olímpica' , 'promedio' , 'disco' , 'tienda' , 'crudo' , 'tanques' , 'transbordador' , 'esta' , 'libro' , 'o' , 'por' , 'tema' , 'grupo' , 'marca' , 'sección' , 'autopista' , 'lo' , 'página' , 'índice' , 'lema' , 'cultura' , 'revolucionó' , 'denominado' , 'sistema' , 'titulado' , 'exposición' , 'sobre' , 'proyecto' , 'a' , 'e' , 'este' , 'carretera' , 'las' , 'programa' , 'una' , 'como', 'al' , 'un' , 'en' , 'y' , 'los' , 'el' , 'del' , 'la' , 'de']
# unigrams["B-ORG"] = ['distributeur' , 'lambiekbrouwer' , 'mediagroep' , 'nord' , 'persagentschap' , 'slorpt' , 'tourjournaal' , 'uitgeverij' , 'die' , 'via' , 'brusselse' , 'maar' , 'om' , 'krant' , 'holding' , 'club' , 'magazine' , 'rond' , 'franse' , 'tijdschrift' , 'tuincentrum' , 'amerikaanse' , 'zaterdag' , 'of' , 'over' , 'tussen' , 'heeft' , 'aan' , 'bedrijf' , 'live' , 'zakenbank' , 'volgens' , 'aandeel' , 'ook' , 'foto' , 'naar' , 'als' , 'zoals' , 'britse' , 'door' , 'op' , 'in' , 'dat' , 'met' , 'voor' , 'en' , 'bij' , 'het' , 'van' , 'de']
# unigrams["B-LOC"] = ['dorp' , 'die' , 'restaurant' , 'provincie' , 'braziliaanse' , 'dorpje' , 'méér' , 'wijk' , 'heel' , 'galerie' , 'staat' , 'morning' , 'rond' , 'tot' , 'maar' , 'wil' , 'door' , 'over' , 'hoofdstad' , 'franse' , 'district' , 'rumoerige' , 'boven' , 'buiten' , 'zoals' , 'antwerpse' , 'stadje' , 'brusselse' , 'aan' , 'nabij' , 'binnen' , 'richting' , 'als' , 'vanuit' , 'bij' , 'of' , 'tussen' , 'stad' , 'met' , 'dat' , 'voor' , 'op' , 'tegen' , 'uit' , 'en' , 'het' , 'naar' , 'de' , 'van' , 'in']
# unigrams["B-PER"] = ['schreef' , 'naar' , 'meneer' , 'waar' , 'liet' , 'gaf' , 'meent' , 'of' , 'vond' , 'had' , 'koning' , 'om' , 'wat' , 'architect' , 'regisseur' , 'prins' , 'op' , 'tussen' , 'zaken' , 'bij' , 'ook' , 'over' , 'is' , 'zoals' , 'bondscoach' , 'was' , 'tegen' , 'maar' , 'premier' , 'familie' , 'aan' , 'aldus' , 'burgemeester' , 'heeft' , 'minister' , 'toen' , 'zei' , 'die' , 'als' , 'volgens' , 'president' , 'professor' , 'pater' , 'voor' , 'door' , 'dat' , 'met' , 'zegt' , 'en' , 'van']
# unigrams["B-MISC"] = ['heeft' , 'hele' , 'legendarische' , 'toenmalige' , 'gemiddelde' , 'om' , 'toenmalig' , 'naam' , 'mijn' , 'geen' , 'ook' , 'elke' , 'nieuwe' , 'finale' , 'zoals' , 'tweede' , 'is' , 'door' , 'uit' , 'oude' , 'gewezen' , 'aan' , 'grote' , 'zogeheten' , 'drie' , 'of' , 'vele' , 'eerste' , 'tijdens' , 'andere' , 'naar' , 'over' , 'miljoen' , 'alle' , 'dat' , 'jonge' , 'twee' , 'deze' , 'die' , 'met' , 'als' , 'voor' , 'op' , 'zijn' , 'en' , 'in' , 'van' , 'een' , 'het' , 'de']

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
<<<<<<< HEAD
MIN_WORD_FREQUENCY = 5
MIN_LABEL_FREQUENCY = 5
=======
MIN_WORD_FREQUENCY = 2
MIN_LABEL_FREQUENCY = 2
>>>>>>> Unigrams added. Stat.py rewrited

def compute_features(data, words, poses, i, previous_label):
    # Condition on previous label.
    if previous_label != "O":
        yield "label-previous={0}".format(previous_label) 

    if data["word_frequencies"].get(words[i], 0) >= MIN_WORD_FREQUENCY:
        yield "word-current={0}".format(words[i])

    labels = data["labelled_words"].get(words[i], dict())
    labels = filter(lambda item: item[1] > MIN_LABEL_FREQUENCY, labels.items())
 
    if ((i == 0) or (words[i - 1] == '.')) and (words[i][0].isupper()):
        yield "firstword-InitCaps"
    if (i > 0) and (words[i - 1] != '.') and (words[i][0].isupper()):
        yield "initCaps"
    if ((i == 0) or (words[i - 1] == '.')) and (not words[i][0].isupper()):
        yield "firstword-notInitCaps="
    
    # mixedCaps !!!
    if (words[i].isupper()):
        yield "allCaps"

    
    if (i > 0) and (words[i - 1][0].isupper()):
        yield "initCaps_prev"
    if (i + 1 < len(words)) and (words[i + 1][0].isupper()):
        yield "initCaps_next"
    if (i > 0) and (words[i - 1][0].isupper()) and (i + 1 < len(words)) and (words[i + 1][0].isupper()):
        if words[i][0].isupper(): 
            yield "initCapsSequence"
        else:
            yield "notInitCapsSequence"
    
    yield "{0}".format(words[i])  
    
    if (i > 0):
        if (words[i - 1][0].isupper()):
		    yield "Prev,initCaps,{0}".format(words[i-1])
        else:
		    yield "Prev,notInitCaps,{0}".format(words[i-1])
    
    if (i + 1 < len(words)):
        if (words[i + 1][0].isupper()):
		    yield "Next,initCaps,{0}".format(words[i + 1])
        else:
		    yield "Next,notInitCaps,{0}".format(words[i + 1])
   
    
        
        

    
    for label in labels:
        yield "was-labelled-as={0}".format(label)
    
    if (i > 0) and (string.lower(words[i - 1]) in unigrams["B-ORG"]):
        yield "UNI-ORG"
        yield "UNI-ORG={0}".format(string.lower(words[i - 1]))
    
    if (i > 0) and (string.lower(words[i - 1]) in unigrams["B-LOC"]):
        yield "UNI-LOC"
        yield "UNI-LOC={0}".format(string.lower(words[i - 1]))
        
    if (i > 0) and (string.lower(words[i - 1]) in unigrams["B-PER"]):
        yield "UNI-PER"
        yield "UNI-PER={0}".format(string.lower(words[i - 1]))
    
    if (i > 0) and (string.lower(words[i - 1]) in unigrams["B-MISC"]):
        yield "UNI-MISC"
        yield "UNI-MISC={0}".format(string.lower(words[i - 1]))
    
    
    if (previous_label == "^") and (words[i][0].isupper()):
        yield "firstword-InitCaps"
    if (previous_label != "^") and (words[i][0].isupper()):
        yield "initCaps"
    if (previous_label == "^") and (not words[i][0].isupper()):
        yield "firstword-notInitCaps"
    
    # mixedCaps !!!
    if (words[i].isupper()):
        yield "allCaps"

    if (i > 0) and (words[i - 1][0].isupper()):
        yield "initCaps_prev"
    if (i + 1 < len(words)) and (words[i + 1][0].isupper()):
        yield "initCaps_next"
    if (i > 0) and (words[i - 1][0].isupper()) and (i + 1 < len(words)) and (words[i + 1][0].isupper()):
        if words[i][0].isupper(): 
            yield "initCapsSequence"
        else:
            yield "notInitCapsSequence"
            
    if (i > 0):
        if (words[i - 1][0].isupper()):
		    yield "Prev_initCaps_{0}".format(words[i-1])
        else:
		    yield "Prev_notInitCaps_{0}".format(words[i-1])
    
    if (i + 1 < len(words)):
        if (words[i + 1][0].isupper()):
		    yield "Next_initCaps_{0}".format(words[i + 1])
        else:
		    yield "Next_notInitCaps_{0}".format(words[i + 1])
    
    if (previous_label != '^'):
        yield "previous_poses={0}".format(poses[i - 1])
    yield "current_poses={0}".format(poses[i])
    
            
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

    print >>sys.stderr, "*** Training options are:"
    print >>sys.stderr, "   ", options

    print >>sys.stderr, "*** First pass: Computing statistics..."
    for n, sentence in enumerate(iterable):
        if (n % 1000) == 0:
            print >>sys.stderr, "   {0:6d} sentences...".format(n)
        for word, pos, label in sentence:
            data["word_frequencies"][word] += 1
            if label.startswith("B-") or label.startswith("I-"):
                if word in data["labelled_words"]:
                    data["labelled_words"][word][label] += 1
                else:
                    data["labelled_words"][word] = defaultdict(long)
                    data["labelled_words"][word][label] = 1

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

        for word, pos, label in zip(words, poses, labels):
            print label
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

