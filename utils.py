# importing libraries
import re
import os
import glob
import random
from nltk.corpus import stopwords
import nltk
from collections import Counter
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import ssl
# used to get around the stopwords import error
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
#-----------------------------------------------------------------------------------------------------------------------------------------
# Initializing and retracting dataset

REPLACE_NO_SPACE = re.compile("[._;:!`¦\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
nltk.download('stopwords')  

def preprocess_text(text):
	stop_words = set(stopwords.words('english'))
	text = REPLACE_NO_SPACE.sub("", text)
	text = REPLACE_WITH_SPACE.sub(" ", text)
	text = re.sub(r'\d+', '', text)
	text = text.lower()
	words = text.split()
	return [w for w in words if w not in stop_words]

def load_training_set(percentage_positives, percentage_negatives):
	vocab = set()
	positive_instances = []
	negative_instances = []
	for filename in glob.glob('train/pos/*.txt'):
		if random.random() > percentage_positives:
			continue
		with open(os.path.join(os.getcwd(), filename), 'r') as f:
			contents = f.read()
			contents = preprocess_text(contents)
			positive_instances.append(contents)
			vocab = vocab.union(set(contents))
	for filename in glob.glob('train/neg/*.txt'):
		if random.random() > percentage_negatives:
			continue
		with open(os.path.join(os.getcwd(), filename), 'r') as f:
			contents = f.read()
			contents = preprocess_text(contents)
			negative_instances.append(contents)
			vocab = vocab.union(set(contents))	
	return positive_instances, negative_instances, vocab

def load_test_set(percentage_positives, percentage_negatives):
	positive_instances = []
	negative_instances = []
	for filename in glob.glob('test/pos/*.txt'):
		if random.random() > percentage_positives:
			continue
		with open(os.path.join(os.getcwd(), filename), 'r') as f:
			contents = f.read()
			contents = preprocess_text(contents)
			positive_instances.append(contents)
	for filename in glob.glob('test/neg/*.txt'):
		if random.random() > percentage_negatives:
			continue
		with open(os.path.join(os.getcwd(), filename), 'r') as f:
			contents = f.read()
			contents = preprocess_text(contents)
			negative_instances.append(contents)
	return positive_instances, negative_instances

def load_data(percentage_positives_train, percentage_negatives_train, percentage_positives_test, percentage_negatives_test):
    posTrain, negTrain, vocab = load_training_set(percentage_positives_train, percentage_negatives_train)
    posTest, negTest  = load_test_set(percentage_positives_test, percentage_negatives_test)
    return posTrain, negTrain, posTest, negTest, vocab

#-----------------------------------------------------------------------------------------------------------------------------------------

#  Naive Bayes Algorithm Helper

# function to the return a dict containing the word and their counts for each instance
# posData: List : positive data set; negData: List:  negative data set
def trainData(posData, negData):
    # word counts for positive data -> convert to dict
    posDict = defaultdict(int)
    for line in posData:
        for word in line:
            posDict[word] += 1
	# word counts for negative data
    negDict = defaultdict(int)
    for line in negData:
        for word in line:
            negDict[word] += 1
	# return dicts containing word counts for each classification
    return posDict, negDict

# function to calculate the posterior probability of a given instance belonging to a specified classification
# posTrain: List : positive training instances; negTrain: List : negative training instances; posDict: Dict: word counts for positive instances; negDict: Dict: word counts for negative instances; 
# vocab: List : set of words in the vocabulary; classification: Str : "Positive"/"Negative"; instance: List : current instance; logProb: bool : return log results; lapaceSmooth: bool : apply lapace smoothing; 
# smoothConst: float : alpha value to apply lapac smoothing
def probabiltyCalc(posTrain, negTrain, posDict, negDict, vocab, classification, instance, logProb=False, laplaceSmooth=False, smoothConst=1):
    # which set and dictionary to use based on classification
    trainData, word_dict = (posTrain, posDict) if classification in ("positive") else (negTrain, negDict)
    cntInstances = len(trainData)
    # prior probability of the classification
    priorProb = cntInstances / (len(posTrain) + len(negTrain))
    # list to store probability of each word in the instance
    instanceProb = []
    denominator = sum(len(i) for i in trainData)
    vocab_size = len(vocab)
    for word in instance:
        # check if Laplace smoothing is enabled
        if not laplaceSmooth:
            if word in word_dict:
                p_word = word_dict[word] / denominator
            else:
                continue
        else:
            p_word = (word_dict.get(word, 0) + smoothConst) / (denominator + smoothConst * vocab_size)
        instanceProb.append(p_word)
	# add prior probability to the beginning of the list
    instanceProb.insert(0, priorProb)
    # check if logarithmic probability is enabled
    if logProb:
        return sum(math.log10(p) for p in instanceProb)
    return math.prod(instanceProb)

# function tests the trained model on the test data
# posTrain: List : positive training instances; negTrain: List : negative training instances; posTest: List: positive testing instances; negTest: List: negative testing instances; 
# vocab: List : set of words in the vocabulary; logProb: bool : return log results; lapaceSmooth: bool : apply lapace smoothing; smoothConst: float : alpha value to apply lapac smoothing
def test_model(posTrain, negTrain, posTest, negTest, vocab, logProb=False, laplaceSmooth=False, smoothConst=1):
    
    posDict, negDict = trainData(posTrain, negTrain)
    #respective counts of each element in confusion matrix
    truePos, trueNeg, falsePos, falseNeg = 0, 0, 0, 0
    # test model on the positive instances in the test data
    for posInstance in [dict(Counter(i)) for i in posTest]:
        probPos = probabiltyCalc(posTrain, negTrain, posDict, negDict, vocab, 'positive', posInstance, logProb, laplaceSmooth, smoothConst)
        probNeg = probabiltyCalc(posTrain, negTrain, posDict, negDict, vocab, 'negative', posInstance, logProb, laplaceSmooth, smoothConst)
        # if the probability of the instance being positive > the probability of it being negative, record it as true positive
        if probPos > probNeg:
            truePos += 1
        else:
            falseNeg += 1
	# test model on the negative instances in the test data
    for negInstance in [dict(Counter(i)) for i in negTest]:
        probPos = probabiltyCalc(posTrain, negTrain, posDict, negDict, vocab, 'positive', negInstance, logProb, laplaceSmooth, smoothConst)
        probNeg = probabiltyCalc(posTrain, negTrain, posDict, negDict, vocab, 'negative', negInstance, logProb, laplaceSmooth, smoothConst)
        # if the probability of the instance being negative > probability of it being positive, record it as true negative
        if probPos < probNeg:
            trueNeg += 1
        else:
            falsePos += 1
    return truePos, trueNeg, falsePos, falseNeg

# creates a confusion matrix to evaluate the performance of the model
# truePosi: int : count of true positive instances; # trueNega: int : count of true negative instances; falsePosi: int : count of false positive instances; falseNega: int : count of false negative instances
def confusionMatrix(truePosi, trueNega, falsePosi, falseNega, title=""):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.axis('off')
    table_data = [
        ["", "Predicted Positive", "Predicted Negative"],
        ["Actual Positive", truePosi, falseNega],
        ["Actual Negative", falsePosi, trueNega]
    ]
    table = ax.table(cellText=table_data, loc='center')
    table.auto_set_font_size(True)
    table.scale(1, 4)
    plt.show()
    
# confusionMatrix(1,2,3,4, title="HelloWorld")
#-----------------------------------------------------------------------------------------------------------------------------------------

# Evaluation Metrics 

# calculate accuracy
def accuracy(truePosi, trueNega, falsePosi, falseNega): 
	return (truePosi + trueNega) / (truePosi + trueNega + falseNega + falsePosi)

# calculate precision
def precision(truePosi, falsePosi):
	if (truePosi + falsePosi) == 0: return 0
	preposi = truePosi / (truePosi + falsePosi)
	return preposi

# calculate recall
def recall(truePosi, falseNega):
	if (truePosi + falseNega)== 0: return 0
	recposi = truePosi / (truePosi + falseNega)
	return recposi

# return all evaluation metrics and their respective results
def evaluate_model(truePos, trueNeg, falsePos, falseNeg, printRes=True):
    acc = accuracy(truePos, trueNeg, falsePos, falseNeg)
    pre = precision(truePos, falsePos)
    rec = recall(truePos, falseNeg)
    if printRes:
        print("Accuracy		: ", format(acc,".6f"))
        print("Precision	: ", format(pre,".6f"))
        print("Recall		: ", format(rec,".6f"))
        
# evaluate_model(500, 100, 500, 100, printRes:bool=True)

#-----------------------------------------------------------------------------------------------------------------------------------------