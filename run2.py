from utils import *
from pprint import pprint
import string
import test

def naive_bayes(posTrain=0.1, negTrain=0.1, posTest=0.1, negTest=0.1, logProb=True, laplaceSmooth=True, smoothConst=1):
    pos_Train, neg_Train, pos_Test, neg_Test, vocab = load_data(posTrain, negTrain, posTest, negTest)
    truePos, trueNeg, falsePos, falseNeg = test_model(pos_Train, neg_Train, pos_Test, neg_Test, vocab, logProb, laplaceSmooth, smoothConst)
    evaluate_model(truePos, trueNeg, falsePos, falseNeg, True)
    return truePos, trueNeg, falsePos, falseNeg

def fixdatabayes(pos_Train, neg_Train, pos_Test, neg_Test, vocab, logProb:bool=True, laplaceSmooth:bool=True, smoothConst:float=1):

	truePos, trueNeg, falsePos, falseNeg = test_model(pos_Train, neg_Train, pos_Test, neg_Test, vocab, logProb, laplaceSmooth, smoothConst)
	evaluate_model(truePos, trueNeg, falsePos, falseNeg, False)
	return truePos,trueNeg,falsePos,falseNeg

if __name__=="__main__":
	naive_bayes(0.1,0.1,0.03,0.03)


