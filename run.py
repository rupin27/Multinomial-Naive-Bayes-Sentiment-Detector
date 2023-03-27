from utils import *

def naiveBayes(posTrain, negTrain, posTest, negTest, logProb=False, laplaceSmooth=False, smoothConst=1):
    pos_Train, neg_Train, pos_Test, neg_Test, vocab = load_data(posTrain, negTrain, posTest, negTest)
    truePos, trueNeg, falsePos, falseNeg = test_model(pos_Train, neg_Train, pos_Test, neg_Test, vocab, logProb, laplaceSmooth, smoothConst)
    evaluate_model(truePos, trueNeg, falsePos, falseNeg, True)
    return truePos, trueNeg, falsePos, falseNeg


# if __name__=="__main__":
# 	naiveBayes(0.2, 0.2, 0.2, 0.2)


