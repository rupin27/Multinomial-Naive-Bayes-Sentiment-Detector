from run import *
import matplotlib.pyplot as plt

#Test 1

truePos1_1, trueNeg1_1, falsePos1_1, falseNeg1_1 = naiveBayes(0.2, 0.2, 0.2, 0.2, logProb=False, laplaceSmooth=False)
confusionMatrix(truePos1_1, trueNeg1_1, falsePos1_1, falseNeg1_1, "Naive Bayes without log-probabilities")

# Repeat the experiment, with log and no laplace smoothing
truePos1_2, trueNeg1_2, falsePos1_2, falseNeg1_2 = naiveBayes(0.2, 0.2, 0.2, 0.2, logProb=True, laplaceSmooth=False)
confusionMatrix(truePos1_2, trueNeg1_2, falsePos1_2, falseNeg1_2, "Naive Bayes with log-probabilities")

#Test2
truePos2_1, trueNeg2_1, falsePos2_1, falseNeg2_1 = naiveBayes(0.2, 0.2, 0.2, 0.2, logProb=True, laplaceSmooth=True, smoothConst=1)
confusionMatrix(truePos2_1, trueNeg2_1, falsePos2_1, falseNeg2_1, "Naive Bayes with Lapace Smoothing (alpha = 1)")

alpha = 0.0001
alphaVals = []
accuracyVals = []
while alpha <= 1000:
    truePos2_2, trueNeg2_2, falsePos2_2, falseNeg2_2 = naiveBayes(0.2, 0.2, 0.2, 0.2, logProb=True, laplaceSmooth = True, smoothConst = alpha)
    accuracyVals.append(accuracy(truePos2_2, trueNeg2_2, falsePos2_2, falseNeg2_2))
    alphaVals.append(alpha)
    alpha*=10

plt.xscale("log")
plt.xlabel("Alpha Value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Alpha")
plt.plot(alphaVals, accuracyVals)
plt.show

# Test3

truePos3, trueNeg3, falsePos3, falseNeg3 = naiveBayes(1, 1, 1, 1, logProb=True, laplaceSmooth=True, smoothConst=10)
confusionMatrix(truePos3, trueNeg3, falsePos3, falseNeg3, 'Naive Bayes on Data Set with alpha = 10')

# Test4

truePos4, trueNeg4, falsePos4, falseNeg4 = naiveBayes(0.5, 0.5, 1, 1, logProb=True, laplaceSmooth=True, smoothConst=10)
confusionMatrix(truePos4, trueNeg4, falsePos4, falseNeg4, r'Naive Bayes on Data Set with alpha = 10 and 50% of the training set')

# Test5

truePos6, trueNeg6, falsePos6, falseNeg6 = naiveBayes(0.1, 0.5, 1, 1, logProb=True, laplaceSmooth=True, smoothConst=1)
confusionMatrix(truePos6, trueNeg6, falsePos6, falseNeg6, 'Naive Bayes on Data Set with alpha = 1')