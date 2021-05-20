# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
import pickle
import sys

from binarytree import bTree
from trainingSet import TrainingSet


def getFeatureSet():
    featureset = list()
    artical = list()
    artical.append("a")
    artical.append("an")
    artical.append("the")
    pronoun = list()
    pronoun.append("I")
    pronoun.append("me")
    pronoun.append("we")
    pronoun.append("you")
    pronoun.append("he")
    pronoun.append("she")
    pronoun.append("it")
    pronoun.append("him")
    pronoun.append("her")
    pronoun.append("they")
    pronoun.append("them")
    pronoun.append("any")
    pronoun.append("most")
    pronoun.append("some")
    pronoun.append("all")
    pronoun.append("this")
    pronoun.append("that")
    pronoun.append("these")
    pronoun.append("those")
    preposition = list()
    preposition.append("on")
    preposition.append("in")
    preposition.append("for")
    preposition.append("near")
    preposition.append("under")
    preposition.append("above")
    preposition.append("of")
    preposition.append("to")
    preposition.append("opposite")
    conjunction = list()
    conjunction.append("and")
    conjunction.append("nor")
    conjunction.append("but")
    conjunction.append("or")
    conjunction.append("so")
    conjunction.append("because")
    conjunction.append("after")
    conjunction.append("before")
    conjunction.append("even")
    conjunction.append("since")
    conjunction.append("unless")
    conjunction.append("as")
    auxiliary = list()
    auxiliary.append("am")
    auxiliary.append("are")
    auxiliary.append("is")
    auxiliary.append("was")
    auxiliary.append("were")
    auxiliary.append("being")
    auxiliary.append("can")
    auxiliary.append("could")
    auxiliary.append("do")
    auxiliary.append("did")
    auxiliary.append("may")
    auxiliary.append("shall")
    auxiliary.append("should")
    auxiliary.append("will")
    auxiliary.append("would")
    featureset.append(artical)
    featureset.append(pronoun)
    featureset.append(preposition)
    featureset.append(conjunction)
    featureset.append(auxiliary)
    return featureset


def readFile(fileName):
    trainSetData = list()
    featureset = getFeatureSet()
    with open(fileName, "r", encoding='utf-8') as f:
        id = 0
        for line in f:
            if len(line) < 10:
                continue
            featurevalues = list()
            for feature in featureset:
                iscontainFeature = False
                for attr in line.strip().split():
                    if attr in feature and len(feature) == len(attr):
                        iscontainFeature = True
                        break
                featurevalues.append(iscontainFeature)
            trainingDataSet = TrainingSet(featurevalues, line.split("|")[0].strip(), 1.0, id)
            id += 1
            trainSetData.append(trainingDataSet)
    return trainSetData


def readModelFile(fileName):
    trainSetData = list()
    featureset = getFeatureSet()
    with open(fileName) as f:
        id = 0
        for line in f:
            if len(line) < 10:
                continue
            featurevalues = list()
            for feature in featureset:
                iscontainFeature = False
                for attr in line.strip().split():
                    if attr in feature:
                        iscontainFeature = True
                        break
                featurevalues.append(iscontainFeature)
            trainSetData.append(featurevalues)
    return trainSetData


def readDecisionTree(tree, trainSetData):
    for data in trainSetData:
        print(predictLanguage(data, tree))


def readadaBoostDecisionTree(adatree, trainSetData):
    for data in trainSetData:
        totalweightEn = 0
        totalweightnl = 0
        for tree in adatree:
            if predictLanguage(data, adatree[tree][0]) == "en":
                totalweightEn += adatree[tree][1]
            else:
                totalweightnl += adatree[tree][1]
        print("en") if totalweightnl < totalweightEn else print("nl")


def predictLanguage(data, tree):
    if tree.left is None and tree.right is None:
        return tree.label
    if data[tree.value]:
        return predictLanguage(data, tree.left)
    else:
        return predictLanguage(data, tree.right)


def calculatefptree(depth, trainSetData, pAttr, tree, isBoosted, iSTrueSet):
    if depth > 1000 and isBoosted:
        tree = bTree(-1, majorityelement(trainSetData), trainSetData, iSTrueSet)
        return tree
    else:
        attrmEntropyfound = -1
        minreminder = 1
        for attr in range(trainSetData[0].getTrainingDataAttrSize()):
            if pAttr.get(attr) == attr:
                continue
            total_weight = 0
            true_a_size = 0
            true_size = 0
            false_a_size = 0
            false_size = 0
            reminderT = 0
            reminderF = 0
            for trainSet in trainSetData:
                total_weight += trainSet.getWeight()
                if trainSet.getTrainingDataAttrValue(attr):
                    true_size += trainSet.getWeight()
                    if trainSet.getLabel() == "en":
                        true_a_size += trainSet.getWeight()
                else:
                    false_size += trainSet.getWeight()
                    if trainSet.getLabel() == "en":
                        false_a_size += trainSet.getWeight()
            if true_size != 0:
                reminderT = (true_size / total_weight) * calcentropy(true_a_size / true_size)
            if false_size != 0:
                reminderF = (false_size / total_weight) * calcentropy(false_a_size / false_size)
            if minreminder >= reminderT + reminderF:
                minreminder = reminderT + reminderF
                attrmEntropyfound = attr
        trainsetT = list()
        trainsetF = list()
        if attrmEntropyfound != -1:
            for trainSet in trainSetData:
                if trainSet.getTrainingDataAttrValue(attrmEntropyfound):
                    trainsetT.append(trainSet)
                else:
                    trainsetF.append(trainSet)
            pAttr[attrmEntropyfound] = attrmEntropyfound
            tree = bTree(attrmEntropyfound, majorityelement(trainSetData), trainSetData if isBoosted else None,
                         iSTrueSet)
            if minreminder != 0:
                if len(trainsetT) == 0 or len(trainsetF) == 0:
                    pass
                else:
                    tree.left = calculatefptree(depth + 1, trainsetT, pAttr.copy(), tree.left, isBoosted, True)
                    tree.right = calculatefptree(depth + 1, trainsetF, pAttr.copy(), tree.right, isBoosted, False)
        return tree


def generatedecisionstumptree(tree, generatedStump):
    if tree.left is None and tree.right is None:
        for trainData in tree.getTrainSetData():
            if tree.getisTrueSet():
                if trainData.label == "en":
                    generatedStump[trainData.getId()] = True
                else:
                    generatedStump[trainData.getId()] = False
            else:
                if trainData.label == "en":
                    generatedStump[trainData.getId()] = False
                else:
                    generatedStump[trainData.getId()] = True
    else:
        generatedStump = generatedecisionstumptree(tree.left, generatedStump)
        generatedStump = generatedecisionstumptree(tree.right, generatedStump)
    return generatedStump


def adaboost(trainingSetData, k):
    hypothysys = dict()
    for setdata in trainingSetData:
        setdata.setWeight(len(trainingSetData))
    for val in range(k):
        tree = calculatefptree(1, trainingSetData, dict(), None, True, True)
        hyp = generatedecisionstumptree(tree, dict())
        correctError = 0
        WrongError = 0
        for setdata in trainingSetData:
            if hyp[setdata.getId()]:
                correctError += setdata.getWeight()
            else:
                WrongError += setdata.getWeight()
        computeError = WrongError / correctError
        sum = 0.0
        for setdata in trainingSetData:
            if hyp[setdata.getId()]:
                sum += setdata.getWeight()
            else:
                sum += (setdata.getWeight() * computeError)
        logcalc = 0
        if computeError != 0:
            logcalc = math.log(abs(1 - computeError) / computeError)
        tree = calculatefptree(1, trainingSetData, dict(), None, False, True)
        hypothysys[val] = tree, logcalc
        trainingSetData = normalizeWeight(sum, trainingSetData)
    return hypothysys


def normalizeWeight(sum, trainingSetData):
    for setdata in trainingSetData:
        setdata.setWeight(sum)
    return trainingSetData


def majorityelement(trainSetData):
    eleTSize = 0
    eleFSize = 0
    for trainSet in trainSetData:
        if trainSet.getLabel() == "en":
            eleTSize += trainSet.weight
        else:
            eleFSize += trainSet.weight
    return "en" if eleFSize < eleTSize else "nl"


def calcentropy(q):
    if q == 0 or q == 1:
        return 0
    return -1 * (q * math.log(q, 2) + (1 - q) * math.log((1 - q), 2))


if __name__ == '__main__':
    pAttr = dict()
    if sys.argv[1] == 'train':
        if sys.argv[4] == 'dt':
            tree = calculatefptree(0, readFile(sys.argv[2]), pAttr, None, False, True)
        else:
            tree = adaboost(readFile(sys.argv[2]), 10)
        with open(sys.argv[3], 'wb') as file:
            pickle.dump(tree, file)
    else:
        with open(sys.argv[2], 'rb') as file:
            tree = pickle.load(file)
        if isinstance(tree, bTree):
            readDecisionTree(tree, readModelFile(sys.argv[3]))
        else:
            readadaBoostDecisionTree(tree, readModelFile(sys.argv[3]))
