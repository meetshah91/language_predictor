class TrainingSet:
    __slots__ = "id","trainingDataAttr", "weight", "label"

    def __init__(self, trainingDataAttr, label, weight,id):
        self.trainingDataAttr = trainingDataAttr
        self.weight = weight
        self.label = label
        self.id = id

    def setWeight(self, updatedWeight):
        self.weight /= updatedWeight

    def getWeight(self):
        return self.weight

    def getTrainingDataAttr(self):
        return self.trainingDataAttr

    def getTrainingDataAttrSize(self):
        return len(self.trainingDataAttr)

    def getTrainingDataAttrValue(self, value):
        return self.trainingDataAttr[value]

    def getLabel(self):
        return self.label

    def getId(self):
        return self.id


    def __str__(self):
        return self.label
