class bTree:
    __slots__ = "value", "left", "right", "label", "trainSetData", "isTrueSet"

    def __init__(self, value, label, trainSetData, isTrueSet):
        self.value = value
        self.left = None
        self.right = None
        self.label = label
        self.trainSetData = trainSetData
        self.isTrueSet = isTrueSet

    def getisTrueSet(self):
        return self.isTrueSet

    def getTrainSetData(self):
        return self.trainSetData

    def __str__(self):
        return self.label
