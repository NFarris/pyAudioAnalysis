from pyAudioAnalysis import audioTrainTest as aT
from pathlib import Path

directory_in_str = "emotionData/Big4"
pathlist = Path(directory_in_str).glob('**/*.wav')

listOfDirs = []
for path in pathlist:
    path_in_str = str(path)
    parent = path.parents[0]
    if str(parent) not in listOfDirs:
        listOfDirs.append(str(parent))

mtWin = 1.0
mtStep = 1.0
stWin = aT.shortTermWindow
stStep = aT.shortTermStep
classifierType = "extratrees"
modelName = "emotionModels/et_big4_1.0"
beat = True

aT.extract_features_and_train(listOfDirs, mtWin, mtStep, stWin, stStep, classifierType, modelName, beat)
