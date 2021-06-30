import os

workingDir = 'midicsv-1.1'
inputDir = '' #specify csv folder
outputDir = '' #specify output folder
os.chdir(workingDir)

csvList = os.listdir(inputDir)

for csv in csvList:
    os.system('csvmidi ' + ' "' + inputDir + csv + '" "' + outputDir + os.path.splitext(csv)[0] + '.mid"')