import os

workingDir = 'midicsv-1.1'
inputDir = '' #specify input folder path
outputDir = '' #specify output folder path
os.chdir(workingDir)

midiList = os.listdir(inputDir)

for midi in midiList:
    os.system('midicsv ' + ' "' + inputDir + midi + '" "' + outputDir + os.path.splitext(midi)[0] + '.csv"')