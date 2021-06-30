import csv
import os
import math
from PIL import Image as img
from numpy import asarray

bpm = 110
micropqn = 60000000//bpm
ppsthn = 480/4

inputDir = "C:\\Users\\ThomH\\Documents\\Kunstmatige Intelligentie\\Jaar 4\\Scriptie\\GAN\\Dataset\\data\\data_img"
outputDir = "C:\\Users\\ThomH\Documents\\Kunstmatige Intelligentie\\Jaar 4\\Scriptie\\GAN\\Dataset\\data\\data_csv"
imgList = os.listdir(inputDir)

noteIndexDict = {
    0 : "47",
    1 : "50",
    2 : "43",
    3 : "38",
    4 : "37",
    5 : "36",
    6 : "46",
    7 : "42",
    8 : "51",
    9 : "49"
}

def importImage(filename):
    os.chdir(inputDir)
    testimg = img.open(filename)
    array = asarray(testimg)
    return array

def getNote(row):
    return noteIndexDict[row]

def writeNoteRows(filewriter, filename):
    array = importImage(filename)
    time = 0
    array = array[3:12,:]
    for column in range(array.shape[1]):
        for row in range(array.shape[0]):
            if array[row,column] != 0:
                velocity = int(math.ceil((array[row,column]/255) * 127))
                filewriter.writerow(['1', str((round(time))), 'Note_on_c', '9', getNote(row), str(velocity)])
        time = time + ppsthn
    filewriter.writerow(['1', str((round(time)+1)), 'End_track'])

def main(filename):
    os.chdir(outputDir)
    with open(os.path.splitext(filename)[0]+'.csv', 'w',  newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['0', '0', 'Header', '0', '1', '480'])
        filewriter.writerow(['1', '0', 'Start_track'])
        filewriter.writerow(['1', '0', 'Tempo', str(micropqn)])
        filewriter.writerow(['1', '0', 'Time_signature', '4', '2', '24', '8'])
        filewriter.writerow(['1', '0', 'Key_signature', '0', '"major"'])
        writeNoteRows(filewriter, filename)
        filewriter.writerow(['0', '0', 'End_of_file'])



for image in imgList:
    main(image)