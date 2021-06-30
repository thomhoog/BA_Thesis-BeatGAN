import csv
import os
import math
import numpy as np
from PIL import Image as im
from numpy.lib.function_base import average

inputDir = "" #specify input directory path
outputDir = "" #specify output directory path

os.chdir(inputDir)
csvList = os.listdir(inputDir)
kickWeight = 100
snareWeight = 1
noteIndexDict = {
    "45" : 0,
    "47" : 0,
    "48" : 1,
    "50" : 1,
    "43" : 2,
    "58" : 2,
    "38" : 3,
    "40" : 3,
    "37" : 4,
    "36" : 5,
    "46" : 6,
    "26" : 6,
    "42" : 7,
    "22" : 7,
    "44" : 7,
    "51" : 8,
    "59" : 8,
    "53" : 8,
    "49" : 9,
    "55" : 9,
    "57" : 9,
    "52" : 9
}

def createArray(sth_notes):
    array = np.zeros(sth_notes * 10)
    array = np.reshape(array, (10, sth_notes))
    # show the array
    return array

def stripCSV(csvreader):
    rows = []
    for row in csvreader:
        stripped = [entry.strip() for entry in row]
        rows.append(stripped)
    return rows

#gets time in MIDI clocks
def getTotalTime(rows):
    x = -1
    while (rows[x][2] != 'Note_on_c'):
        x = x - 1
    total_time = int(rows[x][1]) + 1
    return total_time

def noteToIndex(entry):
    return noteIndexDict[entry[4]]

def processVelocity(velocity):
    return int((float(velocity)/127) * 255)

#loops through all 16th notes, quantizes notes from csv file to their respective place in the array
def fillArray(array, rows, sth_notes):
    entry_counter = 0
    for x in range(sth_notes):
        for y in range(entry_counter, len(rows)): #starts from entry_counter to remember where in the csv file we are at
            entry = rows[y]
            if (entry[2] == 'End_track'):
                break
            elif (entry[2] == 'Note_on_c'):
                if (int(entry[1]) < ((x+1) * 120)): #check if note time (in midi pulses) is smaller than the next 16th
                    array[noteToIndex(entry),x] = processVelocity(entry[5])
                else: #if not, proceed to next 16th note
                    break
            entry_counter = entry_counter + 1 #update index for current entry

def stripArray(array):
    columnstart = 0
    for i in range(array.shape[1]):
        if not (0 in array[:,i]):
            columnstart = i
            break
    j = -1
    while (sum(array[:,j]) == 0):
        j = j - 1
    return array[:,columnstart:(array.shape[1] + j + 1)]

def findSnares(columns):
    if len(columns[0]) < 5:
        return False
    else:
        if (columns[3,4] > 0) or (columns[4,4] > 0):
            return True
        else:
            return False

def sliceArray(array):
    images = []
    partitions = math.floor(len(array[0]) / 16)
    for i in range(partitions):
        range_start = i*16
        range_end = range_start + 16
        images.append(array[:,range_start:range_end])
    return images

def kickCorrection(array, i):
    if (array[5,i] > 0) and (findSnares(array[:,i:])):
        return (array[5,i] * kickWeight) - array[5,i]
    return 0

def getVelocities(array):
    velocities = []
    for i in range(len(array[0])):
        velocities.append(sum(array[:,i])+kickCorrection(array, i)+array[5,i]*5)
    return velocities

def getTotalPeaks(velocities):
    peaksTotal = []
    avgvel = average(velocities)
    for velocity in velocities:
        if velocity > 0.95 * avgvel:
            peaksTotal.append(1)
        else:
            peaksTotal.append(0)
    return peaksTotal

def getPeakCount(peaksIndex):
    count = 1
    for i in range(0,len(peaksIndex),4):
        if peaksIndex[i] == 1:
            count = count + 1
    return count

def getPeaksIndex(peaksTotal):
    count = 0
    index = 0
    maxPeakCount = 0
    for i in range(len(peaksTotal)):
        if (count == 10):
            break
        if peaksTotal[i] == 1:
            count = count + 1
            newPeakCount = getPeakCount(peaksTotal[i:])
            if (maxPeakCount < newPeakCount):
                index = i
                maxPeakCount = newPeakCount
    return index

def findSnares2(column):
    if (column[3] > 0) or (column[4] > 0):
        return True
    else:
        return False

# selects the most likely downbeat from the index provided
# checks if index is a snare, if so, return index - 4
# check if index is kick, return index
def selectDbIndex(array, beatgrid):
    for i in range(len(beatgrid)):
        if beatgrid[i] == 1:
            if findSnares2(array[:,i]):
                return i-4
            elif array[5,i] > 0:
                return i
    return None

def getBeatgrid(length, index):
    beatgrid = np.zeros(length)
    for i in range(index,length,4):
        beatgrid[i] = 1
    for i in range(index,-1,-4):
        beatgrid[i] = 1
    return beatgrid

def getBeatEnd(length):
    return ((length // 16) * 16)

def sliceArray2(array):
    return array[:,0:getBeatEnd(len(array[0]))]

def main(filename):
    c = open(filename, 'r')
    csvreader = csv.reader(c)
    rows = stripCSV(csvreader) #remove leading and trailing whitespace
    total_time = getTotalTime(rows)
    total_time = math.ceil(total_time/120) * 120 #gives us the total time in ppqn, divisible by 120 (ppsthn = ppqn/4)
    sth_notes = int(total_time / 120) #gives us the total number of 16th notes
    if (sth_notes >= 16): #only accept MIDIs larger than 16 16th notes
        print(filename)
        array = createArray(sth_notes) #create numpy array equal to the total length in 16th notes
        fillArray(array, rows, sth_notes)
        #array = stripArray(array)
        velocities = getVelocities(array)
        peaksTotal = getTotalPeaks(velocities)
        beatIndex = getPeaksIndex(peaksTotal)
        beatgrid = getBeatgrid(len(velocities), beatIndex)
        dbIndex = selectDbIndex(array, beatgrid)
        if dbIndex != None:
            if dbIndex < 0:
                prefix = createArray(abs(dbIndex))
                prefix[5,0] = 255
                array = np.hstack([prefix, array])
            else:
                array = array[:,dbIndex:]
            array = sliceArray2(array)
            if len(array[0]) > 0:
                if array[5,0] > 0:
                    imgArray = sliceArray(array)
                    for img in range(len(imgArray)):
                        imgArray[img] = np.pad(imgArray[img], ((3, 3),(0,0)), 'constant')
                        data = im.fromarray(imgArray[img])
                        data = data.convert("L")
                        data.save(os.path.splitext(outputDir + "\\" +filename)[0] + "_" + str(img+1) + '.png')
    c.close()

for file in csvList:
    main(file)