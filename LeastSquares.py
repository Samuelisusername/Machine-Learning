import csv
import numpy as np
with open('test.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    matrix = []
    resultVector = []
    for row in reader:
        rowVec = [float(row['x1']), float(row['x2']), float(row['x3']), float(row['x4']), float(row['x5']),
                  float(row['x6']), float(row['x7']), float(row['x8']), float(row['x9']), float(row['x10'])]
        matrix.append(rowVec)
        resultVector.append(float(row['y']))

    print(np.linalg.lstsq(matrix, resultVector, rcond=None))
