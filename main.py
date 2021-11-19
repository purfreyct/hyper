import os
import cv2
import math
import xlrd
import gdal
import keras
import numpy as np
from spectral import *
from keras import layers, models
from keras.models import Model
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPool2D, LeakyReLU, UpSampling2D, Dropout, Dense, Input, GlobalAveragePooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib



''' ----------------------PREDEFINED FUNCTIONS------------------------- '''


# read BIL
def ReadBilFile(bil):
    img = gdal.Open(bil) # input image
    bands = img.RasterCount
    cols = img.RasterXSize
    rows = img.RasterYSize
    data = np.zeros(shape=(rows, cols, bands)) # build a empty array to store the bil files
    for band in range(bands):
        this_band = img.GetRasterBand(band+1)
        r = this_band.ReadAsArray()
        for col in range(cols):
            for row in range(rows):
                data[row,col,band] = r[row,col]
    return data

# Read all files in the folder
def get_bil_file(path, rule):
    all = []
    for fpathe,dirs,fs in os.walk(path):   # os.walk obtain all files in the menu
        for f in fs:
            filename = os.path.join(fpathe,f)
            if filename.endswith(rule):  # check if it is ending with '.sfx'
                all.append(filename)
    return all

# import the pv or ffa value in the excel table
def import_excel_matrix_PV(path):
    table = xlrd.open_workbook(path).sheets()[0]  # read first page
    row = table.nrows
    col = table.ncols
    datamatrix = np.zeros((row, col))
    for i in range(col):
        cols = np.matrix(table.col_values(i))
        datamatrix[:, i] = cols
    return datamatrix

# Create after PCA BIL
def PC_reduce_dimension(array, fraction):
    pc = principal_components(array) #PCA
    pc_fraction = pc.reduce(fraction=fraction)
    PCArray = pc_fraction.transform(array)
    return PCArray

# Count number of cropped images
def Count_HSI(PVtable, length):
    count = 0
    for x in range(0,length):
        if PVtable[x, 1] > 1:
            count = count + 40
        else:
            count = count + 20
    return count

# obtain sub-image topleft coordinate array of each whole image
def LeftTopPixel(bil_array, bil_category):
    num = 0
    max_row = bil_array.shape[0]
    if bil_category > 1:
        a = np.zeros((40,2), dtype=int)
        for row in range(max_row-60):
            if num < 40:
                for col in range(1540):
                    if num < 40:
                        c = 0
                        for b in range(num):
                            if (abs(row - a[b, 0]) < 50) and (abs(col - a[b, 1]) < 50):
                                c = c + 1
                                break
                        if bil_array[row:row+60, col:col+60, 461].min()>2000 and ((c == 0) or (num == 0)):
                            a[num, 0] = row
                            a[num, 1] = col
                            num += 1
                    else:
                        break
            else:
                break
    else:
        a = np.zeros((20,2), dtype=int)
        for row in range(max_row-60):
            if num < 20:
                for col in range(1540):
                    if num < 20:
                        c = 0
                        for b in range(num):
                            if (abs(row - a[b, 0]) < 50) and (abs(col - a[b, 1]) < 50):
                                c = c + 1
                                break
                        if bil_array[row:row+60, col:col+60, 461].min()>2000 and ((c == 0) or (num == 0)):
                            a[num, 0] = row
                            a[num, 1] = col
                            num += 1
                    else:
                        break
            else:
                break
    return a

# Create PV category table after cropping
def CreateCategoryArray(length, full_length, PVtable):
    y = np.empty(shape=[full_length,], dtype=int)
    highPV_y = 0
    for rows in range(0,length):
        lowPV_y = rows - highPV_y
        if PVtable[rows, 1] > 1:
            for copy_rows in range(0,40):
                y[copy_rows + lowPV_y*20 + highPV_y*40,] = PVtable[rows, 2]
            highPV_y = highPV_y + 1
        else:
            for copy_rows in range(0,20):
                y[copy_rows + lowPV_y*20 + highPV_y*40,] = PVtable[rows, 2]
    return y


''' ----------------------PREPROCESSING TEST DATASETS------------------------- '''


#folder = get_bil_file(r'F:\research_project\testa_on_canarium','.bil')
folder = get_bil_file(r'/data/projects/punim1138/testa_on_canarium','.bil') # Import BIL folder

#PV_table = import_excel_matrix_PV(r'D:\Courses\Research_project\coding\keras\all_in_one\testa_on_PV_ffa.xlsx')
PV_table = import_excel_matrix_PV(r'/data/projects/punim1138/testa_on_canarium/testa_on_PV_ffa.xlsx') # Import label table

length = len(folder) #number of raw input HSIs

full_length = Count_HSI(PV_table, length) # number of cropped input HSIs

PCAbands = 8 # band number of PCA

X = np.zeros((full_length, 60, 60, PCAbands))

y = CreateCategoryArray(length, full_length, PV_table) # create y dataset

page_order = 0

# read all sub-images in a whole array
for order in range(length):
    bil_array = ReadBilFile(folder[order])
    bil_pc = PC_reduce_dimension(bil_array, 0.999)
    bil_category = PV_table[order, 1]
    topleft_array = LeftTopPixel(bil_array, bil_category)
    for sub_order in range(topleft_array.shape[0]):
        start_row = topleft_array[sub_order, 0]
        end_row = start_row + 60
        start_col = topleft_array[sub_order, 1]
        end_col = start_col + 60
        X[page_order, :, :, :] = bil_pc[start_row:end_row, start_col:end_col, 0:PCAbands]        
        page_order = page_order + 1

print('Max value of X is:', X.max())
print('Min value of X is:', X.min())

#X = cv2.normalize(X, 0, 1, norm_type=cv2.NORM_MINMAX) #normalization X
np.save("X_PCA_8.npy", X)
X = (X + 150000) / 300000 # fixed value normalization X, or you can change to minmax normalization but make sure all values are positive

y = np_utils.to_categorical(y, num_classes=3) # recategorize
np.save("y_PCA_8.npy", y)


X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3) # TRAIN AND validation/test SPLIT
X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.333) # validation AND test SPLIT


''' ----------------------CLASSIFICATION MODEL------------------------- '''


classification = models.Sequential(name='classifier')
classification.add(layers.Conv2D(16, (1, 1), padding='valid', kernel_initializer='glorot_uniform', bias_initializer='zeros', input_shape=(60, 60, PCAbands))) 
classification.add(LeakyReLU(alpha=0.05))
classification.add(BatchNormalization())
classification.add(layers.MaxPooling2D((2, 2), padding='valid'))
classification.add(layers.Conv2D(32, (1, 1), padding='valid'))
classification.add(LeakyReLU(alpha=0.05))
classification.add(BatchNormalization())
classification.add(layers.MaxPooling2D((2, 2), padding='valid'))
classification.add(layers.Conv2D(64, (1, 1), padding='valid'))
classification.add(LeakyReLU(alpha=0.05))
classification.add(BatchNormalization())
classification.add(layers.MaxPooling2D((5, 5), padding='valid'))
classification.add(layers.Conv2D(128, (1, 1), padding='valid'))
classification.add(LeakyReLU(alpha=0.05))
classification.add(BatchNormalization())
classification.add(layers.MaxPooling2D((3, 3), padding='valid'))
classification.add(GlobalAveragePooling2D())
classification.add(Dropout(0.5))
classification.add(Dense(3, activation='softmax', name='output'))

classification.summary()
# end of the model

# iteration - fitting the model - Adam
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-3 / 200)
classification.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
history = classification.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5000, batch_size=48)

joblib.dump(classification, "final_classifier.m") #output classifier



''' ----------------------PREDICTION AND ANALYSIS------------------------- '''


classification_score = classification.evaluate(X_test, y_test)
print('Test loss:', classification_score[0])
print('Test accuracy:', classification_score[1])
y_pred = classification.predict(X_test)

confusion_matrix = np.zeros((3,3)) #row: prediction / col: truth
length = len(y_pred)
for true_category in range(3):
    for pre_category in range(3):
        for row in range(length):
            if y_pred[row, pre_category] == y_pred[row,:].max() and y_test[row, true_category] == y_test[row,:].max():
                confusion_matrix[pre_category, true_category] += 1

precision_and_recall = np.zeros((3,2))
precision_and_recall[:,1] = confusion_matrix.diagonal() / confusion_matrix.sum(axis=0)
precision_and_recall[:,0] = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)

np.set_printoptions(threshold=np.inf) # print all numbers
print('confusion matrix:\n',confusion_matrix)
print('\nrecall and precision:\n',precision_and_recall)