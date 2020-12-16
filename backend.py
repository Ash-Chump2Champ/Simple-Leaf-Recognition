import os
import numpy as np
import pandas as pd
import mahotas as mt
from matplotlib import pyplot as plt

import cv2
import my_gui
''' 
Training of Dataset
'''
class driverprogram():

# Load dataset
    dataset = pd.read_csv('Leaf_Features.csv')

    # Accessing csv file X is the values we want to train and Y are the labels we want to predict
    X = dataset.iloc[:,1:].values
    y = dataset.iloc[:,:1].values

    # Splitting the data with 20% of data going to be tested and 80% are going for training
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Scaling of the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Applying KNN classifier Model
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5, p=2, weights ="distance")
    classifier.fit(X_train, np.ravel(y_train) )
    y_pred = classifier.predict(X_test)

    # Backgroung subtract(Start)

    def bg_sub(self,filename):
        test_img_path = filename
        main_img = cv2.imread(test_img_path)
        img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(img, (1600, 1200))
        size_y,size_x,_ = img.shape
        gs = cv2.cvtColor(resized_image,cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gs, (55,55),0)
        ret_otsu,im_bw_otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        kernel = np.ones((50,50),np.uint8)
        closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)
        
        contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        contains = []
        y_ri,x_ri, _ = resized_image.shape
        for cc in contours:
            yn = cv2.pointPolygonTest(cc,(x_ri//2,y_ri//2),False)
            contains.append(yn)

        val = [contains.index(temp) for temp in contains if temp>0]
        index = val[0]
        
        black_img = np.empty([1200,1600,3],dtype=np.uint8)
        black_img.fill(0)
        
        cnt = contours[index]
        mask = cv2.drawContours(black_img, [cnt] , 0, (255,255,255), -1)
        
        maskedImg = cv2.bitwise_and(resized_image, mask)
        white_pix = [255,255,255]
        black_pix = [0,0,0]
        
        final_img = maskedImg
        h,w,channels = final_img.shape
        for x in range(0,w):
            for y in range(0,h):
                channels_xy = final_img[y,x]
                if all(channels_xy == black_pix):
                    final_img[y,x] = white_pix
        
        return final_img


    def feature_extract(self,img):
        names = ['area','perimeter','pysiological_length','pysiological_width','aspect_ratio','rectangularity','circularity', \
                 'mean_r','mean_g','mean_b','stddev_r','stddev_g','stddev_b', \
                 'contrast','correlation','inverse_difference_moments','entropy'
                ]
        df = pd.DataFrame([], columns=names)

        #Preprocessing
        gs = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gs, (25,25),0)
        ret_otsu,im_bw_otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        kernel = np.ones((50,50),np.uint8)
        closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)
        contours ,_ = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        #Shape features
        cnt = contours[0]
        M = cv2.moments(cnt)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)
        x,y,w,h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        rectangularity = w*h/area
        circularity = ((perimeter)**2)/area

        #Color features
        red_channel = img[:,:,0]
        green_channel = img[:,:,1]
        blue_channel = img[:,:,2]
        blue_channel[blue_channel == 255] = 0
        green_channel[green_channel == 255] = 0
        red_channel[red_channel == 255] = 0

        red_mean = np.mean(red_channel)
        green_mean = np.mean(green_channel)
        blue_mean = np.mean(blue_channel)

        red_std = np.std(red_channel)
        green_std = np.std(green_channel)
        blue_std = np.std(blue_channel)

        #Texture features
        textures = mt.features.haralick(gs)
        ht_mean = textures.mean(axis=0)
        contrast = ht_mean[1]
        correlation = ht_mean[2]
        inverse_diff_moments = ht_mean[4]
        entropy = ht_mean[8]

        vector = [area,perimeter,w,h,aspect_ratio,rectangularity,circularity,\
                  red_mean,green_mean,blue_mean,red_std,green_std,blue_std,\
                  contrast,correlation,inverse_diff_moments,entropy
                 ]

        df_temp = pd.DataFrame([vector],columns=names)
        df = df.append(df_temp)
        
        return df

    # Feature Extraction(End)

    def train_file(self, filename):
        # Main Program
        #filename = r'Leaf Dataset\\Acer Palmatum\\1268.jpg'
        bg_rem_img = self.bg_sub(filename)
        features_of_img = self.feature_extract(bg_rem_img)
        scaled_features = self.scaler.transform(features_of_img)
        predicted_plant = self.classifier.predict(scaled_features)
        plant = my_gui.ScreenThree()
        plant.plant_name(predicted_plant)

