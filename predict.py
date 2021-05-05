#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Importing libraries
import numpy as np
import sys

from keras.models import load_model
from keras.preprocessing import image

classifier = load_model('./model')


CATEGORIES = {'black_bishop': 0,
              'black_king': 1,
              'black_knight': 2,
              'black_pawn': 3,
              'black_queen': 4,
              'black_rook': 5,
              'white_bishop': 6,
              'white_king': 7,
              'white_knight': 8,
              'white_pawn': 9,
              'white_queen': 10,
              'white_rook': 11}

def WhichPiece(MyImage):
    piece = image.load_img(MyImage, target_size=(150,150),color_mode="rgb")
    piece = image.img_to_array(piece)
    piece =  np.expand_dims(piece, axis=0)
    max = np.amax(classifier.predict(piece))
    result = ((np.where(classifier.predict(piece)==max))[1].tolist())[0]
    type = [key  for (key, value) in CATEGORIES.items() if value == result][0]
    return type

#print(sys.argv[1])
print(WhichPiece(sys.argv[1]))



#test2 = image.load_img('dataset/prediction/toto6.jpg',target_size=(150,150),color_mode="rgb")
#test2 = image.img_to_array(test2)
#test2 = np.expand_dims(test2, axis=0)


#max = np.amax(classifier.predict(test2))
#result = ((np.where(classifier.predict(test2)==max))[1].tolist())[0]
#type = [key  for (key, value) in CATEGORIES.items() if value == result][0]
#print("resultat = ",type)


#classifier.predict(test2)
