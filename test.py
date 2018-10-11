import tensorflow as tf
import numpy as np
from util import process_image, plot_image
from model import model_generator
from keras.models import load_model
from cv2 import cv2
import matplotlib.pyplot as plt
drawing = False # true if mouse is pressed
ix,iy = -1,-1
color = (1,1,1)
size = 10
imgae_size = (256, 256)

def erase_img(temp_img):

    # mouse callback function
    def erase_rect(event,x,y,flags,param):
        global ix,iy,drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            if drawing == True:
                cv2.rectangle(temp_img,(x-size,y-size),(x+size,y+size),color,-1)
                cv2.rectangle(mask,(x-size,y-size),(x+size,y+size),color,-1)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.rectangle(temp_img,(x-size,y-size),(x+size,y+size),color,-1)
                cv2.rectangle(mask,(x-size,y-size),(x+size,y+size),color,-1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(temp_img,(x-size,y-size),(x+size,y+size),color,-1)
            cv2.rectangle(mask,(x-size,y-size),(x+size,y+size),color,-1)
   
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',erase_rect)
    cv2.setMouseCallback('mask',erase_rect)
    mask = np.zeros((256, 256, 1), dtype=np.uint8)
    while(1):
        img_show = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
        cv2.imshow('image',img_show)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return mask

def test(generator):
    filenames = ["1.jpg", "2.jpg" ,"3.jpg"]
    # orignal image for predict without mask
    for filename in filenames:
        img = process_image('test/' + filename)
        ## image for dreawing 
        temp_img = process_image('test/' + filename)
        print("Testing ...")
        mask = erase_img(temp_img)
        
        img = np.expand_dims(img, 0)
        mask = np.expand_dims(mask, 0)

        completion_image = generator.predict([img, mask])

        # # Delete Batch dimension
        completion_image = np.squeeze(completion_image, 0)
        img = np.squeeze(img, 0)

        #cv2 show
        #completion_image = cv2.cvtColor(completion_image, cv2.COLOR_BGR2RGB)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(6, 3))
        plot_image(temp_img, 'Input', 1)
        plot_image(completion_image, 'Output', 2)
        plot_image(img, 'Ground Truth', 3)
        plt.savefig("result/" + filename.split('.')[0] + "_test")
        plt.show()
        

        # cv2.imshow("result",completion_image)
        # cv2.waitKey()
        print("Done.....")

def main():
    generator = load_model('model/generator.h5')
    test(generator)

if __name__ == "__main__":
    main()
