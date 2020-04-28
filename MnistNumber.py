from matplotlib import pyplot as plt
import pathlib
import numpy as np
import math

class MnistNumber:
    img_data = np.zeros([28,28])
    bw_img_data = np.zeros([28,28]) > 0
    img_lbl = 0
    num_img = 0

    guess = -1
    pixelGuess = -1
    lmsGuess = -1
    hTransitionGuess = -1
    vTransitionGuess = -1
    voterGuess = -1

    threshhold = 80
    pixel_count = 0
    h_trans_count = 0
    v_trans_count = 0
    
    # Initializes using the given number and image array
    def __init__(self, number, img = None):
        self.img_lbl = number
        if img != None:
            self.add_img(img)
        else:
            self.img_data = np.zeros([28,28])   

    # Average the given image with the current average
    def add_img(self, new_data):
        self.num_img += 1
        self.img_data += (np.reshape(new_data, (28,28)) - self.img_data) / self.num_img

        self.bw_img_data = (self.img_data > self.threshhold)
        self.pixel_count = self.bw_img_data.sum()

        h_flat = self.bw_img_data.flatten(order="C")
        v_flat = self.bw_img_data.flatten(order="F")
        
        last_v = h_flat[0]
        last_h = v_flat[0]

        self.h_trans_count = 0
        self.v_trans_count = 0
        
        for num in range(len(h_flat)):
            if (last_h != h_flat[num]):
                self.h_trans_count += 1
            if (last_v != v_flat[num]):
                self.v_trans_count += 1
            last_h = h_flat[num]
            last_v = v_flat[num]


    # Get the difference in energy of this number with the given number
    def get_energy_diff(self, com_num):
        return np.linalg.norm(com_num.img_data - self.img_data)

    def get_pixel_diff(self, com_num):
        return (((com_num.pixel_count - self.pixel_count)**2)**(1/2))

    def get_h_transition_diff(self, com_num):
        return np.linalg.norm(com_num.h_trans_count - self.h_trans_count)

    def get_v_transition_diff(self, com_num):
        return np.linalg.norm(com_num.v_trans_count - self.v_trans_count)

    # Check to see if this number is the same as the given number
    def check_num(self, num):
        return (self.img_lbl == num)

    # Show a plot of this number
    def plot(self, folder=""):
        if self.guess == -1:
            plt.title(f"Image of {self.img_lbl}")
        else:
            plt.title(f"Image of {self.img_lbl}; Guess {self.guess}")

        plt.imshow(self.img_data)

        if folder:
            pathlib.Path(f"./images/{folder}").mkdir(parents=True, exist_ok=True)
            plt.savefig(f"./images/{folder}/{self.img_lbl}")
        else:
            plt.show()

        plt.cla()
        plt.clf()

    def plotLMS(self):
        plt.figure(self.img_lbl)
        plt.plot(list(range(0, 28**2)), self.lms_data)
        plt.show()