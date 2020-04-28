from matplotlib import pyplot as plt
import pathlib
import numpy as np
import math

class MnistNumber:
    img_data = np.zeros([28,28])
    pixel_count = 0
    img_lbl = 0
    num_img = 0
    guess = -1
    pixelGuess = -1
    lmsGuess = -1
    voterGuess = -1
    threshhold = 80
    lms_data = np.zeros([0,28**2])
    
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
        if (self.pixel_count == 0):
            self.pixel_count = (np.reshape(new_data, (28,28)) > self.threshhold).sum()
        else:
            self.pixel_count = (self.pixel_count +  (np.reshape(new_data, (28,28)) > self.threshhold).sum()) / 2
            

    # Get the difference in energy of this number with the given number
    def get_energy_diff(self, com_num):
        return np.linalg.norm(com_num.img_data - self.img_data)

    def get_pixel_diff(self, com_num):
        return (((com_num.pixel_count - self.pixel_count)**2)**(1/2))

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

    #def getOutputY(self):

