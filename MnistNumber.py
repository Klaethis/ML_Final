from matplotlib import pyplot as plt
import numpy as np

class MnistNumber:
    img_data = np.zeros([28,28])
    img_lbl = 0
    num_img = 0
    guess = -1
    
    # Initializes using the given number and image array
    def __init__(self, number, img = None):
        self.img_lbl = number
        if img != None:
            self.img_data = np.reshape(img, (28,28))
            self.num_img = 1
        else:
            self.img_data = np.zeros([28,28])
            self.num_img = 0

    # Average the given image with the current average
    def add_img(self, new_data):
        self.num_img += 1
        self.img_data += (np.reshape(new_data, (28,28)) - self.img_data) / self.num_img

    # Get the difference in energy of this number with the given number
    def get_energy_diff(self, com_num):
        return ((com_num.img_data - self.img_data)**2).sum()

    # Check to see if this number is the same as the given number
    def check_num(self, num):
        return (self.img_lbl == num)

    # Show a plot of this number
    def plot(self):
        if self.guess == -1:
            plt.title(f"Image of {self.img_lbl}")
        else:
            plt.title(f"Image of {self.img_lbl}; Guess {self.guess}")
        plt.imshow(self.img_data)
        plt.show()
        plt.cla()
        plt.clf()