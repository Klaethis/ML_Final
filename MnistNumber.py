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
        self.img_data = np.zeros([28,28,5,5])
        if img != None:
            self.add_img(img)
        else:
            self.num_img = 0

    # Average the given image with the current average
    def add_img(self, new_data):
        new_img = np.reshape(new_data, (28,28))
        self.num_img += 1
        for col in range(28):
            for row in range(28):
                img_slice = np.zeros([5,5])

                start_col = col - 2 if col - 2 >= 0 else 0 if col - 1 < 0 else col - 1
                end_col = col + 2 if col + 2 <= len(new_img) else len(new_img) if col + 1 < len(new_img) else col + 1
                start_row = row - 2 if row - 2 >= 0 else 0 if row - 1 < 0 else row - 1
                end_row = row + 2 if row + 2 <= len(new_img) else len(new_img) if row + 1 < len(new_img) else row + 1

                img_slice[len(img_slice) - (end_col - start_col):, len(img_slice) - (end_row - start_row):] = new_img[start_col:end_col, start_row:end_row]
                self.img_data[col, row] += (img_slice - self.img_data[col, row]) / self.num_img

    # Get the difference in energy of this number with the given number
    def get_energy_diff(self, com_num):
        energy_matrix = np.zeros([28,28])
        for col in range(28):
            for row in range(28):
                energy_matrix[col, row] = np.linalg.norm((com_num.img_data[col, row] - self.img_data[col, row]))
        return np.linalg.norm(energy_matrix)

    # Check to see if this number is the same as the given number
    def check_num(self, num):
        return (self.img_lbl == num)

    # Show a plot of this number
    def plot(self):
        if self.guess == -1:
            plt.title(f"Image of {self.img_lbl}")
        else:
            plt.title(f"Image of {self.img_lbl}; Guess {self.guess}")
        num_image = np.zeros([28,28])
        for col in range(28):
            for row in range(28):
                num_image[col, row] = np.average(self.img_data[col, row])
        plt.imshow(num_image)
        plt.show()
        plt.cla()
        plt.clf()