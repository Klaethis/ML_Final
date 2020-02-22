from mlxtend.data import loadlocal_mnist
from MnistNumber import MnistNumber
import time

def main_task():
    # Grab the mnist training data
    mnist_training_img_data, mnist_training_lbl_data = loadlocal_mnist(images_path="./Training_Data/train-images.idx3-ubyte", labels_path="./Training_data/train-labels.idx1-ubyte")
    # Grab the mnist testing data
    mnist_testing_img_data, mnist_testing_lbl_data = loadlocal_mnist(images_path="./Testing_Data/t10k-images.idx3-ubyte", labels_path="./Testing_Data/t10k-labels.idx1-ubyte")

    # Average all the training data by numbers
    training_data = []
    for num in range(10):
        training_data.append(MnistNumber(num))

    for num in range(len(mnist_training_img_data)):
        for obj in training_data:
            if obj.img_lbl == mnist_training_lbl_data[num]:
                obj.add_img(mnist_training_img_data[num])

    # Go through each number in the testing data, and try them against the training data
    testing_data = []
    misses = []
    num_to_check = len(mnist_testing_img_data)
    start = time.time()
    for num in range(num_to_check):
        # Create a MnistNumber using the current testing number
        testing_data.append(MnistNumber(mnist_testing_lbl_data[num]))
        testing_data[num].add_img(mnist_testing_img_data[num])

        # Guess the lowest energy difference of the testing number and the training data
        testing_data[num].guess = 0
        for obj in training_data:
            if (obj.get_energy_diff(testing_data[num]) < training_data[testing_data[num].guess].get_energy_diff(testing_data[num])):
                testing_data[num].guess = obj.img_lbl
        if (testing_data[num].guess != testing_data[num].img_lbl):
            misses.append(testing_data[num])
    stop = time.time()
    # Print the Error rate
    print(f"Total checked={num_to_check}; Missed={len(misses)}; Error Rate={len(misses)/num_to_check*100}%; Time each check={(stop-start)/num_to_check} s")

    # Uncomment the following two lines if you want to see what the misses looked like
    # for obj in misses:
    #     obj.plot()

if __name__ == "__main__":
    main_task()