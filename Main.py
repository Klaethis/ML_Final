from mlxtend.data import loadlocal_mnist
from MnistNumber import MnistNumber
import numpy as np
import time

MAX_READ_NUMBERS = 500

def main_task():
    # Grab the mnist training data
    mnist_training_img_data, mnist_training_lbl_data = loadlocal_mnist(images_path="./Training_Data/train-images.idx3-ubyte", labels_path="./Training_data/train-labels.idx1-ubyte")
    # Grab the mnist testing data
    mnist_testing_img_data, mnist_testing_lbl_data = loadlocal_mnist(images_path="./Testing_Data/t10k-images.idx3-ubyte", labels_path="./Testing_Data/t10k-labels.idx1-ubyte")

    # Average all the training data by numbers
    training_data = [MnistNumber(i) for i in range(10)]

    MAX_READ_NUMBERS = len(mnist_training_img_data)

    for obj in training_data:
        for num in range(MAX_READ_NUMBERS):
            if obj.img_lbl == mnist_training_lbl_data[num]:
                obj.add_img(mnist_training_img_data[num])

    # Go through each number in the testing data, and try them against the training data
    num_to_check = len(mnist_testing_img_data)
    testing_data = []
    misses = []
    hits = []

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
        else:
            hits.append(testing_data[num])
    stop = time.time()

    # Print the Error rate
    print(f"Total checked={num_to_check}; Missed={len(misses)}; Error Rate={round(len(misses)/num_to_check*100, 2)}%; Total time={round(stop-start, 2)}s; Time each check={round((stop-start)/num_to_check*1000,2)}ms")
    
    # Print miss table
    grouped_misses = [[0 for i in range(10)] for j in range(10)]
    for num in misses:
        grouped_misses[num.img_lbl][num.guess] += 1
    for num in grouped_misses:
        print(f"{num} : {np.array(num).sum()}")

    grouped_hits = [0 for i in range(10)]
    for num in hits:
        grouped_hits[num.img_lbl] += 1
    for num in range(len(grouped_hits)):
        print(f"{num}: Hits: {grouped_hits[num]}\tMisses:{np.array(grouped_misses[num]).sum()}\tTotal: {(grouped_hits[num]+np.array(grouped_misses[num]).sum())}\tError:{np.round(np.array(grouped_misses[num]).sum()/(grouped_hits[num]+np.array(grouped_misses[num]).sum()),4)*100}%")

    # Uncomment the following two lines if you want to see what the misses looked like
    # for obj in misses:
    #     obj.plot()

if __name__ == "__main__":
    main_task()