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

    # Average training data by pixels
    for obj in training_data:
        print(f"{obj.img_lbl} = {obj.pixel_count}")

    # Create the Least Means Squared Graphs for each number from the average matrices
    X = np.zeros([1,28 ** 2])
    for obj in training_data:
        obj.lms_data = obj.img_data.flatten()
        X = np.vstack((X, obj.lms_data)) #this adds the row to it
    X = np.delete(X, 0, 0) #remove the first row because it's 0's
    onesLeft = np.ones([10, 1])
    X = np.hstack((onesLeft, X))
    Y = np.identity(10) #identity matrix of 10 for numbers 0-9 and identity to show the classification
    Bhat, residuals, rank, s = np.linalg.lstsq(np.transpose(X).dot(X), np.transpose(X).dot(Y))
    #Yhat = X.dot(Bhat)
    



    # Go through each number in the testing data, and try them against the training data
    num_to_check = len(mnist_testing_img_data)
    testing_data = []
    misses = []
    hits = []
    missesPixel = []
    hitsPixel = []
    missesLMS = []
    hitsLMS = []

    start = time.time()
    for num in range(num_to_check):
        # Create a MnistNumber using the current testing number
        testing_data.append(MnistNumber(mnist_testing_lbl_data[num]))
        testing_data[num].add_img(mnist_testing_img_data[num])

        # Guess the lowest energy difference of the testing number and the training data
        testing_data[num].guess = 0
        testing_data[num].pixelGuess = 0
        testing_data[num].lmsGuess = 0
        maxY = 0
        lmsGuess = 0
        counter = 0
        for obj in training_data:
            if (obj.get_energy_diff(testing_data[num]) < training_data[testing_data[num].guess].get_energy_diff(testing_data[num])):
                testing_data[num].guess = obj.img_lbl
            if (obj.get_pixel_diff(testing_data[num]) < training_data[testing_data[num].pixelGuess].get_pixel_diff(testing_data[num])):
                testing_data[num].pixelGuess = obj.img_lbl

        testing_data[num].lms_data = testing_data[num].img_data.flatten()
        testRow = testing_data[num].lms_data
        testRow1 = [1]
        testRow = np.hstack((testRow1, testRow))
        yHat = testRow.dot(Bhat)

        for yVal in yHat:
            if (yVal > maxY):
                 maxY = yVal #get the largest value in the yArray so that we know the guess
                 lmsGuess = counter # this is the spot in the array (the actual guess)
            counter = counter + 1
        testing_data[num].lmsGuess = lmsGuess   



        if (testing_data[num].guess != testing_data[num].img_lbl):
            misses.append(testing_data[num])
        else:
            hits.append(testing_data[num])

        if (testing_data[num].pixelGuess != testing_data[num].img_lbl):
            missesPixel.append(testing_data[num])
        else:
            hitsPixel.append(testing_data[num])

        if (testing_data[num].lmsGuess != testing_data[num].img_lbl):
            missesLMS.append(testing_data[num])
        else:
            hitsLMS.append(testing_data[num])
    stop = time.time()

    # Print the Error rate
    print(f"Matrix Average Method")
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


# Do it for pixels
    print(f"\nPixel Average Method")
    print(f"Total checked={num_to_check}; Missed={len(missesPixel)}; Error Rate={round(len(missesPixel)/num_to_check*100, 2)}%; Total time={round(stop-start, 2)}s; Time each check={round((stop-start)/num_to_check*1000,2)}ms")


# Print miss table for pixels
    grouped_missesPixel = [[0 for i in range(10)] for j in range(10)]
    for num in missesPixel:
        grouped_missesPixel[num.img_lbl][num.pixelGuess] += 1
    for num in grouped_missesPixel:
        print(f"{num} : {np.array(num).sum()}")

    grouped_hitsPixel = [0 for i in range(10)]
    for num in hitsPixel:
        grouped_hitsPixel[num.img_lbl] += 1
    for num in range(len(grouped_hitsPixel)):
        print(f"{num}: Hits: {grouped_hitsPixel[num]}\tMisses:{np.array(grouped_missesPixel[num]).sum()}\tTotal: {(grouped_hitsPixel[num]+np.array(grouped_missesPixel[num]).sum())}\tError:{np.round(np.array(grouped_missesPixel[num]).sum()/(grouped_hitsPixel[num]+np.array(grouped_missesPixel[num]).sum()),4)*100}%")



    # Do it for LMS
    print(f"\nLinear Regression Method")
    print(f"Total checked={num_to_check}; Missed={len(missesLMS)}; Error Rate={round(len(missesLMS)/num_to_check*100, 2)}%; Total time={round(stop-start, 2)}s; Time each check={round((stop-start)/num_to_check*1000,2)}ms")


    # Print miss table for LMS
    grouped_missesLMS = [[0 for i in range(10)] for j in range(10)]
    for num in missesLMS:
        grouped_missesLMS[num.img_lbl][num.lmsGuess] += 1
    for num in grouped_missesLMS:
        print(f"{num} : {np.array(num).sum()}")

    grouped_hitsLMS = [0 for i in range(10)]
    for num in hitsLMS:
        grouped_hitsLMS[num.img_lbl] += 1
    for num in range(len(grouped_hitsLMS)):
        print(f"{num}: Hits: {grouped_hitsLMS[num]}\tMisses:{np.array(grouped_missesLMS[num]).sum()}\tTotal: {(grouped_hitsLMS[num]+np.array(grouped_missesLMS[num]).sum())}\tError:{np.round(np.array(grouped_missesLMS[num]).sum()/(grouped_hitsLMS[num]+np.array(grouped_missesLMS[num]).sum()),4)*100}%")



    # Uncomment the following two lines if you want to see what the misses looked like
    # for obj in misses:
    #     obj.plot()

if __name__ == "__main__":
    main_task()