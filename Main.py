from mlxtend.data import loadlocal_mnist
from MnistNumber import MnistNumber
import numpy as np
import time

MAX_READ_NUMBERS = 500

def main_task():
    # Grab the mnist training data
    mnist_training_img_data, mnist_training_lbl_data = loadlocal_mnist(images_path="./Training_Data/train-images.idx3-ubyte", labels_path="./Training_Data/train-labels.idx1-ubyte")
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
    Bhat, residuals, rank, s = np.linalg.lstsq(np.transpose(X).dot(X), np.transpose(X).dot(Y), rcond=None)
    
    # Error tables
    MA_error = 1 - np.array([0.1041,0.0379,0.2432,0.1941,0.1741,0.3139,0.1367,0.1673,0.2628,0.1933])
    PA_error = 1 - np.array([0.3561,0.4952,0.8924,0.8515,0.9104,0.8722,0.8904,0.9056,0.9928,0.9584])
    LMS_error = 1 - np.array([0.2898,0.1137,0.3537,0.2218,0.2525,0.3475,0.2276,0.2714,0.2916,0.2270])
    H_Trans_error = 1 - np.array([0.0673,0.9982,0.9738,0.9257,1.0000,1.0000,0.9937,1.0000,0.9220,1.0000])
    V_Trans_error = 1 - np.array([0.8939,0.1656,0.8324,0.2683,0.7159,1.0000,1.0000,0.8696,1.0000,1.0000])

    # Go through each number in the testing data, and try them against the training data
    num_to_check = len(mnist_testing_img_data)
    testing_data = []
    misses = []
    hits = []
    missesPixel = []
    hitsPixel = []
    missesLMS = []
    hitsLMS = []
    missesHTransition = []
    hitsHTransition = []
    missesVTransition = []
    hitsVTransition = []
    missesVoter = []
    hitsVoter = []

    start = time.time()
    for num in range(num_to_check):
        # Create a MnistNumber using the current testing number
        testing_data.append(MnistNumber(mnist_testing_lbl_data[num]))
        testing_data[num].add_img(mnist_testing_img_data[num])

        # Guess the lowest energy difference of the testing number and the training data
        testing_data[num].guess = 0
        testing_data[num].pixelGuess = 0
        testing_data[num].lmsGuess = 0
        testing_data[num].hTransitionGuess = 0
        testing_data[num].vTransitionGuess = 0
        testing_data[num].voterGuess = 0
        
        for obj in training_data:
            if (obj.get_energy_diff(testing_data[num]) < training_data[testing_data[num].guess].get_energy_diff(testing_data[num])):
                testing_data[num].guess = obj.img_lbl
            if (obj.get_pixel_diff(testing_data[num]) < training_data[testing_data[num].pixelGuess].get_pixel_diff(testing_data[num])):
                testing_data[num].pixelGuess = obj.img_lbl
            if (obj.get_h_transition_diff(testing_data[num]) < training_data[testing_data[num].hTransitionGuess].get_h_transition_diff(testing_data[num])):
                testing_data[num].hTransitionGuess = obj.img_lbl
            if (obj.get_v_transition_diff(testing_data[num]) < training_data[testing_data[num].vTransitionGuess].get_v_transition_diff(testing_data[num])):
                testing_data[num].vTransitionGuess = obj.img_lbl

        testRow = np.hstack(([1], testing_data[num].img_data.flatten()))
        yHat = testRow.dot(Bhat)

        testing_data[num].lmsGuess = np.argmax(yHat) 

        last_confidence = 0
        testing_data[num].voterGuess = 0
        for count_num in range(10):
            cur_confidence = 0
            if (testing_data[num].guess == count_num):
                cur_confidence += MA_error[testing_data[num].guess]
            if (testing_data[num].pixelGuess == count_num):
                cur_confidence += PA_error[testing_data[num].pixelGuess]
            if (testing_data[num].lmsGuess == count_num):
                cur_confidence += LMS_error[testing_data[num].lmsGuess]
            if (testing_data[num].hTransitionGuess == count_num):
                cur_confidence += H_Trans_error[testing_data[num].hTransitionGuess]
            if (testing_data[num].vTransitionGuess == count_num):
                cur_confidence += V_Trans_error[testing_data[num].vTransitionGuess]
            if (cur_confidence > last_confidence):
                testing_data[num].voterGuess = count_num
                last_confidence = cur_confidence

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

        if (testing_data[num].hTransitionGuess != testing_data[num].img_lbl):
            missesHTransition.append(testing_data[num])
        else:
            hitsHTransition.append(testing_data[num])

        if (testing_data[num].vTransitionGuess != testing_data[num].img_lbl):
            missesVTransition.append(testing_data[num])
        else:
            hitsVTransition.append(testing_data[num])

        if (testing_data[num].voterGuess != testing_data[num].img_lbl):
            missesVoter.append(testing_data[num])
        else:
            hitsVoter.append(testing_data[num])
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


    # Do it for LMS
    print(f"\nHorizontal Transition Method")
    print(f"Total checked={num_to_check}; Missed={len(missesHTransition)}; Error Rate={round(len(missesHTransition)/num_to_check*100, 2)}%; Total time={round(stop-start, 2)}s; Time each check={round((stop-start)/num_to_check*1000,2)}ms")


    # Print miss table for horizontal Transition
    grouped_missesHTransition = [[0 for i in range(10)] for j in range(10)]
    for num in missesHTransition:
        grouped_missesHTransition[num.img_lbl][num.hTransitionGuess] += 1
    for num in grouped_missesHTransition:
        print(f"{num} : {np.array(num).sum()}")

    grouped_hitsHTransition = [0 for i in range(10)]
    for num in hitsHTransition:
        grouped_hitsHTransition[num.img_lbl] += 1
    for num in range(len(grouped_hitsHTransition)):
        print(f"{num}: Hits: {grouped_hitsHTransition[num]}\tMisses:{np.array(grouped_missesHTransition[num]).sum()}\tTotal: {(grouped_hitsHTransition[num]+np.array(grouped_missesHTransition[num]).sum())}\tError:{np.round(np.array(grouped_missesHTransition[num]).sum()/(grouped_hitsHTransition[num]+np.array(grouped_missesHTransition[num]).sum()),4)*100}%")


    # Do it for LMS
    print(f"\nVertical Transition Method")
    print(f"Total checked={num_to_check}; Missed={len(missesVTransition)}; Error Rate={round(len(missesVTransition)/num_to_check*100, 2)}%; Total time={round(stop-start, 2)}s; Time each check={round((stop-start)/num_to_check*1000,2)}ms")

    # Print miss table for Vertical Transition
    grouped_missesVSTransition = [[0 for i in range(10)] for j in range(10)]
    for num in missesVTransition:
        grouped_missesVSTransition[num.img_lbl][num.vTransitionGuess] += 1
    for num in grouped_missesVSTransition:
        print(f"{num} : {np.array(num).sum()}")

    grouped_hitsVSTransition = [0 for i in range(10)]
    for num in hitsVTransition:
        grouped_hitsVSTransition[num.img_lbl] += 1
    for num in range(len(grouped_hitsVSTransition)):
        print(f"{num}: Hits: {grouped_hitsVSTransition[num]}\tMisses:{np.array(grouped_missesVSTransition[num]).sum()}\tTotal: {(grouped_hitsVSTransition[num]+np.array(grouped_missesVSTransition[num]).sum())}\tError:{np.round(np.array(grouped_missesVSTransition[num]).sum()/(grouped_hitsVSTransition[num]+np.array(grouped_missesVSTransition[num]).sum()),4)*100}%")


    # Do it for horizontal Transition
    print(f"\nVoter Method")
    print(f"Total checked={num_to_check}; Missed={len(missesVoter)}; Error Rate={round(len(missesVoter)/num_to_check*100, 2)}%; Total time={round(stop-start, 2)}s; Time each check={round((stop-start)/num_to_check*1000,2)}ms")


    # Print miss table for Voter
    grouped_missesVoter = [[0 for i in range(10)] for j in range(10)]
    for num in missesVoter:
        grouped_missesVoter[num.img_lbl][num.voterGuess] += 1
    for num in grouped_missesVoter:
        print(f"{num} : {np.array(num).sum()}")

    grouped_hitsVoter = [0 for i in range(10)]
    for num in hitsVoter:
        grouped_hitsVoter[num.img_lbl] += 1
    for num in range(len(grouped_hitsVoter)):
        print(f"{num}: Hits: {grouped_hitsVoter[num]}\tMisses:{np.array(grouped_missesVoter[num]).sum()}\tTotal: {(grouped_hitsVoter[num]+np.array(grouped_missesVoter[num]).sum())}\tError:{np.round(np.array(grouped_missesVoter[num]).sum()/(grouped_hitsVoter[num]+np.array(grouped_missesVoter[num]).sum()),4)*100}%")

    # Uncomment the following two lines if you want to see what the misses looked like
    # for obj in misses:
    #     obj.plot()

if __name__ == "__main__":
    main_task()