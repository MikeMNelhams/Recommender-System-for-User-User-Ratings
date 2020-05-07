# ~~~~~~~ Import Statements ~~~~~~~

import numpy as np
import pandas as pd
from itertools import chain  # For quickly unnesting lists
import math
import matplotlib.pyplot as plt
import cProfile  # For profiling

# References:
# https://www.ole.bris.ac.uk/bbcswebdav/pid-4195150-dt-content-rid-14072139_2/courses/EMAT22220_2019_TB-4/brozovsky07recommender.pdf
# https://towardsdatascience.com/intro-to-recommender-system-collaborative-filtering-64a238194a26


# ~~~~~ Functions ~~~~~
def convert_to_dict(ratings_vector):
    # Convert the ratings vector (matrix) Nx2 into a dictionary
    keys = ratings_vector[:, [0]].tolist()  # Convert the first column into the keys
    keys = list(chain(*keys))  # Unnest the lists
    values = ratings_vector[:, [1]].tolist()  # Convert the second column into the values
    values = list(chain(*values))
    dictionary = dict(zip(keys, values))
    return dictionary


def get_indices(data_input):
    # It avoids O(N^2) if we predetermine the indices in one loop
    # A lot slower short term. Extreme amount faster for big numpy arrays
    print('Compiling indexes of unique users ...')
    j = 0
    user = data_input[0][0]
    indices = np.array([0])
    for i in data_input:
        if i[0] != user:
            indices = np.append(indices, j)
            user = i[0]
        j += 1
    print('Indexes compiled')
    return indices


def get_indices_col2(data_input):
    # It avoids O(N^2) if we predetermine the indices in one loop
    # A lot slower short term. Extreme amount faster for big numpy arrays
    print('Compiling indexes of unique ratings ...')
    j = 0
    user = data_input[0][1]
    indices = np.array([0])
    for i in data_input:
        if i[1] != user:
            indices = np.append(indices, j)
            user = i[1]
        j += 1
    print('Indexes compiled')
    return indices


def get_ratings_vector(data_input, user, indices):
    # Input data matrix (Nx3) and the user N, output a Nx2 vector of their ratings
    # The first output column is the indices and the second column is the ratings
    user_max = data_input[-1, :][0]
    data_length = np.size(data_input, 0)

    # Just to verify that the user is contained within the data
    if user > user_max:
        print('User with null ratings found')
        return np.array([[0, 0]])  # Empty array since any user ratings past this will be null

    # If the indices vector is precompiled
    if user == user_max:
        # The exception is the last case
        pointer_start = indices[user_max-1]
        pointer_end = data_length
    else:
        # The users are shifted one from the user indexes
        pointer_start = indices[user-1]
        pointer_end = indices[user]

    # Use np array splicing to quickly get the output rating vector and then remove the first column
    output = data_input[pointer_start:pointer_end, [1, 2]]

    return output


def get_rating_target(data_input, user, target, indices):
    # Find the rating R_i,t the rating user gave about the target t
    R_u = get_ratings_vector(data_input, user, indices)
    output = 0
    if target in R_u[:, [0]]:
        for rating in R_u:
            if rating[0] == target:
                output = rating[1]
                return output

    return output


def get_mean_rating_target3(data_input, target, indices, default_mean):
    # Input data matrix (Nx3) sorted by the 2nd column and the user N, output a Nx2 vector of their ratings
    # The first output column is the indices and the second column is the ratings
    target_max = data_input[-1, :][1]
    data_length = np.size(data_input, 0)

    # Just to verify that the user is contained within the data
    if target > target_max:
        print('Null ratings found')
        return 0  # The user does not have a rating

    # If the indices vector is precompiled
    if target == target_max:
        # The exception is the last case
        pointer_start = indices[target_max]
        pointer_end = data_length
    else:
        # The users are shifted one from the user indexes
        pointer_start = indices[target - 1]
        pointer_end = indices[target]

    total_ratings = data_input[pointer_start:pointer_end, [2]]
    # print('total ratings: ', total_ratings)
    n = len(total_ratings)

    # If the target was never rated, calculate the mean value and use this as the default
    if n == 0:
        # The default rating is either 0, 5 or total mean
        return default_mean

    output = sum(total_ratings) / n
    return output


def get_overlapped_users(data_input, a, j, indices):
    # Find the users which both a and j have rated
    # The overlap will be the users which exist in both ratings vectors
    R_a = get_ratings_vector(data_input, a, indices)
    R_j = get_ratings_vector(data_input, j, indices)
    overlapped_users = []
    for rating in R_a:
        if rating[0] in R_j[:, [0]]:
            overlapped_users.append(rating[0])
    return overlapped_users


def pearson_correlation2(data_input, a, r_a_mean, j, indices1, indices2, default_mean):
    # Return the pearson correlation between the two users, summed over the users that a and j have both rated
    # Formula is online
    # Default mean can be toggled (mean of a user w/o ratings)
    # print('Calculating the pearson correlation')

    overlap = get_overlapped_users(data_input, a, j, indices1)
    r_j_mean = get_mean_rating_target3(data_input, j, indices2, default_mean)

    # print('a: ', a, 'mean a: ', r_a_mean, ' j: ', j,  'mean j: ', r_j_mean)
    # Very unlikely but just a fail-safe
    if not overlap:
        return 0

    numerator_sum = 0
    denominator_sum1 = 0
    denominator_sum2 = 0
    for i in overlap:
        r_a_i = get_rating_target(data_input, a, i, indices1)
        r_j_i = get_rating_target(data_input, j, i, indices1)
        # b1 = r_a_i - r_a_mean
        # b2 = r_j_i - r_j_mean
        numerator_sum += (r_a_i - r_a_mean)*(r_j_i - r_j_mean)
        denominator_sum1 += (r_a_i - r_a_mean)**2
        denominator_sum2 += (r_j_i - r_j_mean)**2
        # print('numerator_sum: ', numerator_sum)

    output = numerator_sum / math.sqrt((denominator_sum1*denominator_sum2))
    # print('output: ', output)
    return output


def euclidean_distance2(v_1, t):
    # Return the euclidean distance between the two vectors, missing overlap only squares the number
    # Input v_1 is an Nx2 rating vector and Input t is an Nx2 rating vector
    # If a rating does not overlap, then do the square of the rating
    # Performs a sparse dictionary comprehension on the dictionary form matrices, rather than for slow loops
    # Significantly faster (400%) than the for loop version

    # First convert the numpy arrays into python dictionary
    d1 = convert_to_dict(v_1)
    d2 = convert_to_dict(t)

    # Perform the dictionary comprehension version of the euclidean distance
    output = sum((d1.get(d, 0) - d2.get(d, 0)) ** 2 for d in set(d1) & set(d2))
    return output


def knn(data_input, user, indices, min0=1, maxN=50):
    # Find the K-Nearest-Neighbours for input user 'user'
    # Minimum of min0 overlaps and a Maximum of maxN neighbours
    # Predefine the array for time complexity O(N) rather than resorting the list for a minimum of O(N^2)
    # Last user is the number of users for which you can do non-trivial euclidean distances for
    print('Compiling the knn euclidean distance ranking, roughly 30 seconds...')
    data_length = data_input[-1][0]
    knn_rank_vector = np.zeros((data_length - 1, 2))  # Predefine the array to save time

    # The entire array is shifted by 1 to avoid missing index error
    for i in range(data_length-1):
        # Get the two ratings vectors for the input user and user i
        r_v_u = get_ratings_vector(data_input, user, indices)
        r_v_i = get_ratings_vector(data_input, i+1, indices)

        # Calculate the euclidean distance between the two users rating vectors
        knn_rank_vector[i] = [i+1, euclidean_distance2(r_v_u, r_v_i)]
        # knn_rank_vector[i] = [i+1, 0]

    # Sort the KNN_rank_vector by the euclidean distances and choose the first max0 values
    knn_rank_vector = knn_rank_vector[knn_rank_vector[:, 1].argsort()]

    # Output the vector of all the KNN, however make sure to remove the repeat value at 0th index
    output = knn_rank_vector[1:maxN+1, 0]
    output = output.astype(int)
    return output


def calculate_predictions3(data_input, user_range, a, min0=1, maxN=50, gamma=2, s=5, gc=0):
    # Correct Algorithm, faster than predic2 by ~50%
    # Precompile the indexes vector of the start of all of the users, in order to speed up calculations
    indexes_vector = get_indices(data)

    # The default rating is neither 0 nor 5 and it is important not to use either
    total_mean = np.mean(data_input[:, [2]])
    print('Total mean: ', total_mean)

    # Precompile a numpy array of the data, sorted by the second column
    data_sorted_by2 = data_input[data_input[:, 1].argsort(kind='mergesort')]  # Stable sorter is faster

    # Precompile the indexes vector by the second column, in order to speed up calculations
    indexes_vector2 = get_indices_col2(data_sorted_by2)

    # Calculate the predicted ratings of a on all the users in the data set
    r_a_mean = get_mean_rating_target3(data_sorted_by2, a, indexes_vector2, total_mean)  # Mean rating of the active user

    # Takes about 30 seconds per KNN calculation, however it is taken OUT of the loop
    knn_users = knn(data_input, a, indexes_vector, min0, maxN)

    # Precompile the ratings vector for a
    r_a = get_ratings_vector(data_input, a, indexes_vector)

    # Calculate the PMCC values and n_mean values outside of the j-loop to increase efficiency
    m1_vector = []
    m2_vector = []
    n_mean_vector = []
    for n in knn_users:
        # PMCC value ranges between -1 and 1
        m1 = pearson_correlation2(data_input, a, r_a_mean, n, indexes_vector, indexes_vector2, total_mean)
        m1_vector.append(m1)
        n_mean_vector.append(get_mean_rating_target3(data_sorted_by2, n, indexes_vector2, total_mean))

    # k Is the number of K-Nearest Neighbours
    k = np.size(knn_users, 0)

    # Normalizing factor to take the mean of the values, otherwise -k*10 < 10*(k+1)
    normalizing_factor = 1 / k

    # Loop through all of the users
    predictions = []
    predictions2 = []
    num_ratings = []
    increases = []
    for j in user_range:

        counter = 0
        beta_total = 0
        for n in knn_users:
            m1 = m1_vector[counter]
            # KNN value ranges between -10 and 10
            r_n_j = get_rating_target(data_input, n, j, indexes_vector)
            m2 = r_n_j - n_mean_vector[counter]

            # m1*m2 ranges between -10 and 10
            beta_total += m1*m2

            # print('m1: ', m1, ' m2: ', m2)
            counter += 1  # Increase the m1_vector counter

        # Calculate the bias for j, formula in the report
        r_j = get_ratings_vector(data_input, j, indexes_vector)
        b = np.size(r_j) / 2
        num_ratings.append(b)  # Add the number of ratings for plotting increase vs number of ratings
        b = total_mean / (1 + b)

        # Prediction value 1, for a rating j. Ranges from -10 < p <= 21 (NOT NORMALIZED)
        # With Bias
        p_a_j = r_a_mean + normalizing_factor*beta_total + b
        p_a_j = p_a_j[0]
        # Without Bias
        p2_a_j = r_a_mean + normalizing_factor * beta_total
        p2_a_j = p2_a_j[0]

        # Prediction value 2, for a rating j. Ranges from 0 < p < 10 (NORMALIZED) 1 < G < 3 optimal
        p_a_j2 = 10 / (1 + math.exp(- (p_a_j / gamma)))
        p2_a_j2 = 10 / (1 + math.exp(- (p2_a_j / gamma)))

        printable = 'Target user: %s,    p: %s,    \u03C3(p): %s' % (j, round(p_a_j, 4), round(p_a_j2, 4))
        print(printable)
        predictions.append(p_a_j2)
        predictions2.append(p2_a_j2)
        increases.append(p_a_j2 - p2_a_j2)

    col_1 = np.array(user_range)
    col_2 = np.array(predictions)
    col_1_2 = col_1
    col_2_2 = np.array(predictions2)

    #  ~~~~ Attach the users to their predictions along the columns
    p_values = np.column_stack((col_1, col_2))  # With Bias
    p_values2 = np.column_stack((col_1_2, col_2_2))  # Without Bias

    #  ~~~~ Sort the list then reverse the list (biggest at 0)
    # With Bias
    p_values = p_values[p_values[:, 1].argsort(kind='mergesort')]
    p_values = np.flipud(p_values)

    # Without Bias
    p_values2 = p_values2[p_values2[:, 1].argsort(kind='mergesort')]
    p_values2 = np.flipud(p_values2)

    # ~~~~ Remove all the values from the predictions which are a itself or that a has already rated or a null value (0)
    # Copy the p_values array, then add only the desired values to the temp, then copy the temp array
    # (avoids dynamic 'for loop' sizing errors)

    # With Bias
    p_values_temp1 = []  # First column
    p_values_temp2 = []  # Second column
    i = 0
    for p in p_values:
        if (p[1] > gc) and (int(p[0]) != a) and (p[0] not in r_a[:, [0]]):
            p_values_temp1.append(p[0])
            p_values_temp2.append(p[1])
        i += 1

    # Without Bias
    p_values_temp3 = []  # First column (2)
    p_values_temp4 = []  # Second Column (2)
    i = 0
    for p in p_values2:
        if (p[1] > gc) and (int(p[0]) != a) and (p[0] not in r_a[:, [0]]):
            p_values_temp3.append(p[0])
            p_values_temp4.append(p[1])
        i += 1

    p_values_temp1 = np.array(p_values_temp1)
    p_values_temp2 = np.array(p_values_temp2)
    p_values_temp3 = np.array(p_values_temp3)
    p_values_temp4 = np.array(p_values_temp4)

    p_values_temp = np.column_stack((p_values_temp1, p_values_temp2))
    p_values = p_values_temp

    p_values_temp = np.column_stack((p_values_temp3, p_values_temp4))
    p_values2 = p_values_temp

    print(' ')

    # Print the top S predictions (With Bias)
    print(' ~~~~~~ WITH BIAS ~~~~~~')
    print('Top {} predictions'.format(s))
    if np.size(p_values) == 0:
        print('No Bias Predictions')
    else:
        for i in range(s):
            print('User: {}. Rating: {}.'.format(int(p_values[i][0]), round(p_values[i][1], 4)))

    print(' ')

    # Print the top S predictions (Without Bias)
    print(' ~~~~~~ WITHOUT BIAS ~~~~~~')
    print('Top {} predictions'.format(s))
    if np.size(p_values2) == 0:
        print('No non-bias Predictions')
    else:
        for i in range(s):
            print('User: {}. Rating: {}.'.format(int(p_values2[i][0]), round(p_values2[i][1], 4)))

    # ~~~~~~ Plot the data Functions~~~~~~
    # plot_predictions(a, p_values, p_values2)
    # plot_boxplot(a, p_values, p_values2)
    # plot_increase(a, increases, num_ratings)

    return p_values


def plot_predictions(a, predictions, predictions2):
    # Plot a graph of the predictions versus the number of ratings each user has
    # Predictions is WITH bias
    # Predictions2 is WITHOUT bias
    # VOID function
    x = predictions[:, [0]]
    y = predictions[:, [1]]

    x2 = predictions2[:, [0]]
    y2 = predictions2[:, [1]]

    # The first scatter plot without biases
    p1 = plt.scatter(x, y, marker="x", c='blue')
    # p2 = plt.scatter(x2, y2, marker="x")

    # Title the plot and axes automatically based on the active user:
    title = "Predictions with biases for active user {} ordered by target users".format(a)
    plt.title(title)
    plt.ylabel('Prediction \u03C3($P_{a,j}$)')
    plt.xlabel('Target user j')
    # plt.legend(['With Bias', 'Without Bias'])

    plt.show()
    return 0


def plot_boxplot(a, predictions, predictions2):
    # Plot a boxplot of the predictions
    # VOID function
    y1 = predictions[:, [1]]
    y2 = predictions2[:, [1]]

    fig1, ax1 = plt.subplots()

    plots = [y1, y2]
    ax1.boxplot(plots, whis=[0, 100], labels=['With bias', 'Without Bias'], vert=False)

    # Graph formatting
    title = 'Comparing boxplots of the predictions for active user {}'.format(a)
    ax1.set_title(title)
    ax1.set_xlabel('Prediction \u03C3($P_{a,j}$)')

    plt.show()
    return 0


def plot_increase(a, increasesv, num_ratingsv):
    # Plot the increase against the number of ratings
    # VOID function
    plt.scatter(num_ratingsv, increasesv)
    plt.xlabel('Number of ratings of each target user j')
    plt.ylabel('Increase in prediction \u0394\u03C3($P_{a,j}$)')
    plt.title('Plotting the increase in prediction against the number of ratings for active user {}'.format(a))
    plt.show()
    return 0

# ~~~~~ Main ~~~~~
# The aim, is for any input active user A
# Active user A is looking for a recommended users, and users n_0, n_1, n_2 all like similar people to A.
# This step is done by KNN with a similarity measure of Pearson's Correlation or Cosine Correlation.
# The neighbours are the K Nearest Neighbours
# minO – minimum number of common ratings between users necessary to calculate user-user similarity
# maxN – maximum number of user neighbors to be used during the computation
# User j is some input user which the system will predict the rating for


# ~~ Globals ~~
# Users that have not rated are classed as 'non-comparisons' right now
gammaG = 2  # The normalizing factor used in the sigmoid normalizing function
aG = 1  # Active user we a recommending users to
min0G = 1  # Minimum number of common ratings between users necessary to calculate user-user similarity (Unused)
maxNG = 40  # Maximum number of user neighbours to be used during the computation
sG = 10  # The top s predictions will be printed
GCG = 0  # The graph cutoff, won't show values less than GCG

# ~~ Main Code ~~
# Load the ratings data
print('Loading Ratings ...')
dataF = pd.read_csv('ratings.csv')
print('Ratings Loaded')

data = dataF.to_numpy()  # Convert from a DataFrame to a numpy array
users = np.unique(data[:, 0])  # Trim any duplicates in the array
num_users = users[-1]  # The number of users will be the last user that rated people

print('Total number of users: ', len(users))

# The predictions data set
user_rangeG = [x for x in range(1, num_users + 1) if x != aG]

# Calculate and print the top # 'sG' predictions
# P = calculate_predictions3(data, user_rangeG, aG, min0G, maxNG, gammaG, sG, GCG)  # Profiling
cProfile.run('calculate_predictions3(data, user_rangeG, aG, min0G, maxNG, gammaG, sG, GCG)')
