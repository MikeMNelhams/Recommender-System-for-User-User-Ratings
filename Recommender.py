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
        pointer_start = indices[user_max]
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


def get_mean_rating_target(data_input, target, default_mean):
    # Return the mean rating for the target in the data Nx3 matrix
    total_ratings = []
    for rating in data_input:
        if rating[1] == target:
            total_ratings.append(rating[2])
    n = len(total_ratings)

    # If the target was never rated, calculate the mean value and use this as the default
    if np.size(total_ratings) == 0:
        # The default rating is neither 0 nor 5 and it is important not to use either
        return default_mean

    output = sum(total_ratings)/n
    return output


def get_mean_rating_target2(data_input, target, default_mean):
    # Return the mean rating for the target in the data Nx3 matrix
    total_ratings = [rating[2] for rating in data_input if rating[1] == target]
    n = len(total_ratings)

    # If the target was never rated, calculate the mean value and use this as the default
    if np.size(total_ratings) == 0:
        # The default rating is either 0, 5 or total mean
        return default_mean

    output = sum(total_ratings) / n
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


def pearson_correlation(data_input, a, r_a_mean, j, indices, default_mean):
    # Return the pearson correlation between the two users, summed over the users that a and j have both rated
    # Formula is online
    # Default mean can be toggled (mean of a user w/o ratings)
    # print('Calculating the pearson correlation')

    overlap = get_overlapped_users(data_input, a, j, indices)
    r_j_mean = get_mean_rating_target3(data_input, j, indices, default_mean)

    # print('a: ', a, 'mean a: ', r_a_mean, ' j: ', j,  'mean j: ', r_j_mean)
    # Very unlikely but just a fail-safe
    if not overlap:
        return 0

    numerator_sum = 0
    denominator_sum1 = 0
    denominator_sum2 = 0
    for i in overlap:
        r_a_i = get_rating_target(data_input, a, i, indices)
        r_j_i = get_rating_target(data_input, j, i, indices)
        # b1 = r_a_i - r_a_mean
        # b2 = r_j_i - r_j_mean
        numerator_sum += (r_a_i - r_a_mean)*(r_j_i - r_j_mean)
        denominator_sum1 += (r_a_i - r_a_mean)**2
        denominator_sum2 += (r_j_i - r_j_mean)**2
        # print('numerator_sum: ', numerator_sum)

    output = numerator_sum / math.sqrt((denominator_sum1*denominator_sum2))
    # print('output: ', output)
    return output


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


def euclidean_distance(v_1, t):
    # Return the euclidean distance between the two vectors, missing overlap only squares the number
    # Input v_1 is an Nx2 rating vector and Input t is an Nx2 rating vector
    # If a rating does not overlap, then do the square of the rating
    v_1_length = np.size(v_1, 0)  # Number of ratings in vector v_1
    t_length = np.size(t, 0)  # Number of ratings in vector t

    # Sum the euclidean distances between the two vectors
    distance = 0

    # First sum through v_1 / t and v_1 intersect t
    for i in range(v_1_length):
        if v_1[i][0] in t[:, [1]]:
            # distance += (v_1[i][1] - t[np.where(t[:, [0]] == v_1[i][0])][1])**2
            distance += (v_1[i][1] - t[where(t, v_1[i][0])][1])**2
        else:
            distance += v_1[i][1]**2

    # Sum through t / v_1
    for i in range(t_length):
        if t[i][0] not in v_1:
            distance += t[i][1]**2

    return distance


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


def filter_overlap(data_input, user, min, indices):
    # Input a data set, minimum overlaps and the user we can comparing overlaps
    # Perform with dictionary operations
    # Not an efficient algorithm yet
    print('Compiling overlap ...')

    # Convert user to a dictionary
    user_d = convert_to_dict(get_ratings_vector(data, user, indices))
    user_d_keys = set(user_d.keys())

    # Allocate the memory for the filtered data list
    output = []

    # Loop through the data, convert them to dictionaries, perform the intersect and see if this exceeds the length
    for i in data_input:
        i_r = get_ratings_vector(data, i[0], indices)
        i_r_d = convert_to_dict(i_r)

        # Calculate the intersect of the two dictionaries
        i_r_d_keys = set(i_r_d.keys())
        intersection = user_d_keys & i_r_d_keys
        if len(intersection) > min:
            output.append(i)


def where(vector, query):
    # Return the first index where the vector element is equal to the query
    # Sadly this function is FAR faster than the numpy np.where() in-built function
    j = 0
    for i in vector:
        if i == query:
            return j
    j += 1
    return 0


def knn(data_input, user, indices, min0=1, maxN=50):
    # Find the K-Nearest-Neighbours for input user 'user'
    # Minimum of min0 overlaps and a Maximum of maxN neighbours
    # Predefine the array for time complexity O(N) rather than resorting the list for a minimum of O(N^2)
    # Last user is the number of users for which you can do non-trivial euclidean distances for
    print('Compiling the knn euclidean distance ranking, roughly 30 seconds ...')
    print('KNN user: ', user)
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


def calculate_predictions(data_input, user_range, a, min0, maxN, gamma):
    # Correct Algorithm
    # Precompile the indexes vector of the start of all of the users, in order to speed up calculations
    indexes_vector = get_indices(data)

    # The default rating is neither 0 nor 5 and it is important not to use either
    total_mean = np.mean(data_input[:, [2]])
    print('Total mean: ', total_mean)

    # Calculate the predicted ratings of a on all the users in the data set
    r_a_mean = get_mean_rating_target(data, a, total_mean)  # Mean rating of the active user

    # Takes about 30 seconds per KNN calculation, however it is taken OUT of the loop
    knn_users = knn(data_input, a, indexes_vector, min0, maxN)

    # Loop through all of the users
    predictions = []
    for j in user_range:

        beta_total = 0
        for n in knn_users:
            # PMCC value ranges between -1 and 1
            m1 = pearson_correlation(data_input, a, r_a_mean, n, indexes_vector, total_mean)

            # KNN value ranges between -10 and 10
            r_n_j = get_rating_target(data_input, n, j, indexes_vector)
            r_n_mean = get_mean_rating_target(data_input, n, total_mean)
            m2 = r_n_j - r_n_mean

            # m1*m2 ranges between -10 and 10
            beta_total += m1*m2

            print('m1: ', m1, ' m2: ', m2)

        # k Is the number of K-Nearest Neighbours
        k = np.size(knn_users, 0)

        # Normalizing factor to take the mean of the values, otherwise -k*10 < 10*(k+1)
        normalizing_factor = 1 / k

        # Prediction value 1, for a rating j. Ranges from -10 < p < 20 (NOT NORMALIZED)
        p_a_j = r_a_mean + normalizing_factor*beta_total

        # Prediction value 2, for a rating j. Ranges from 0 < p < 10 (NORMALIZED) 1 < G < 3 optimal
        p_a_j2 = 10 / (1 + np.exp(- (p_a_j / gamma)))
        # p_a_j3 = 10 / (1 + math.exp(- (p_a_j / gamma)))
        print('p before: ', p_a_j, ' p after: ', p_a_j2)
        predictions.append(p_a_j2)

    return predictions


def calculate_predictions2(data_input, user_range, a, min0, maxN, gamma):
    # Correct Algorithm, faster than predic2 by ~50%
    # Precompile the indexes vector of the start of all of the users, in order to speed up calculations
    indexes_vector = get_indices(data)

    # The default rating is neither 0 nor 5 and it is important not to use either
    total_mean = np.mean(data_input[:, [2]])
    print('Total mean: ', total_mean)

    # Calculate the predicted ratings of a on all the users in the data set
    r_a_mean = get_mean_rating_target2(data, a, total_mean)  # Mean rating of the active user

    # Takes about 30 seconds per KNN calculation, however it is taken OUT of the loop
    knn_users = knn(data_input, a, indexes_vector, min0, maxN)

    # Calculate the PMCC values and n_mean values outside of the j-loop to increase efficiency
    m1_vector = []
    m2_vector = []
    n_mean_vector = []
    for n in knn_users:
        # PMCC value ranges between -1 and 1
        m1 = pearson_correlation(data_input, a, r_a_mean, n, indexes_vector, total_mean)
        m1_vector.append(m1)
        n_mean_vector.append(get_mean_rating_target2(data_input, n, total_mean))

    # k Is the number of K-Nearest Neighbours
    k = np.size(knn_users, 0)

    # Normalizing factor to take the mean of the values, otherwise -k*10 < 10*(k+1)
    normalizing_factor = 1 / k

    # Loop through all of the users
    predictions = []
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

        # Prediction value 1, for a rating j. Ranges from -10 < p < 20 (NOT NORMALIZED)
        p_a_j = r_a_mean + normalizing_factor*beta_total

        # Prediction value 2, for a rating j. Ranges from 0 < p < 10 (NORMALIZED) 1 < G < 3 optimal
        p_a_j2 = 10 / (1 + np.exp(- (p_a_j / gamma)))
        print('p before: ', p_a_j, ' p after: ', p_a_j2)
        predictions.append(p_a_j2)

    return predictions


def calculate_predictions3(data_input, user_range, a, min0=1, maxN=50, gamma=2, s=5):
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

        # Prediction value 1, for a rating j. Ranges from -10 < p < 20 (NOT NORMALIZED)
        p_a_j = r_a_mean + normalizing_factor*beta_total
        p_a_j = p_a_j[0]

        # Prediction value 2, for a rating j. Ranges from 0 < p < 10 (NORMALIZED) 1 < G < 3 optimal
        p_a_j2 = 10 / (1 + math.exp(- (p_a_j / gamma)))
        printable = 'p before: %s    p after: %s' % (p_a_j, p_a_j2)  # Faster than string concatenation with +
        # print(printable)
        predictions.append(p_a_j2)

    col_1 = np.array(user_range)
    col_2 = np.array(predictions)

    # print(col_1)
    # print(col_2)

    # Attach the users to their predictions along the columns
    p_values = np.column_stack((col_1, col_2))

    # Sort the list then reverse the list (biggest at 0)
    p_values = p_values[p_values[:, 1].argsort(kind='mergesort')]
    p_values = np.flipud(p_values)

    # Remove all the values from the predictions which are a itself or that a has already rated or a null value (0)
    p_values_temp = np.zeros(np.shape(p_values))  # Predefine the array to be overwritten to save time
    i = 0
    for p in p_values:
        if (int(p[0]) != a) and (p[0] not in r_a[:, [0]]) and p[0] != 0:
            p_values_temp[i] = p
        i += 1

    p_values = p_values_temp  # Change to the temporary array

    # Print the top S predictions
    for i in range(s):
        print(p_values[i][0])
        s += 1

    return p_values


def plot_predictions(data_input, user_range, a, predictions):
    # Plot a graph of the predictions versus the number of ratings each user has
    # VOID function
    x = predictions[:, [0]]
    y = predictions[:, [1]]
    print('x: ', x)
    print('y: ', y)
    plt.scatter(x, y)
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
# users that have not rated are classed as 'non-comparisons' right now
gammaG = 2  # The normalizing factor puts the similarity into proportion with the mean rating
aG = 1  # Active user we a recommending users to
min0G = 1  # Minimum number of common ratings between users necessary to calculate user-user similarity
maxNG = 5  # Maximum number of user neighbours to be used during the computation
sG = 5  # The top s predictions will be printed

# ~~ Main Code ~~
# Load the ratings data
print('Loading Ratings ...')
dataF = pd.read_csv('ratings.csv')
print('Ratings Loaded')

data = dataF.to_numpy()  # Convert from a DataFrame to a numpy array
users = np.unique(data[:, 0])  # Trim any duplicates in the array
num_users = users[-1]  # The number of users will be the last user that rated people

print('Number of users: ', len(users))

# print(filter_overlap(data, a, 1, indexes_vector))
# cProfile.run('knn(data, a, indexes_vector, 0, 5)')

# knn_vector = knn(data, 3, indexes_vector, min0G, maxNG)
# print(knn_vector)

# The predictions dataset
user_rangeG = [x for x in range(1, num_users + 1) if x != aG]
# print('user range: ', user_rangeG)

# cProfile.run('calculate_predictions3(data, user_rangeG, a, min0G, maxNG, gamma)')
# Calculate and print the top # 'sG' predictions
P = calculate_predictions3(data, user_rangeG, aG, min0G, maxNG, gammaG, sG)

# plot_predictions(data, user_rangeG, aG, P)


