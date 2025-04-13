#import necessary libraries
import pandas as pd
import random
import pickle

#function to generate 10 lists for creating spatial folds, 5 with 
#one observation between 1-15 (representing fossil site clusters)
#and 8 observations between 16-75 (representing map interpretations)
#and 5 lists with two observations between 1-15 and 4 between 16-75
def generate_lists(seed): #input a seed num for random shuffling
    # Create a new random generator instance with the given seed
    rng = random.Random(seed)
    # Create the ranges representing fossil vs map sites
    lista = list(range(1, 16))   # fossil sites
    listb = list(range(16, 76))  # map interps
    rng.shuffle(lista)
    rng.shuffle(listb)
    #create empty list for first set of lists
    groups_1a_8b = []
    for _ in range(5): #iterates through 5 times
        #pop one item from lista
        group_a = [lista.pop()]
        #pop 8 items from listb
        group_b = [listb.pop() for j in range(8)]
        #append to empty list
        groups_1a_8b.append(group_a + group_b)
    #create empty list for second set of lists
    groups_2a_4b = []
    for _ in range(5): #iterates through 5 times
        #pop two items from lista
        group_a = [lista.pop() for j in range(2)]
        #pop four items from listb
        group_b = [listb.pop() for j in range(4)]
        #append to empty list
        groups_2a_4b.append(group_a + group_b)
    #combine the two lists of lists
    all_groups = groups_1a_8b + groups_2a_4b
    #returns a list with 10 lists containing the appropriate
    #cluster numbers for the 10 spatial folds
    return all_groups

#function that uses the generate_lists function to create 10
#spatial folds from the training data
#inputs positive training dataframe, background training df,
#random seed number, cluster column name
def create_folds(df_pos, df_back, seed, cluster_column):
    #generates cluster lists using generate_lists function
    list_of_folds = generate_lists(seed)
    complete_folds = []  # To store the extracted dataframes
    # Shuffle background df with a fixed seed
    dfb_shuffled = df_back.sample(frac=1, \
                                  random_state=seed).reset_index(drop=True)
    #split background df into 10 parts
    dfb_parts = [dfb_shuffled.iloc[i::10] for i in range(10)]
    #iterating through the lists of folds (total 10 times)
    for i, fold in enumerate(list_of_folds):
        complete_fold = [] #create empty storage list
        #extract df rows with 'Cluster' values in the current list
        filtered_df = df_pos[df_pos[cluster_column].isin(fold)]
        #combine the positive extracted training rows with
        #the shuffled fraction of background rows
        complete_fold = pd.concat([filtered_df, dfb_parts[i]], \
                                  axis=0, ignore_index=True)
        #append the completed fold to the growing list of folds
        complete_folds.append(complete_fold)
    # return a list of 10 dfs, each with positive points correlated
    #to their cluster lists and a different random subset of negative
    #training points
    return complete_folds

#function that takes the 10 training folds and prepares them for
#k-fold cross validation by iterating through and leaving one
#fold out for training and combinging the remaining folds for
#testing
def create_train_test_folds(fold_dfs): #input the list of fold dfs
    train_folds = []  # To store training folds (each combining 9 dfs)
    test_folds = []   # To store test folds (each containing 1 df)
    #iterate through 10 times
    for i in range(len(fold_dfs)):
        # Create training fold by concatenating all dfs 
        #except the one at index i
        train_fold = pd.concat([df for j, df in enumerate(fold_dfs) \
                                if j != i], axis=0, ignore_index=True)
        # Test fold is df at index i
        test_fold = fold_dfs[i]
        # Append the created train and test fold to storage lists
        train_folds.append(train_fold)
        test_folds.append(test_fold)
    return train_folds, test_folds  # Return lists of training and test folds

#import positive and negative training csvs
#positive training targets contain training observations from 75 individual spatial clusters,
#15 of which contain 36 observations (540 total), and 60 of which contain 9 observations (540)
#for a total of 1080 positive training targets
df_pos = pd.read_csv('TrainingObservations_Positive.csv')
#negative training targets contain randomly generated background points, equal in number
#to positive points
df_back = pd.read_csv('TrainingObservations_Negative.csv')

#create 10 spatial folds
complete_folds = create_folds(df_pos, df_back, 42, 'Cluster')

#create training and test folds
train_folds, test_folds = create_train_test_folds(complete_folds)

#save list of train folds
with open('train_folds.pkl', 'wb') as f:
    pickle.dump(train_folds, f)

#save list of test folds
with open('test_folds.pkl', 'wb') as f:
    pickle.dump(test_folds, f)
