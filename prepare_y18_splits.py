import os
import random

def get_kfold_splits(random, data, k):
    # Calculate the size of each fold
    fold_size = len(data) // k
    
    # Shuffle the data
    random.shuffle(data)

    # Create k-fold cross-validation splits
    for fold in range(k):
        # Determine the start and end indices for the current fold
        start_index = fold * fold_size
        end_index = (fold + 1) * fold_size
        
        # Get the data for the current fold
        fold_data = data[start_index:end_index]
        yield fold_data

def combine_txt_files(split_list):
    training_data = []
    
    for file_name in split_list:
        with open(file_name, "r") as file:
            file_data = file.read().splitlines()
            training_data.extend(file_data)
    
    return training_data

def train_with_cross_validation(k, split_dir):
    split_list = [f"{split_dir}/fold{i+1}.txt" for i in range(k)]
    
    # Perform k-fold cross-validation
    for i in range(k):
        # Combine the data from the 9 other folds for training
        train_files = [split_list[j] for j in range(k) if j != i]
        training_data = combine_txt_files(train_files)
        with open(f"{split_dir}/train{i+1}.txt", "w") as file:
            file.write("\n".join(training_data))
        


# 10-fold split list generation
if __name__ == '__main__':
    datapath = "/scratch/jiaqi006/others/Yaseen_CHSSUMF/"
    output_dir = "Y18_10_fold"
    folders = ['AS_New_3', 'MVP_New_3', 'N_New_3', 'MS_New_3', 'MR_New_3']
    data_dir = []
    
    
    for i in folders:
        data_dir.append(list("/"+i+"/"+f for f in os.listdir(datapath+i)))
    
    gen = random.Random(123)
    
    #get a big list 5 items each containing 10 folds of 20 files each
    big_list = []
    for i in range(len(folders)):
        big_list.append(list(get_kfold_splits(gen, data_dir[i],10)))
    
    # recombine the big list into 10 folds with 100 files each, with an equal distribution of each category
    for fold in range(10):
        fold_list = []
        for j in range(len(folders)):
            fold_list.extend(big_list[j][fold])

        #need to shuffle to ensure randomness
        gen.shuffle(fold_list)

        #save the folds
        file_name = f"fold{fold+1}.txt"
        with open(f"{output_dir}/{file_name}", "w") as file:
            file.write("\n".join(fold_list))

    # combine folds for training sets
    train_with_cross_validation(10, output_dir)