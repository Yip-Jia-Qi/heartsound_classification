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
        print(len(training_data))
        
        # Load the data from the current fold for testing
        test_file = split_list[i]
        with open(test_file, "r") as file:
            testing_data = file.read().splitlines()
        print(len(testing_data))
        # Perform training using training_data
        # Perform testing using testing_data
        # ...

# Example usage
split_dir = "Y18_10_fold"
train_with_cross_validation(10, split_dir)