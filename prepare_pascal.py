import os
import csv

def prepare_lists(data_dir,save_dir):
    print(os.listdir(data_dir))

    dataseta = ['Atraining_normal', 'Atraining_murmur']
    data_A_list = []
    for i in dataseta:
        for filepath in os.listdir(data_dir+'/'+i):
            if filepath == '.DS_Store':
                continue
            data_A_list.append('/'+i+'/'+filepath)

    # print(len(data_A_list))
    # print(data_A_list[0])
    # print(data_A_list[-1])
    # print(data_A_list[0].split('training_')[1].split('/')[0])
    # print(data_A_list[-1].split('training_')[1].split('/')[0])

    with open(save_dir+'DatasetA_n_m.txt', 'w') as f:
        for line in data_A_list:
            f.write(f"{line}\n")
    print("datasetA list saved")

    datasetb = ['Training B Normal', 'Btraining_murmur']
    data_B_list = []
    for i in datasetb:
        for filepath in os.listdir('/'+data_dir+'/'+i):
            if filepath == '.DS_Store':
                continue
            if filepath == 'Btraining_noisynormal':
                continue
            if filepath == 'Btraining_noisymurmur':
                continue
            data_B_list.append('/'+i+'/'+filepath)
    
    with open(save_dir+'DatasetB_clean_n_m.txt', 'w') as f:
        for line in data_B_list:
            f.write(f"{line}\n")
    print("datasetB list saved")
    
    # print(len(data_B_list))
    # print(data_B_list[0])
    # print(data_B_list[-1])
    # print(data_B_list[0].split('raining')[1].split('/')[0].strip("_").strip(" B ").lower())
    # print(data_B_list[-1].split('raining')[1].split('/')[0].strip("_").strip(" B ").lower())

    datasetb_noisy = ['Training B Normal/Btraining_noisynormal', 'Btraining_murmur/Btraining_noisymurmur']
    data_B_noisy_list = []
    for i in datasetb_noisy:
        for filepath in os.listdir(data_dir+'/'+i):
            if filepath == '.DS_Store':
                continue
            data_B_noisy_list.append('/'+i+'/'+filepath)

    with open(save_dir+'DatasetB_noisy_n_m.txt', 'w') as f:
        for line in data_B_noisy_list:
            f.write(f"{line}\n")
    print("datasetB_noisy list saved")

    # print(len(data_B_noisy_list))
    # print(data_B_noisy_list[0])
    # print(data_B_noisy_list[-1])

    # cleaner = lambda x: x.split("noisy")[-1].split("/")[0]
    # print(data_B_noisy_list[0].split("noisy")[-1].split("/")[0])
    # print(data_B_noisy_list[-1].split("noisy")[-1].split("/")[0])
    # print(cleaner(data_B_noisy_list[0]))
    # print(cleaner(data_B_noisy_list[-1]))

if __name__ == '__main__':
    data_dir = "/scratch/jiaqi006/others/PASCAL"
    save_dir = "./pascal_lists/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    prepare_lists(data_dir,save_dir)