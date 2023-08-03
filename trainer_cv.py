import os
import shutil
import csv
import yaml
import torch
import torchaudio
import torchaudio.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from dataset import YaseenDataset, PascalDataset
from utils import collate_fn, pad_sequence, count_parameters
from wavenet import WaveNetModel
from wav2vec import getWav2VecCLS
from models.ACANet import ACANet
from models.M5 import M5
import inspect
from report import calculate_metrics

import logging

class Trainer:
    def __init__(self, config_file):
        self.config = self.load_config(config_file)
        self.device = torch.device(self.config["device"])
        self.save_dir = f"./results/{self.config['runname']}"
        os.makedirs(self.save_dir, exist_ok=True)

        self.datafolder = self.config["datafolder"]
        self.fold_dir = self.config["fold_dir"]
        self.n_folds = self.config["n_folds"]

        self.transform = self.load_transform()
        
        #Folder Locations for Testing Dataset
        # data_dir = "/scratch/jiaqi006/others/PASCAL"
        self.pascal_dir = self.config["pascal_dir"]
        self.pascal_split_dir = self.config["pascal_split_dir"]
        
        self.log_interval = self.config["log_interval"]
        self.n_epoch = self.config["n_epoch"]
        self.losses = []
        self.valid_acc = []
        self.curr_epoch = 1

        if os.path.exists(os.path.join(self.save_dir, "checkpoint.pt")):
            self.load_checkpoint()

        logging.basicConfig(level=logging.DEBUG, filename=f"./results/{self.config['runname']}/log.txt", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    def copyModelFile(self):
        modelfile = inspect.getfile(self.model.__class__)
        save_path = os.path.join(self.save_dir, modelfile.split("/")[-1])
        shutil.copyfile(modelfile, save_path)
        self.saved_model_file = save_path

    def save_checkpoint(self, is_best, fold, epoch):
        checkpoint_path = os.path.join(self.save_dir, f'{fold}_checkpoint.pt')
        best_model_path = os.path.join(self.save_dir, f'{fold}_best_model.pt')
        
        # Save the current model state
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, checkpoint_path)
        self.copyModelFile()

        # If this is the best model, save a separate copy as the best model
        if is_best:
            shutil.copyfile(checkpoint_path, best_model_path)
            logging.info(f'best model copied at epoch {epoch}')

    def load_checkpoint(self, fold):
        checkpoint_path = os.path.join(self.save_dir, f'{fold}_checkpoint.pt')
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.curr_epoch = checkpoint['epoch']

        def optimizer_to(optim, device):
            for param in optim.state.values():
                # Not sure there are any global tensors in the state dict
                if isinstance(param, torch.Tensor):
                    param.data = param.data.to(device)
                    if param._grad is not None:
                        param._grad.data = param._grad.data.to(device)
                elif isinstance(param, dict):
                    for subparam in param.values():
                        if isinstance(subparam, torch.Tensor):
                            subparam.data = subparam.data.to(device)
                            if subparam._grad is not None:
                                subparam._grad.data = subparam._grad.data.to(device)

        optimizer_to(self.optimizer, self.device)
        print(f'Epoch {self.curr_epoch} checkpoint loaded!')
        
        # Load the model state from the checkpoint
        # self.model.load_state_dict(torch.load(checkpoint_path))

    def load_bestmodel(self, fold):
        best_model_path = os.path.join(self.save_dir, f'{fold}_best_model.pt')
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Best model from fold {fold}, Epoch {checkpoint["epoch"]} loaded!')

    def load_config(self, config_file):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        return config

    def load_model(self):
        model_config = self.config["model"]
        # Import and initialize the model from a separate file
        if model_config == "wav2vec":
            model = getWav2VecCLS()
        elif model_config["model_name"] == "acanet":
            model = ACANet(**model_config)
        elif model_config["model_name"] == "M5":
            model = M5(**model_config)
        else:
            model = WaveNetModel(
                input_channels=model_config["input_channels"],
                classes=model_config["classes"],
                layers=model_config["layers"],
                blocks=model_config["blocks"],
                dilation_channels=model_config["dilation_channels"],
                residual_channels=model_config["residual_channels"],
                skip_channels=model_config["skip_channels"],

                kernel_size=model_config["kernel_size"],
                dtype=torch.FloatTensor,
                bias=model_config["bias"],
                fast=model_config["fast"],
            )
        
        print(f'Model loaded. model has {count_parameters(model)/1e6:.2f}M params')
        
        return model

    def load_data(self,fold):
        batch_size = self.config["batch_size"]
        num_workers = self.config["num_workers"]
        pin_memory = self.config["pin_memory"]

        train_set, valid_set_original, valid_set_codec = self.get_datasets(fold, self.config["codec_augment"])

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        valid_loader_original = torch.utils.data.DataLoader(
            valid_set_original,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        valid_loader_codec = torch.utils.data.DataLoader(
            valid_set_codec,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return train_loader, valid_loader_original, valid_loader_codec

    def get_datasets(self,fold, codec_augment = False):
        # Import the train dataset from a separate file
        train_set = YaseenDataset(self.datafolder, f'{self.fold_dir}/train{fold}.txt', codec_augment, True)
        valid_set_original = YaseenDataset(self.datafolder, f'{self.fold_dir}/fold{fold}.txt', False, True)
        valid_set_codec = YaseenDataset(self.datafolder, f'{self.fold_dir}/fold{fold}.txt', True, False)
        return train_set, valid_set_original, valid_set_codec

    def load_pascal_data(self):
        batch_size = self.config["batch_size"]
        num_workers = self.config["num_workers"]
        pin_memory = self.config["pin_memory"]

        dataA, dataBc, dataBn = self.get_pascal_datasets()

        dataA_loader = torch.utils.data.DataLoader(
            dataA,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        dataBc_loader = torch.utils.data.DataLoader(
            dataBc,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        dataBn_loader = torch.utils.data.DataLoader(
            dataBn,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return dataA_loader, dataBc_loader, dataBn_loader 

    def get_pascal_datasets(self):
        
        dataA = PascalDataset(self.pascal_dir,f'{self.pascal_split_dir}/DatasetA_n_m.txt',"A")
        dataBc = PascalDataset(self.pascal_dir,f'{self.pascal_split_dir}/DatasetB_clean_n_m.txt',"B_clean")
        dataBn = PascalDataset(self.pascal_dir,f'{self.pascal_split_dir}/DatasetB_noisy_n_m.txt',"B_noisy")

        return dataA, dataBc, dataBn

    def load_optimizer(self):
        optimizer_config = self.config["optimizer"]
        return optim.Adam(
            self.model.parameters(),
            lr=optimizer_config["lr"],
            weight_decay=optimizer_config["weight_decay"],
        )

    def load_scheduler(self):
        scheduler_config = self.config["scheduler"]
        return optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=scheduler_config["step_size"],
            gamma=scheduler_config["gamma"],
        )

    def load_transform(self, orig_freq = None, new_freq = None):
        transform_config = self.config["transform"]
        if orig_freq is not None:
            print(f'Orig_freq: {orig_freq}')
        if new_freq is not None:
            print(f'New_fred: {new_freq}')
        return transforms.Resample(
            orig_freq=transform_config["orig_freq"] if orig_freq == None else orig_freq,
            new_freq=transform_config["new_freq"] if new_freq == None else new_freq,
        ).to(self.device)

    def number_of_correct(self, pred, target):
        # count number of correct predictions
        return pred.squeeze().eq(target).sum().item()

    def get_likely_index(self,tensor):
        # find most likely label index for each element in the batch
        return tensor.argmax(dim=-1)

    def train_epoch(self, fold, epoch):
        self.model.train()
        self.model.to(self.device)
        
        #Loop for one epoch
        for batch_idx, (data, target) in enumerate(self.train_loader):

            data = data.to(self.device)
            target = target.to(self.device)

            # apply transform and model on whole batch directly on device
            data = self.transform(data)
            output = self.model(data)
            
            mapped_target = torch.tensor(list(map(lambda x: 0 if x == 0 else 1, target))).to(self.device)
            mapped_output = torch.stack((output[:,0],output[:,1:].sum(dim=1)),dim=1)

            if self.config["double_loss"]:
                loss1 = F.cross_entropy(output.squeeze(), target)
                loss2 = F.cross_entropy(mapped_output, mapped_target)
                
                loss = loss1+loss2
            else:
                loss = F.cross_entropy(output.squeeze(), target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        # test model
        acc_original = self.test_ds(self.model, self.valid_loader_original)
        acc_codec = self.test_ds(self.model, self.valid_loader_codec)

        # Check if the current loss is the best loss so far
        is_best = False
        is_best_codec = False
        logging.info(f'Current org acc: {acc_original}, best acc: {self.epoch_best_acc}, acc>best acc: {acc_original > self.epoch_best_acc}')
        logging.info(f'Current org codec acc: {acc_codec}, best codec acc: {self.epoch_best_acc_codec}, acc_codec>best codec acc: {acc_codec > self.epoch_best_acc_codec}')
        
        if acc_original > self.epoch_best_acc:
            self.epoch_best_acc = acc_original
            logging.info("new best found")
            is_best = True
        
        if acc_codec > self.epoch_best_acc_codec:
            self.epoch_best_acc_codec = acc_codec
            logging.info("new codec best found")
            is_best_codec = True

        # Save the model at checkpoints
        self.save_checkpoint(is_best, fold, epoch)
        
        #outside loop
        # print(f"Train Epoch: {epoch}\tLoss: {loss.item():.6f}\n")
        logging.info(f'\nTrain Epoch: {epoch}\tLoss: {loss.item():.6f}')

    def train_cv(self):
        self.fold_best = []
        self.fold_best_codec = []
        
        #loop for all folds
        for k in range(1,self.n_folds+1):
            self.train_loader, self.valid_loader_original, self.valid_loader_codec = self.load_data(k)

            #reload everything
            self.model = self.load_model()
            self.optimizer = self.load_optimizer()
            self.scheduler = self.load_scheduler()
            print(f'Now running Fold {k}')

            #loop for all epochs in one fold
            self.epoch_best_acc = 0
            self.epoch_best_acc_codec = 0

            for epoch in tqdm(range(self.curr_epoch, self.n_epoch + 1)):
                self.train_epoch(k, epoch)
                self.scheduler.step()
            
            #record best accuracy of the fold
            self.fold_best.append(self.epoch_best_acc)
            self.fold_best_codec.append(self.epoch_best_acc_codec)

        print("overall acc:",sum(self.fold_best)/len(self.fold_best))
        print("overall acc codec:",sum(self.fold_best_codec)/len(self.fold_best_codec))
        
        logging.info(f'list of fold best: {self.fold_best}')
        logging.info(f'list of fold best codec: {self.fold_best_codec}')

        logging.info(f'overall acc: {sum(self.fold_best)/len(self.fold_best)}')
        logging.info(f'overall acc codec: {sum(self.fold_best_codec)/len(self.fold_best_codec)}')

    def test_ds(self, model, dataloader):
        model.train(mode=False);  #evaluation
        model.to(self.device)
        correct = 0

        for data, target in dataloader:
            data = data.to(self.device)
            target = target.to(self.device)

            # apply transform and model on whole batch directly on device
            data = self.transform(data)
            output = model(data)
            pred = self.get_likely_index(output)
            correct += self.number_of_correct(pred, target)
        
        acc = correct/len(dataloader.dataset)
        return acc

    def test_OOD(self, fold):
        self.model = self.load_model()
        self.load_bestmodel(fold)
        self.model.train(mode=False);  #evaluation
        self.model.to(self.device)
        
        dataloaders = self.load_pascal_data() #tuple of the following: "Dataset A", "Dataset B clean", "Dataset B noisy"
        _, valid_loader_original, valid_loader_codec = self.load_data(fold)
        dataloaders = dataloaders + (valid_loader_original,)
        dataloaders = dataloaders + (valid_loader_codec,)

        dataset_names = ["Dataset A", "Dataset B clean", "Dataset B noisy", "Y18_Original","Y18_Codec"]


        test_accs = []

        for name_id, dataloader in enumerate(dataloaders):
            print(f'testing on {dataset_names[name_id]}...')

            #For saving the output files
            save_file = os.path.join(self.save_dir, f'predictions/{fold}_{dataset_names[name_id]}.csv')
            os.makedirs(os.path.join(self.save_dir, f'predictions'), exist_ok=True)
            save_targs = []
            save_preds = []

            self.transform = self.load_transform(dataloader.dataset[0][1])
            correct = 0
            for data, target in dataloader:

                data = data.to(self.device)
                target = target.to(self.device)

                # apply transform and model on whole batch directly on device
                data = self.transform(data)
                output = self.model(data)
                pred = self.get_likely_index(output)
                
                if name_id < 3:
                    map_index = lambda x: 0 if x==0 else 1
                    pred = torch.tensor(list(map_index(i) for i in pred)).to(self.device)
                
                save_targs = [*save_targs,*target.tolist()]
                save_preds = [*save_preds,*pred.tolist()]
                correct += self.number_of_correct(pred, target)
            
            #combine lists for saving into csv
            save_output = list(zip(save_targs, save_preds))
            with open(save_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Targets', 'Predictions'])  # Write the header row
                writer.writerows(save_output)  # Write the data rows

            calculate_metrics(save_file)
            acc = correct/len(dataloader.dataset)
            test_accs.append(acc)

        return test_accs
    
    def test_OOD_cv(self):
        self.fold_best_OOD = {
            "Dataset A": [],
            "Dataset Bc": [],
            "Dataset Bn": [],
            "Y18_Original": [],
            "Y18_Codec": [],
        }
        
        #loop for all folds
        for k in range(1,self.n_folds+1):
            accs = self.test_OOD(k)
            for i, dataset in enumerate(self.fold_best_OOD.keys()):
                self.fold_best_OOD[dataset].append(accs[i])
        
        print(f'list of fold best: {self.fold_best_OOD}')
        logging.info(f'list of fold best: {self.fold_best_OOD}')

        for i, dataset in enumerate(self.fold_best_OOD.keys()):
            print(f'average acc for {dataset}: {sum(self.fold_best_OOD[dataset])/len(self.fold_best_OOD[dataset])}')
            logging.info(f'average acc for {dataset}: {sum(self.fold_best_OOD[dataset])/len(self.fold_best_OOD[dataset])}')