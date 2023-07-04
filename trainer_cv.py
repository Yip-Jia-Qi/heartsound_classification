import os
import shutil

import yaml
import torch
import torchaudio
import torchaudio.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from dataset import YaseenDataset
from utils import collate_fn, pad_sequence, count_parameters
from wavenet import WaveNetModel
from wav2vec import getWav2VecCLS
from models.ACANet import ACANet
import inspect

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
        self.model = self.load_model()
        self.optimizer = self.load_optimizer()
        self.scheduler = self.load_scheduler()
        self.transform = self.load_transform()
        
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

    def save_checkpoint(self, is_best, epoch):
        checkpoint_path = os.path.join(self.save_dir, "checkpoint.pt")
        best_model_path = os.path.join(self.save_dir, "best_model.pt")
        
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

    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.save_dir, "checkpoint.pt")
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

    def load_bestmodel(self):
        best_model_path = os.path.join(self.save_dir, "best_model.pt")
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Best model from Epoch {checkpoint["epoch"]} loaded!')

    def load_config(self, config_file):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        return config

    def load_model(self):
        model_config = self.config["model"]
        # Import and initialize the model from a separate file
        if model_config == "wav2vec":
            model = getWav2VecCLS()
        if model_config["model_name"] == "acanet":
            model = ACANet(**model_config)
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

        train_set, valid_set = self.get_datasets(fold)

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return train_loader, valid_loader

    def get_datasets(self,fold):
        # Import the train dataset from a separate file
        train_set = YaseenDataset(self.datafolder, f'{self.fold_dir}/train{fold}.txt')
        valid_set = YaseenDataset(self.datafolder, f'{self.fold_dir}/fold{fold}.txt')
        return train_set, valid_set

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

    def load_transform(self):
        transform_config = self.config["transform"]
        return transforms.Resample(
            orig_freq=transform_config["orig_freq"],
            new_freq=transform_config["new_freq"],
        )

    def number_of_correct(self, pred, target):
        # count number of correct predictions
        return pred.squeeze().eq(target).sum().item()

    def get_likely_index(self,tensor):
        # find most likely label index for each element in the batch
        return tensor.argmax(dim=-1)

    def train_epoch(self, epoch):
        self.model.train()
        self.model.to(self.device)
        
        for batch_idx, (data, target) in enumerate(self.train_loader):

            data = data.to(self.device)
            target = target.to(self.device)

            # apply transform and model on whole batch directly on device
            data = self.transform(data)
            output = self.model(data)
            
            loss = F.nll_loss(output.squeeze(), target)
            

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # record loss
            self.losses.append(loss.item()) #accessing variable that is outside the function

            # Check if the current loss is the best loss so far
            is_best = False
            if loss.item() < min(self.losses):
                logging.info("new best found")
                is_best = True
            
            # Save the model at checkpoints
            self.save_checkpoint(is_best, epoch)
        
        #outside loop
        # print(f"Train Epoch: {epoch}\tLoss: {loss.item():.6f}\n")
        logging.info(f"\nTrain Epoch: {epoch}\tLoss: {loss.item():.6f}")

    def valid_epoch(self):
        self.model.train(mode=False);  #evaluation
        self.model.to(self.device)
        correct = 0
        
        for data, target in self.valid_loader:

            data = data.to(self.device)
            target = target.to(self.device)

            # apply transform and model on whole batch directly on device
            data = self.transform(data)
            output = self.model(data)

            pred = self.get_likely_index(output)
            correct += self.number_of_correct(pred, target)

        #record acc
        curr_acc = 100. * correct / len(self.valid_loader.dataset)
        self.valid_acc.append(curr_acc)
        print(f'\nValid Accuracy: {correct}/{len(self.valid_loader.dataset)} ({100. * correct / len(self.valid_loader.dataset):.0f}%)')
        logging.info(f'\nValid Accuracy: {correct}/{len(self.valid_loader.dataset)} ({100. * correct / len(self.valid_loader.dataset):.0f}%)')
    
    def train(self):
        self.valid_acc = []
        # with tqdm(total=self.n_epoch) as pbar:
        for k in range(1,self.n_folds+1):
            self.train_loader, self.valid_loader = self.load_data(k)
            print(f'Now running Fold {k}')
            for epoch in tqdm(range(self.curr_epoch, self.n_epoch + 1)):
                self.train_epoch(epoch)
                self.scheduler.step()
            self.valid_epoch()
        print(self.valid_acc)
        print("overall acc:",sum(self.valid_acc)/len(self.valid_acc))
        logging.info(self.valid_acc)
        logging.info(f'overall acc: {sum(self.valid_acc)/len(self.valid_acc)}')
        
        # self.test()

    def test(self):
        self.load_bestmodel()
        self.model.train(mode=False);  #evaluation
        self.model.to(self.device)
        correct = 0
        for data, target in self.test_loader:

            data = data.to(self.device)
            target = target.to(self.device)

            # apply transform and model on whole batch directly on device
            data = self.transform(data)
            output = self.model(data)

            pred = self.get_likely_index(output)
            correct += self.number_of_correct(pred, target)

        print(f"\nTest Accuracy: {correct}/{len(self.test_loader.dataset)} ({100. * correct / len(self.test_loader.dataset):.0f}%)\n")