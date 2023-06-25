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
import inspect

class Trainer:
    def __init__(self, config_file):
        self.config = self.load_config(config_file)
        self.device = torch.device(self.config["device"])
        self.save_dir = f"./results/{self.config['runname']}"
        os.makedirs(self.save_dir, exist_ok=True)

        self.datafolder = self.config["datafolder"]
        self.model = self.load_model()
        self.train_loader, self.valid_loader, self.test_loader = self.load_data()
        self.optimizer = self.load_optimizer()
        self.scheduler = self.load_scheduler()
        self.transform = self.load_transform()
        
        self.log_interval = self.config["log_interval"]
        self.n_epoch = self.config["n_epoch"]
        self.pbar_update = 1 / (len(self.train_loader) + len(self.test_loader))
        self.losses = []
        self.valid_acc = []
        self.curr_epoch = 1

        if os.path.exists(os.path.join(self.save_dir, "checkpoint.pt")):
            self.load_checkpoint()

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

    def load_data(self):
        batch_size = self.config["batch_size"]
        num_workers = self.config["num_workers"]
        pin_memory = self.config["pin_memory"]

        train_set, valid_set, test_set = self.get_datasets()

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

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return train_loader, valid_loader, test_loader

    def get_datasets(self):
        # Import the train dataset from a separate file
        train_set = YaseenDataset(self.datafolder, self.config["train_list"])
        valid_set = YaseenDataset(self.datafolder, self.config["valid_list"]) #Change this to a proper validation dataset later. Or use K-fold.
        test_set = YaseenDataset(self.datafolder, self.config["test_list"])
        return train_set, valid_set, test_set

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

    def train_epoch(self, epoch, pbar):
        self.model.train()
        self.model.to(self.device)
        for batch_idx, (data, target) in enumerate(self.train_loader):

            data = data.to(self.device)
            target = target.to(self.device)

            data = self.transform(data)
            output = self.model(data)

            loss = F.nll_loss(output.squeeze(), target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            

            if batch_idx % self.log_interval == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} ({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

            pbar.update(self.pbar_update)  #progress bar #this vaiable is outside the function
            # record loss
            self.losses.append(loss.item()) #accessing variable that is outside the function

    def valid_epoch(self, epoch, pbar):
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

            # update progress bar
            pbar.update(self.pbar_update)

        #record acc
        curr_acc = 100. * correct / len(self.valid_loader.dataset)
        self.valid_acc.append(curr_acc)

        # Check if the current loss is the best loss so far
        is_best = False
        if curr_acc >= min(self.valid_acc):
            print("new best found")
            is_best = True
            
        # Save the model at checkpoints
        self.save_checkpoint(is_best, epoch)

        print(f"\nValid Epoch: {epoch}\tAccuracy: {correct}/{len(self.valid_loader.dataset)} ({100. * correct / len(self.valid_loader.dataset):.0f}%)\n")
    
    def train(self):
        self.valid_acc = []

        with tqdm(total=self.n_epoch) as pbar:
            for epoch in range(self.curr_epoch, self.n_epoch + 1):
                self.train_epoch(epoch, pbar)
                self.valid_epoch(epoch, pbar)
                self.scheduler.step()

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