import sys
from trainer_cv import Trainer
from tqdm import tqdm


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the filepath of the config file.")
        sys.exit(1)

    config_file = sys.argv[1]
    trainer = Trainer(config_file)

    trainer.train()
    # trainer.test()    
    # Start training
    # with tqdm(total=trainer.n_epoch) as pbar:
    #     for epoch in range(1, trainer.n_epoch + 1):
    #         trainer.train(epoch, pbar)
    #         trainer.test(epoch, pbar)
    #         trainer.scheduler.step()

    # Perform prediction using the trained model
    # Example prediction with a sample tensor
    # sample_tensor = ...
    # prediction = trainer.predict(sample_tensor)
    # print("Prediction:", prediction)
