import sys
from trainer_cv import Trainer
from tqdm import tqdm


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the filepath of the config file.")
        sys.exit(1)

    config_file = sys.argv[1]
    trainer = Trainer(config_file)

    # trainer.train_cv()
    trainer.test_OOD_cv()