from dataclasses import dataclass
import os


@dataclass
class Params:

    train_dir: str = "examples/data_classification/train"
    validate_dir: str = "examples/data_classification/valid"
    test_dir: str = "examples/data_classification/test"

    # Newly trained checkpoints are written here. Re-running training may
    # overwrite files with the same names in this directory.
    training_output_dir: str = "checkpoints/trained"

    # Offline inference uses the manuscript checkpoint by default.
    released_checkpoint_path: str = os.path.join(
        "checkpoints",
        "released",
        "best_model_classification.pth",
    )

    best_model_filename: str = "best_model_classification.pth"
    top_model_filename: str = "top_model_{}_{}.pth"
    interrupted_model_filename: str = "interrupted_model_epoch_{}.pth"

    # model hyperparameters
    window_size: int = 120
    label_threshold: int = 120
    stride: int = 1
    num_to_ignore: int = 3
    start_index: int = 0
    early_stopping_threshold: float = 99.0

    # training hyperparameters
    lr: float = 0.0001
    batch_size: int = 8
    num_workers: int = 4
    num_epoch: int = 2500

    @property
    def training_best_model_path(self) -> str:
        return os.path.join(self.training_output_dir, self.best_model_filename)

    def training_top_model_path(self, accuracy, epoch) -> str:
        filename = self.top_model_filename.format(accuracy, epoch)
        return os.path.join(self.training_output_dir, filename)

    def training_interrupted_model_path(self, epoch) -> str:
        filename = self.interrupted_model_filename.format(epoch)
        return os.path.join(self.training_output_dir, filename)
