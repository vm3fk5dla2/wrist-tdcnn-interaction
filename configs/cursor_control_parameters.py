from dataclasses import dataclass
import os


@dataclass
class Params:
    # BLE parameters
    ble_characteristic_uuid: str = "YOUR_BLE_DEVICE_UUID"
    ble_device_address: str = "YOUR_BLE_DEVICE_ADDRESS"


    # directories
    model_dir: str = "examples/models"


    # model filenames
    model_filename: str = "best_model_cursor_control.pth"


    # model hyperparameters
    window_size: int = 120
    label_threshold: int = 120
    pinch_threshold: int = 120
    stride: int = 1
    
    num_to_ignore: int = 3
    start_index: int = 0
    early_stopping_threshold: float = 85.0
    selected_channels: tuple[int, ...] = (2, 1)


    # training hyperparameters
    lr: float = 0.0001
    batch_size: int = 8
    num_workers: int = 4
    num_epoch: int = 200

    @property
    def model_path(self) -> str:
        return os.path.join(self.model_dir, self.model_filename)