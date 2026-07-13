import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from models.flexion_extension_classification_model import UltraLightCNN1D
from preprocessing.flexion_extension_classification_preprocessing import SensorDataset
from configs.flexion_extension_classification_parameters import Params


def validate(device, model, validate_sensor_dataLoader, criterion):
    model.eval ()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad ():
        for _, batch in enumerate (validate_sensor_dataLoader):
            resistance = batch["sensors"]

            if isinstance (resistance, list):
                resistance = torch.stack (resistance).permute (1, 0, 2).to (device)
            else:
                resistance = resistance.permute (1, 0, 2).to (device)
            resistance = resistance.float ()
            target = batch["label"].long ().to (device)

            output = model (resistance)
            predicted_labels = output.argmax (dim = 1)

            loss = criterion (output, target)
            total_loss += loss.item () * target.size (0)

            total_correct += (predicted_labels == target).sum ().item ()
            total_samples += target.size (0)

    average_loss = total_loss / total_samples
    overall_accuracy = 100.0 * total_correct / total_samples

    return overall_accuracy, average_loss


def main ():
    device = torch.device ('cuda' if torch.cuda.is_available () else 'cpu')
    torch.cuda.empty_cache ()

    params = Params()

    validate_sensor_dataset = SensorDataset (params)
    for file_name in os.listdir (params.test_dir):
        file = os.path.join (params.test_dir, file_name)
        if os.path.isfile (file):
            validate_sensor_dataset.load_file (file)
    validate_sensor_dataset.parsing ()

    validate_sensor_dataLoader = DataLoader (validate_sensor_dataset,
                                             batch_size = params.batch_size,
                                             shuffle = False,
                                             num_workers = params.num_workers
                                             )

    model = UltraLightCNN1D ().to (device)
    model.load_state_dict(torch.load(params.best_model_path, map_location = device))

    criterion = nn.CrossEntropyLoss ()

    return validate (device, model, validate_sensor_dataLoader, criterion)


if __name__ == "__main__":
    main ()