import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import os
import copy

from models.flexion_extension_classification_model import UltraLightCNN1D
from preprocessing.flexion_extension_classification_preprocessing import SensorDataset
from configs.flexion_extension_classification_parameters import Params


def train (device, model, optimizer, train_dataloader, criterion):
    model.train ()

    for _, batch in enumerate (train_dataloader):
        resistance = batch["sensors"]

        if isinstance (resistance, list):
            resistance = torch.stack (resistance).permute (1, 0, 2).to (device)
        else:
            resistance = resistance.permute (1, 0, 2).to (device)
        resistance = resistance.float ()
        target = batch["label"].long ().to (device)

        output = model (resistance)
        loss = criterion (output, target)

        optimizer.zero_grad ()
        loss.backward ()
        optimizer.step ()


def validate(device, model, validate_sensor_dataLoader):
    model.eval ()

    total_correct = 0
    total_samples = 0

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

            total_correct += (predicted_labels == target).sum ().item ()
            total_samples += target.size (0)

    overall_accuracy = 100.0 * total_correct / total_samples

    return overall_accuracy


def main ():
    device = torch.device ('cuda' if torch.cuda.is_available () else 'cpu')
    torch.cuda.empty_cache ()

    params = Params()

    os.makedirs(params.training_output_dir, exist_ok = True)

    model = UltraLightCNN1D ().to (device)

    train_sensor_dataset = SensorDataset (params)
    for file_name in os.listdir (params.train_dir):
        file = os.path.join (params.train_dir, file_name)
        if os.path.isfile (file):
            train_sensor_dataset.load_file (file)
    train_sensor_dataset.parsing ()

    validate_sensor_dataset = SensorDataset (params)
    for file_name in os.listdir (params.validate_dir):
        file = os.path.join (params.validate_dir, file_name)
        if os.path.isfile (file):
            validate_sensor_dataset.load_file (file)
    validate_sensor_dataset.parsing ()

    train_sensor_dataLoader = DataLoader (train_sensor_dataset,
                                          batch_size = params.batch_size,
                                          shuffle = True,
                                          num_workers = params.num_workers
                                          )

    validate_sensor_dataLoader = DataLoader (validate_sensor_dataset,
                                             batch_size = params.batch_size,
                                             shuffle = False,
                                             num_workers = params.num_workers
                                             )

    optimizer = optim.Adam (model.parameters (), lr = params.lr)
    criterion = nn.CrossEntropyLoss ()

    best_accuracy = -1
    best_epoch = 0
    top_five_acc_models = []

    try:
        for epoch in range (params.num_epoch):
            train (device, model, optimizer, train_sensor_dataLoader, criterion)
            acc = validate (device, model, validate_sensor_dataLoader)

            if acc >= best_accuracy:
                torch.save (model.state_dict (), params.training_best_model_path)
                best_accuracy = acc
                best_epoch = epoch + 1

            if acc >= params.early_stopping_threshold:
                deep_copied_model = copy.deepcopy (model)
                top_five_acc_models.append ((deep_copied_model, acc, epoch))
                if len (top_five_acc_models) >= 5:
                    break

        for (deep_copied_model, acc, epoch) in top_five_acc_models:
            torch.save (deep_copied_model.state_dict (), params.training_top_model_path(acc, epoch + 1))

    except KeyboardInterrupt:
        torch.save (model.state_dict(), params.training_interrupted_model_path(best_epoch))


if __name__ == "__main__":
    main ()
