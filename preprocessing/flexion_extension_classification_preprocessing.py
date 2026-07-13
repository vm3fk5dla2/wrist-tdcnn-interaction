import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class SensorDataset (Dataset):
    def __init__ (self, params):
        self.sensors = {"sensor1": [], "sensor2": [], "sensor3": [], "sensor4": [], "sensor5": []}
        self.total_index = []
        self.labels = []
        self.total_data = []
        self.params = params


    def load_file (self, file):
        data = pd.read_csv(file,
                           usecols=["Counts", "GestureIdentifier", "Thumb", "Index", "Middle", "Ring", "Little"],
                           encoding='ISO-8859-1')
        data_dict = data.to_dict (orient = 'list')

        cur_counts = data_dict["Counts"]
        cur_labels = data_dict["GestureIdentifier"]
        sensor1 = data_dict["Thumb"]
        sensor2 = data_dict["Index"]
        sensor3 = data_dict["Middle"]
        sensor4 = data_dict["Ring"]
        sensor5 = data_dict["Little"]

        self.sensors["sensor1"].extend (sensor1)
        self.sensors["sensor2"].extend (sensor2)
        self.sensors["sensor3"].extend (sensor3)
        self.sensors["sensor4"].extend (sensor4)
        self.sensors["sensor5"].extend (sensor5)
        self.labels.extend (cur_labels)
        self.total_index.extend (cur_counts)


    def min_max_normalization (self):
        for (sensor_name, sensor_value) in self.sensors.items ():
            sensor_value = np.array (sensor_value)
            min_sensor_value = np.amin (sensor_value)
            max_sensor_value = np.amax (sensor_value)
            self.sensors[sensor_name] = (sensor_value - min_sensor_value) / (max_sensor_value - min_sensor_value)


    def smoothing (self, chunked_list, num_to_ignore):
        trimmed_list = np.sort (chunked_list)[num_to_ignore: -num_to_ignore]

        return sum (trimmed_list) / len (trimmed_list) if len (trimmed_list) > 0 else 0


    def sliding_timewindow_and_smoothing (self, window_size, num_to_ignore):
        total_data = {"sensors": [], "label": []}
        total_index = {"counts": self.total_index, "gesture_inits": []}

        for i in range (self.params.start_index, len (self.sensors["sensor1"]) - window_size + 1, self.params.stride):
            total_index["gesture_inits"].append (self.total_index[i])

            empty_list = list ()
            for sensor_value in self.sensors.values ():
                sensor_list = sensor_value[i: i + window_size]
                chunked_list = [sensor_list[j: j + 10] for j in range (0, window_size, 10)]
                sorted_chunked_list = [np.sort (list (chunk)) for chunk in chunked_list]
                smooth_list = np.array ([self.smoothing (chunk, num_to_ignore) for chunk in sorted_chunked_list])
                empty_list.append (smooth_list)

            total_data["sensors"].append (empty_list)
            total_data["label"].append (self.labels[i: i + window_size])

        self.total_data = total_data
        self.total_index = total_index


    def labeling (self, label_threshold):
        new_labels = []
        only_gestures = []
        only_gestures_index = []
        i = 0
        for labels in (self.total_data["label"]):
            only_gestures.append (self.total_data["sensors"][i])
            only_gestures_index.append (self.total_index["gesture_inits"][i])
            if labels.count (51) >= label_threshold:
                new_labels.append (0)
            elif labels.count (52) >= label_threshold:
                new_labels.append (1)
            elif labels.count (61) >= label_threshold:
                new_labels.append (2)
            elif labels.count (62) >= label_threshold:
                new_labels.append (3)
            elif labels.count (71) >= label_threshold:
                new_labels.append (4)
            elif labels.count (72) >= label_threshold:
                new_labels.append (5)
            elif labels.count (81) >= label_threshold:
                new_labels.append (6)
            elif labels.count (82) >= label_threshold:
                new_labels.append (7)
            elif labels.count (91) >= label_threshold:
                new_labels.append (8)
            elif labels.count (92) >= label_threshold:
                new_labels.append (9)
            else:
                only_gestures.pop ()
                only_gestures_index.pop ()
            i += 1

        self.total_data["label"] = new_labels
        self.total_data["sensors"] = only_gestures
        self.total_index["gesture_inits"] = only_gestures_index


    def parsing (self):
        self.min_max_normalization ()
        self.sliding_timewindow_and_smoothing (self.params.window_size, self.params.num_to_ignore)
        self.labeling (self.params.label_threshold)

    def __len__ (self):
        return len (self.total_data["label"])

    def __getitem__ (self, index):
        item = {"sensors": self.total_data["sensors"][index], "label": self.total_data["label"][index]}
        return item
