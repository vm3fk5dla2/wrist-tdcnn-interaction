class BLEPacketDataset:
    def __init__(self, params):
        self.sensors = {}
        self.params = params


    def add_packet(self, packet):
        """
        Add one already-parsed BLE packet.

        Expected format:
            {"Sensor 1": value1, "Sensor 2": value2, ...}
        """
        # 1) Fast-path: pre-parsed sensors
        if isinstance(packet, dict):
            for sensor, value in packet.items():
                if sensor not in self.sensors:
                    self.sensors[sensor] = []
                self.sensors[sensor].append(value)
            return

        # 2) Back-compat: raw BLE payload as bytes/bytearray/hex str
        if isinstance(packet, (bytes, bytearray)):
            packet_hex = packet.hex()
        elif isinstance(packet, str):
            packet_hex = packet
        else:
            raise ValueError("Unsupported packet type. Must be bytes, bytearray, str, or dict.")

        sensors = self.parse_ble_packet(
            packet_hex,
            selected_channels=self.params.selected_channels,
        )
        for sensor, value in sensors.items():
            if sensor not in self.sensors:
                self.sensors[sensor] = []
            self.sensors[sensor].append(value)


    @staticmethod
    def parse_ble_packet (packet_hex, selected_channels = (1, )) -> dict:
        packet_meaningful = packet_hex[2:]
        if len (packet_meaningful) % 4 != 0:
            raise ValueError ("The data does not break evenly into 4-character chunks.")
        channels_raw = [packet_meaningful[i: i + 4] for i in range (0, len (packet_meaningful), 4)]

        sensors = {}
        for idx, channel in enumerate (selected_channels, start = 1):
            index = channel - 1
            if index >= len (channels_raw):
                raise ValueError (f"Channel {channel} is not available in the packet.")
            chunk = channels_raw[index]
            reversed_chunk = chunk[2:] + chunk[:2]
            value = int (reversed_chunk, 16)
            sensors[f"Sensor {idx}"] = value
        return sensors