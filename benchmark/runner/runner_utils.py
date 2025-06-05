import yaml

def get_baud_rate(device_name, mode=None, yaml_path="devices.yaml"):
    with open(yaml_path, 'r') as f:
        devices = yaml.safe_load(f)

    for device in devices:
        if device.get("name") == device_name and "baud" in device:
            baud = device["baud"]
            if isinstance(baud, dict):
                if not mode:
                    raise ValueError(f"Mode required to resolve baud for device '{device_name}'")
                mode_map = {'e': 'energy', 'p': 'performance', 'a': 'accuracy'}
                baud_key = mode_map.get(mode.lower())
                if baud_key and baud_key in baud:
                    return baud[baud_key]
                else:
                    raise ValueError(f"Baud mode '{mode}' not found for device: {device_name}")
            return baud
    raise ValueError(f"No baud rate found for device: {device_name}")
