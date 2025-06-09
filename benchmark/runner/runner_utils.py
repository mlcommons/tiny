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


def print_tee(str_to_print, outfile=None, line_end="\n"):
    """
    prints str_to_print to the stdout and optionally also to a file
    If outfile is a string, it opens that file and writes str_to_print
    to it.  If outfile is an open filehandle, write str_to_print to the file.
    """
    print(str_to_print)
    if outfile and isinstance(outfile, str):
        with open(outfile, 'a') as fpo:
            fpo.write(str_to_print+line_end)
    elif outfile and isinstance(outfile, io.IOBase) and not outfile.closed:
        outfile.write(str_to_print+line_end)