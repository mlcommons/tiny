# LPM01A-lib
## These files were obtained from https://github.com/lazicdanilo/LPM01A-lib

## Description

This is a Python library for the [X-NUCLEO-LPM01A](https://www.st.com/en/evaluation-tools/x-nucleo-lpm01a.html) STM32 Power shield, Nucleo expansion board for power consumption measurement.

For more information about the X-NUCLEO-LPM01A communication protocol please refer to the [UM2269 Getting started with PowerShield firmware](https://www.st.com/resource/en/user_manual/dm00418905-getting-started-with-powershield-firmware-stmicroelectronics.pdf)

## Prerequisites

### Data Acquisition

- [PySerial](https://pypi.org/project/pyserial/)

### Data Analysis

- [Click](https://pypi.org/project/click/)
- [Pandas](https://pypi.org/project/pandas/)
- [Matplotlib](https://pypi.org/project/matplotlib/)

## Usage

### LPM01A

```python
from src.LPM01A import LPM01A

lpm = LPM01A("/dev/ttyACM0", 3864000)
lpm.init_device(mode="ascii", voltage=3300, freq=5000, duration=0)
lpm.start_capture()
lpm.read_and_parse_data()
```

See [data_acquisition.py](data_acquisition.py) for a complete example.

### DataAnalysis

```python
from src.DataAnalysis import DataAnalysis

da = DataAnalysis("example.csv", 10_000, 30_000)
print(f"Average current consumption {da.calculate_average_current()} Ah")
da.plot_current_vs_timestamp()
```

See [data_analysis.py](data_analysis.py) for a complete example.

### data_analysis.py

```bash
# Calculates average current consumption between 3ms and 3.7ms and plots data
./data_analysis.py example.csv -s 3_000_000 -e 3_700_000 -p 
```

![Usage example](assets/pics/data_analysis_usage_example.gif)

## Limitations

Both the data acquisition and data analysis scripts are limited to Unux-like systems, as the serial port is accessed through the `/dev/ttyACM0` path.
Both scripts were tested on an Arch Linux and a Raspberry Pi 3B+ running Raspbian.

As this driver can work with ASCII mode only, the acquisition frequency of the LPM01A is limited to maximum of 50k samples / second.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
