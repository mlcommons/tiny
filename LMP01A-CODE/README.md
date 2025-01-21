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

## Running Your own device and storing data
If you run this command in your terminal
```bash
python -m data_acquisition.py
```
This will collect data until a keyboard interrupt (ctrl+c for windows) is enacted

Below will plot the data, remove the -s and -e to plot the fully collected data

```bash
# Calculates average current consumption between 1ms and 4ms and plots data
./data_analysis.py csv_folder_name/your.csv -s 1_000_000 -e 4_000_000 -p 
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
