[[_TOC_]]

## Energy Test Connections

### Power Board (LPM01A)
![LPM01A Wiring](img/LPM01A.jpg)
### Interface Board (STM32H573I-DK)
![STM32H573I-DK Top Wiring](img/STM32H573I-DK-Top.png)
![STM32H573I-DK Bottom Wiring](img/STM32H573I-DK-Bottom.png)
### Device Under Test (L4R5ZI)
![DUT Wiring](img/L4R5ZI.png)

There should be an extra wire connecting from CN14 port 2 to the Port on the interace board directly next to open jumper on the DUT Board


## Test Runner
The test runner connects to the interface board and the power board and the dut.  It will execute test scripts.
Test script is determined by the configuration of the hardware.

### Test Scripts `tests.yaml`
#### Example
```yaml
<<model_id>>:
  name: <<test_name>>
  model: <<model_id>>   # the same as above
  truth_file: <<path>> # The path to ground truth values
  script:
    - list
    - of
    - commands
```

#### Syntax

- `download` Download data to the test device
- `loop` Run the commands a number of time
- `infer` Run inference For a number of cycles

### Device Configuration `devices.yaml`
The device file defines available devices that are automatically detected by the `DeviceManager`

#### `name`: The name of the device
#### `type`: the device type (`interface` or `power`)
#### `preference`: The relative importance if two are detected.  Higher numbers are higher preference.
#### `usb`: `dict` where the key is `vid` and the value is a `pid` or list of `pid`s
#### `usb_description`: String to match in the USB description

### Running the File
Define --mode (Defaults to accuracy)

### `-e` (energy)
### `-p` (power)
### `-a` (accuracy)

The call sign to run the file in power shell is
```
python main.py --dataset_path=C:\Users\robet\GitHubRepos\energyrunner\datasets --mode=e
```

### Device Under Test Configuration `dut.yml`
Optionally define `baud` and `voltage`

## Open Items
- Add start and stop playback to the script
- fix looping
- write the data out in the original format
