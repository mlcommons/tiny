[[_TOC_]]
## Performance/Accuracy Metrics
### Device Configurations
Connect just the device (L4R5ZI) to the computer
### Step 1: Update the Baud Rate
Ensure the DUT board has a file attached and note its **BAUD rate**.  
In `device_under_test.py`, update **line 9** to reflect this baud rate.

### Step 2: Run a Test Trial
Each test trial will be stored in a log file in the local folder.


#### Run the Test in PowerShell:
```powershell
python main.py --dataset_path=C:\Your\Dataset\Path --mode=p
```

Mode for accuracy is a, mode for performance is p. This needs to be lowercase and we default to accuracy

** As of writing this the Performance Metric has not been implimented **

## Energy Test Connections

### Power Board (LPM01A)
![LPM01A Wiring](img/LPM01A.jpg)

### Interface Board (STM32H573I-DK)
![STM32H573I-DK Top Wiring](img/STM32H573I-DK-Top.png)
![STM32H573I-DK Bottom Wiring](img/STM32H573I-DK-Bottom.png)

### Device Under Test (L4R5ZI)
![DUT Wiring](img/L4R5ZI.png)


---

## Test Runner

The test runner connects to the **interface board**, **power board**, and **DUT**. It executes test scripts determined by the hardware configuration.

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

- `download` - Download data to the test device
- `loop` - Run the commands a specified number of times
- `infer` - Run inference for a specified number of cycles

---

### Device Configuration `devices.yaml`
The device file defines available devices that are automatically detected by the `DeviceManager`.

#### Parameters:
- **`name`**: The name of the device
- **`type`**: The device type (`interface` or `power`)
- **`preference`**: The relative importance if two devices are detected. Higher numbers indicate higher preference.
- **`usb`**: `dict` where the key is `vid` and the value is a `pid` or a list of `pid`s.
- **`usb_description`**: A string used to match the USB description.

---

### Device Under Test Configuration `dut.yml`
Optionally define:
- **`baud`**: The baud rate
- **`voltage`**: The operating voltage

---

## Running the File
Follow these steps to run tests on the device.

### Step 1: Update the Baud Rate
Ensure the DUT board has a file attached and note its **BAUD rate**.  
In `device_under_test.py`, update **line 9** to reflect this baud rate.

### Step 2: Configure the Interface Board
If your **interface board** is set up correctly (reference the `interface` folder), you should have **two separate baud rates**:
- One for the DUT board connection.
- One for the computer connection.

In `io_manager.py`, update **line 6** to reflect the **computer → interface** baud rate.

### Step 3: Verify Wiring
Double-check the wiring as per the beginning of the README.  
**Note**: No I2C transmission will be used.

For **Power Tests**, follow the energy setup images **exactly**.

For **Accuracy/Energy Tests**:
- TX and RX wiring should be configured identically.
- Ground the interface board to the DUT board.
- Connect the **3.3V** ports of both boards.

### Step 4: Run a Test Trial
Each test trial will be stored in a log file in the local folder.


#### Run the Test in PowerShell:
```powershell
python main.py --dataset_path=C:\Your\Dataset\Path --mode=e
```

You must define mode as e and if for some reason the PowerBoard is not detecting it will return a runtime error
## Troubleshooting Section

If you encounter errors while running the test, refer to the guide below.

### **Error: 'NoneType' Appears**
This typically indicates a **UART transmission error**.

#### **Steps to Resolve:**
1. **Check your wiring** – Ensure all connections are correct.
2. **Run PowerShell as Administrator** –  
   - On Windows, this error is often caused by restricted access to Serial ports.
   - Open **PowerShell** as an **administrator** and retry the test.
3. **Ensure no other application is using the device ports** –  
   - If another process has locked the ports, the above fixes will not work.

---

## FX Media Open Not Working
If `FX_FAT_READ_ERROR` is triggered, there may be **issues with the SD card or its formatting**.

### **Formatting the SD Card (Windows)**
1. Open **Command Prompt** and type:
   ```powershell
   diskpart
   ```
2. This will queue up another terminal for formatting a sd device the run the following commands
   ```powershell
    list disk
    select disk _  // Ensure you select the correct disk
    clean
    create partition primary
    format fs=fat32 quick 
    assign
    exit
   ```

### Baud Rate for Interface board:
Located in file /application/user/core/usart.c
   
