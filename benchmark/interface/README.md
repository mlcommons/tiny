# Interface board software

## Features
- Interacts with `runner` python code on the host machine.
- Interacts with the device under test
- Reads `wav` files from the SD cards and plays them back through the headphone jack or the i2c interface.


## Communication
- The interface board communicates with the host over a serial connection.  

## Notes on Running with Macs
- On Mac, you may need to install [picocom](https://github.com/npat-efault/picocom) and run it with the following command: `picocom -b 115200 /dev/tty.usbmodem14301`
## LEDs
- Blue: SD card inserted and running
- Green: Toggles state when each task is run

## Audio Playback
The application can play `wav` files from the SD card.  It sends the audio to the i2c interface (see the README in the
runner for connections).  But can be compiles to playback using the headphone jack (define `HEADPHONE_PLAYBACK`)

## Command Interface
The interface board implements a command interface similar to that of the original Arduino interface board.

Commands sent from the host machine are terminated with `%`

The results are sent back to the host and the results are terminated with `m-ready\r\n`

### `dut` - wrap passthrough commands for the device under test
#### syntax: `dut command to pass to device`

The command is sent to the dut terminated with `%`

Returns the results of the command

### ls - list the files available on the SD card
#### syntax: `ls`

### name - return the name of the control board

### play - play audio

## Implementation

The application code base is generated using STM32CubeMX, HW configuration changes should be done there.

In the generated files, you will find `/* USER CODE BEGIN ... */` and `/* USER CODE END ... */` comments.
Any application code needs to be placed between these comment pairs.

The general approach taken in this implementation, is to minimize the code added to generated files.

The application functionality is implemented in c++ classes that have thin, c-compatible 
wrappers that are called from the generated code if needed.

### Key Classes

#### `CLI::InterfaceMenu` and `CLI::Menu`
These classes are responsible for defining the command set and interacting with the 
To centralize and abstract the interactions with the hardware, the configuration of hardware is defined in 

#### `::ResourceManager`  
This class allows for inversion of control in the target.  Static/singleton resources are defined in this class, and
this class can be used by other classes to use these resources.

Resources managed by this class:
- `Task::TaskRunner`
- `Audio::WaveSink`
- `IO::FileSystem`
- `Test::DeviceUnderTest`

#### `Task::TaskRunner`
This class executes submitted tasks in sequence

#### `Audio::WaveSink` and `Audio::HeadphoneWaveSink` and `Audio::RawI2SWaveSink`
Audio playback destination.  The choice of playback can be chosen at compile time.  `Audio::RawI2SWaveSink` is the
default, `Audio::HeadphoneWaveSink` if `HEADPHONE_PLAYBACK` is defined.
When using I2S output, it appears (please verify) that the software uses SAI B of SAI 1, configured as master with master clock out, with SAI1_SD_B on PF6 (CN3.1, CN6.1), SAI1_MCLK_B on PF7 (CN3.4, CN6.4), SAI1_SCK_B on PF8 (CN3.3, CN6.3), and SAI1_FS_B on PF9 (CN3.2, CN6.2). See UM3143 Table 20, Table 21. These connections assume that the solder bridges SB2-SB9 are in their default configuration.  See UM3143 Table 19.  If you look carefully at connector CN3 (CN6), you will see that pins 1,11 (1,7) are marked on the left end of the connector.  CN6.5, CN6.11 (top and bottom second from the left) are both GND.

#### `IO::FileSystem`
This class reads interacts with the File System on the SD Card

#### `Test::DeviceUnderTest`
This class reads and writes to the DUT

### ThreadX
The application is based on `ThreadX` and `FileX` standard libraries.

The application has two threads.
#### App_ThreadX thread (`Core/Src/app_threadx.c`):
Thread to monitor the CLI port and execute commands.  The main loop in this thread is launched from `tx_app_thread_entry()` in `app_threadx.c`, which calls `CLI_Run()` in `InterfaceMenu.cpp`, which in turn calls `Menu::Run()` in `Menu.cpp`, which repeatedly takes '%'-terminated strings from UART1 and passes them to `HandleCommand()`.  The specific commands are listed in `InterfaceMenu::menu_struct[]` in `InterfaceMenu.cpp`.

### Misc Notes
* The basic unit of the CLI is the `command`, of the type `menu_command_t` which comprises a string and a function pointer, defined in Menu.hpp.
* For debugging, it may be useful to add something like this to `FileX/App/app_filex.c`.

    ```
    #include "usart.h"
    HAL_UART_Transmit(&huart1, (uint8_t *)"test message\r\n", <num_characters>, HAL_MAX_DELAY);
    ```


#### MX_FileX thread (`FileX/App/app_filex.c`):
Thread to monitor the SD card and stand up the filesystem when a card is inserted.

Tasks are processed in this thread when the SD card is inserted.

Currently, this does not support hot-swapping SD cards.  There is an error in the hal that does not handle SD
card removal

#### References
file:///Users/sreckamp/Downloads/Azure_RTOS_ThreadX_User_Guide.pdf
file:///Users/sreckamp/Downloads/Azure_RTOS_FileX_User_Guide%20(1).pdf
https://www.st.com/resource/en/user_manual/um2298-stm32cube-bsp-drivers-development-guidelines-stmicroelectronics.pdf
file:///Users/sreckamp/Downloads/en.mb1677-h573i-c02-schematic.pdf
https://www.st.com/resource/en/user_manual/um3143-discovery-kit-with-stm32h573ii-mcu-stmicroelectronics.pdf
https://www.st.com/resource/en/reference_manual/rm0481-stm32h563h573-and-stm32h562-armbased-32bit-mcus-stmicroelectronics.pdf
https://app.trinetexpense.com/users/login

## Troubleshooting notes
- This is specifically for the 'Nonetype' error when trying to run the test in the runner folder
- Make sure you are using the updated STM32 compiler code. In the UART file specifically the baud rate should be 115200
- Access the device port using Device manager and use PUTTY or a similar software to connect to the device
    - Keep in mind if the device is configured properly you only need the Interface board plugged into the computer
- Send commands such as dut profile%, and dut name%
- dut name% should return
  ```serial
  m-dut-passthrough(name)
  m-ready
  [dut]: m-name-dut-[unspecified]
  [dut]: m-ready
  ```
- dut profile% should return
  ```serial
  m-dut-passthrough(profile)
  m-ready
  [dut]: m-profile-[ULPMark for tinyML Firmware V0.0.1]
  [dut]: m-model-[kws01]
  [dut]: m-ready
  ```
- If you have to send the commands multiple times to get the correct results you will need to make a change to the 'device_under_test.py' file

- The code changes will involve adding a counter loop so the program tries to fetch the data multiple times before moving on. Heres an example for the get name function
```python
def _get_name(self):
    name_retrieved = False
    for attempt in range(5):  # This line allows for us to retry fetching the name
        print(f"Attempting to retrieve name, attempt {attempt + 1}")
        for l in self._port.send_command("name"):
            match = re.match(r'^m-(name)-dut-\[([^]]+)]$', l)
            if match:
                self.__setattr__(f"_{match.group(1)}", match.group(2))
                name_retrieved = True
        if name_retrieved:
            print(f"Device name retrieved: {self._name}")  # Print name for troubleshooting purposes
            break
        print(f"WARNING: Attempt {attempt + 1} failed, retrying...")
        time.sleep(1)  # Wait for 1 second before retrying
```

## Open Items
- Add a second thread for the `Task::TaskRunner`
- I2C playback does not finish the audio clip (need to send the remainder of the buffer after all the blocks are sent)
- Remove MemoryReader from IO::FileSystem.AsyncOpenFile
