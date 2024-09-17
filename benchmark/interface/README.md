# Interface board software

## Features
- Interacts with `runner` python code on the host machine.
- Interacts with the device under test
- Reads `wav` files from the SD cards and plays them back through the headphone jack or the i2c interface.

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
default, `Audio::RawI2SWaveSink` if `HEADPHONE_PLAYBACK` is defined.

#### `IO::FileSystem`
This class reads interacts with the File System on the SD Card

#### `Test::DeviceUnderTest`
This class reads and writes to the DUT

### ThreadX
The application is based on `ThreadX` and `FileX` standard libraries.

The application has two threads.
#### App_ThreadX thread (`Core/Src/app_threadx.c`):
Thread to monitor the CLI port and execute commands

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

## Open Items
- Add a second thread for the `Task::TaskRunner`
- I2C playback does not finish the audio clip (need to send the remainder of the buffer after all the blocks are sent)
- Remove MemoryReader from IO::FileSystem.AsyncOpenFile
