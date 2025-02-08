
In this directory, are two projects, `sww_ref_l4r5zi` and `sww_testing_l4r5zi`, both targeted at the NUCLEO L4R5ZI reference board. 

### Test Project
The test project `sww_testing_l4r5zi` is primarily intended for testing individual components of the system, such as running a model on a static input tensor, running feature extraction, and verifying I2S communication.  
This project is designed to take commands over UART (via USB and the ST-LINK connector) directly from the host at 115200 baud. Each command is terminated with a '%' character.
* name -- print out an identifying message
* run_model -- run the NN model. An optional  argument class0, class1, or class2 runs the model on a selected input that is expected to return 0 (WW), 1 (silent), or 2(other)
* extract -- run the feature extractor on the first block of a predefined wav form (test_wav_marvin)
* i2scap -- Captures about 1s of stereo audio over an I2S link
* log -- The I2S capture function can write debug messages to a log. Prints and clears that log.
* help -- Print a help message

To demonstrate an I2S transaction, you will need to have a stereo (2-channel) wav file on the SD card on the interface board.  Assume that file is named `test.wav`.  Connect the I2S ports as described below in "I2S Connections" and connect both boards to the host via USB.  Establish a terminal connection to each of them.  Optionally type `name%` into each terminal to verify that UART communication is working and that you know which device is on which terminal.  Start I2S reception on the DUT with the command `i2scap%`.  Then start I2S transmission on the interface board with the command `play happy_2ch.wav%`.  The length of the capture is defined by the variable `g_i2s_wav_len` in sww_util.c. After capturing that many (currently 8192) samples, it will print the values out in a format that is relatively easy to import into python.  If you want to check the recorded samples against the original file, you'll need to turn on some kind of log capture in your terminal program.  (I use `screen -L /dev/cu.usbmodelXXXX 115200`, but the `XXXX` part changes whenever the device is reconnected.)



### Reference Project
The main reference project `sww_ref_l4r5zi` will be a complete working implementation of the the streaming wakeword detection benchmark.


### I2S Connections 
For both projects, you should connect the I2S port on the DUT to the I2S port on the interface board (STM32H573I-DK). On the interface board, the I2S connection is on CN6, the PMOD connector.  On the DUT, it is on CN9, the lower left connector (with the ST-LINK connector on top) on the top of the board.

| DUT Signal | DUT Pin | Interface Board Signal | Interface Board Pin | 
| ---------  | ------- | ---------------------  | ------------------- |
| `SAI1_FS_A` | CN9.16 |  `SAI1_FS_B`           | CN6.2 | 
| `SAI1_SCK_A`| CN9.18 |  `SAI1_SCK_B`          | CN6.3 | 
| `SAI1_SD_A` | CN9.20 |  `SAI1_SD_B`           | CN6.1 | 

