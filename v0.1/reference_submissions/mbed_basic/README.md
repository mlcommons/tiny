# Empty API Example Project for MBED

This example is a basic empty example designed to be build and flashed using
the mbed cli. In order to use this example, run `./setup_mbed.sh`

Connect your mbed board and ensure it is mounted properly as a USB mass storage
device. Somtimes configuring fstab to mount /dev/sd<a-z. to /media is necessary,
followed by running `sudo mount -a`.

to flash, run `mbed compile -f`
