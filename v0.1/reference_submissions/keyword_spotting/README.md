## Steps to Build and Run

```
TF_DIR=/Users/jeremy/pkgs/tensorflow # replace /Users/jeremy with your path
LOCAL_ARCH=osx_arm64_default # or linux_x86_64w
pushd $TF_DIR
gmake -f tensorflow/lite/micro/tools/make/Makefile generate_hello_world_mbed_project
popd
mv ${TF_DIR}/tensorflow/lite/micro/tools/make/gen/${LOCAL_ARCH}/prj/hello_world/mbed/* .
mbed config root .                                                                     
mbed deploy
cp -r  ../../api . 
cp ../../main.cpp .
cp -r ../../util . 
rm -r tensorflow/lite/micro/examples/hello_world
cp -r ../../vww 
mbed compile --target NUCLEO_L4R5ZI --toolchain GCC_ARM -v
```