## remove these first two lines
echo "You need to modify this script before running it"
exit 1 
TF_DIR=/path/to/source/of/tensorflow # replace with your path
git clone https://github.com/tensorflow/tensorflow.git ${TF_DIR}
LOCAL_ARCH=linux_x86_64w  # OR osx_x86_64  OR  osx_arm64_default
pushd $TF_DIR
gmake -f tensorflow/lite/micro/tools/make/Makefile generate_hello_world_mbed_project
popd
mv ${TF_DIR}/tensorflow/lite/micro/tools/make/gen/${LOCAL_ARCH}/prj/hello_world/mbed/* .
mbed config root .
mbed deploy
mkdir api
cp ../../api/internally_implemented.* api
cp ../../main.cpp .
cp -r ../../util . 
rm -rf tensorflow/lite/micro/examples/hello_world


