#cd $(dirname $0)
TFMICRO_DIR=tensorflow
if [ ! -f "$TFMICRO_DIR" ]; then
  wget https://github.com/tensorflow/tensorflow/archive/master.zip
  unzip -o master.zip
  pushd tensorflow-master
  make -f tensorflow/lite/micro/tools/make/Makefile generate_hello_world_mbed_project -j18
  mkdir /tmp/mbed_tmp
  mv tensorflow/lite/micro/tools/make/gen/linux_x86_64/prj/hello_world/mbed/* /tmp/mbed_tmp
  popd
  mv /tmp/mbed_tmp/* .
  rm tensorflow/lite/micro/examples/hello_world -r 
fi

mbed-tools new .
cp ../../api . -r
cp ../../main.cpp .
cp ../../util . -r
cp ../../vww . -r
