if [ "$1" == "clean" ]; then
  rm -rf api
  rm -rf main.cpp
  rm -rf util
  rm -rf vww
  rm -rf tensorflow-master.zip
  rm -rf tensorflow
  rm -rf mbed* 
  rm -rf master*
  rm -rf tensorflow-master
  rm -f CMakeLists.txt
  rm -rf BUILD
  rm -rf third_party
  rm -f LICENSE
  rm -f README_MBED.md
  rm -rf __pycache__

else

#cd $(dirname $0)
  TFMICRO_DIR=tensorflow
  if [ ! -f "$TFMICRO_DIR" ]; then
    wget https://github.com/tensorflow/tensorflow/archive/master.zip
    unzip -o master.zip
    pushd tensorflow-master
    make -f tensorflow/lite/micro/tools/make/Makefile third_party_downloads
    make -f tensorflow/lite/micro/tools/make/Makefile generate_hello_world_mbed_project -j18
    mv tensorflow/lite/micro/tools/make/gen/*/prj/hello_world/mbed/* ../
    popd
    rm -rf tensorflow-master
    rm -rf tensorflow/lite/micro/examples/hello_world
  fi

  mbed new .
  cp ../../api/internally* api/
  cp ../../main.cpp .
  cp -r ../../util .

fi
