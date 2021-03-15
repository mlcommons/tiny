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
    mv -n tensorflow/lite/micro/tools/make/gen/linux_x86_64_default/prj/hello_world/mbed/* ../
    popd
    rm -rf tensorflow-*
    rm -rf tensorflow/lite/micro/examples/hello_world
  fi

  mbed config root .
  mbed deploy .
  cp -r ../../api/internally* api/
  cp -r ../../main.cpp .
  cp -r ../../util .
  cp -r ../../training/visual_wake_words/trained_models/vww .
fi
