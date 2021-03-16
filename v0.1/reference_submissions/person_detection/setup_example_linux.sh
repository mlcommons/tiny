# define TensorFlow version as git branch/tag/hash
TF_VERSION=v2.3.1

# enable CMSIS-NN
TF_MAKE_TAGS="cmsis-nn"

if [ "$1" == "clean" ]; then
  rm -rf api/internally*
  rm -rf main.cpp
  rm -rf util
  rm -rf vww
  rm -rf tensorflow-master.zip
  rm -rf tensorflow
  rm -rf mbed-os 
  rm -rf $TF_VERSION*
  rm -rf $TF_VERSION.zip
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
    wget https://github.com/tensorflow/tensorflow/archive/$TF_VERSION.zip
    unzip -o $TF_VERSION.zip
    pushd tensorflow-*
    make -f tensorflow/lite/micro/tools/make/Makefile TAGS=$TF_MAKE_TAGS third_party_downloads
    make -f tensorflow/lite/micro/tools/make/Makefile TAGS=$TF_MAKE_TAGS generate_hello_world_mbed_project -j18
    mv -n tensorflow/lite/micro/tools/make/gen/linux_x86_64/prj/hello_world/mbed/* ../
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
