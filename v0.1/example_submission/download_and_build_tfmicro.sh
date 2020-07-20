cd $(dirname $0)
TFMICRO_BINARY=libtensorflow-microlite.a
if [ ! -f "$TFMICRO_BINARY" ]; then
  wget https://github.com/tensorflow/tensorflow/archive/master.zip
  unzip -o master.zip
  mv -f tensorflow-master/tensorflow .
  rm -rf tensorflow-master
  make -f tensorflow/lite/micro/tools/make/Makefile microlite -j18
  mv tensorflow/lite/micro/tools/make/gen/linux_x86_64/lib/libtensorflow-microlite.a .
  rm master.zip
fi
