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

Please run the following in a separate directory:
```
export TFMICRO_DIR=<separate directory location>
wget https://github.com/tensorflow/tensorflow/archive/master.zip
unzip -o master.zip
pushd tensorflow-master
make -f tensorflow/lite/micro/tools/make/Makefile third_party_downloads
make -f tensorflow/lite/micro/tools/make/Makefile generate_hello_world_mbed_project -j18
popd
```
Find the location of the generated mbed hello_world project. On Linux it should
be:
tensorflow-master/tensorflow/lite/micro/tools/make/gen/linux_x86_64_default/prj/hello_world/mbed
On Windows/mac the directory under make/gen will be different.

```
cp -r <mbed project directory>/* <tiny
root>/tiny/v0.1/reference_submissions/person_detecton/

cd <tiny root>/tiny/v0.1/reference_submissions/person_detection/
rm -rf tensorflow/lite/micro/examples/hello_world
mbed config root .
mbed deploy
cp -r ../../api .
cp -r ../../main.cpp .
cp -r ../../util .
cp -r ../../training/visual_wake_words/trained_models/vww .
```
