if [ "$1" == "clean" ]; then
  rm -rf api
  rm -rf main.cpp
  rm -rf util
  rm -rf runner*

else

#cd $(dirname $0)
  cp -r ../../api .
  cp -r ../../main.cpp .
  cp -r ../../util .
fi
