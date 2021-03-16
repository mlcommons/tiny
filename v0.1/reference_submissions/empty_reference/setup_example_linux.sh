if [ "$1" == "clean" ]; then
  rm -rf api
  rm -rf main.cc
  rm -rf util
  rm -rf runner*

else

#cd $(dirname $0)
  cp -r ../../api .
  cp -r ../../main.cc .
  cp -r ../../util .
fi
