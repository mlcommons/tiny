#!/bin/sh

if [ ! -f "$1" ]; then
    echo "File $1 not found.  Exiting"
    exit 1
fi

echo "Converting file $1 to file kws_model_data.cc"

xxd -i ${1} > kws_model_data.cc

# the variable name in the xxd output is the file name with [./] mapped to '_'
FNAME_CONV=`echo $1 | tr '/.' '_'`

sed -i .bak '1 s/^/#include "kws_model_data.h"\n\n/'  kws_model_data.cc
sed -i .bak 's/unsigned char/const unsigned char/' kws_model_data.cc
sed -i .bak "s/$FNAME_CONV/g_kws_model_data/" kws_model_data.cc
sed -i .bak 's/unsigned int/const unsigned int/' kws_model_data.cc

rm  -f kws_model_data.cc.bak