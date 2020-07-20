CXXFLAGS += -DTF_LITE_STATIC_MEMORY -DNDEBUG -O3 -DTF_LITE_DISABLE_X86_NEON -Iexample_submission -Iexample_submission/third_party/gemmlowp -Iexample_submission/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include -Iexample_submission/third_party/ruy
CCFLAGS +=  -DTF_LITE_STATIC_MEMORY -DNDEBUG -O3 -DTF_LITE_DISABLE_X86_NEON -Iexample_submission -Iexample_submission/third_party/gemmlowp -Iexample_submission/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include/ -Iexample_submission/third_party/ruy

DOWNLOADS += \
example_submission/libtensorflow-microlite.a \
example_submission/tensorflow

LDFLAGS += -Lexample_submission -ltensorflow-microlite

SRCS += \
example_submission/model.cc \
example_submission/timers.cc \
example_submission/logging.cc

DOWNLOAD_DEPS += example_submission/download_and_build_tfmicro.sh

