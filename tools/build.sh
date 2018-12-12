#!/bin/bash

# argument values: iPhoneOS or iPhoneSimulator

PLATFORM=OS

echo "DEVICE:  $DEVICE"

if [ $DEVICE = "iPhoneSimulator" ]; then
    PLATFORM=SIMULATOR
fi
echo "PLATFORM:  $PLATFORM"

# cd ./third_party
cd ./3rd_party
./build-protobuf-3.1.0.sh $DEVICE
mkdir ../build_$DEVICE
cd ../build_$DEVICE
rm -rf *
cmake .. -DCMAKE_TOOLCHAIN_FILE=../3rd_party/ios-cmake/toolchain/iOS.cmake \
    -DIOS_PLATFORM=$PLATFORM -D3RD_PARTY=1
make -j 4
