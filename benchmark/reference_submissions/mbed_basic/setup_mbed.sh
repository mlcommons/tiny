# Copy API files since all source files must be within an mbed project.
cp -r ../../api .
cp ../../main.cpp .

# Create mbed project and checkout to restore the overwritten main.cpp.
mbed-tools new .
git checkout .
