# stixel
Stixel c++ code based on gish523's algo.

### Requirement
Opencv4.0

### Compilation
```bash
mkdir build && cd build
cmake ..
make
```

Make sure your are in the `build` folder to run the executables.

### Running
```bash

./stixel [dir] [camera param]

dir should be compatible with Kitti dataset.   
./stixel ../data/ ../camera.xml

