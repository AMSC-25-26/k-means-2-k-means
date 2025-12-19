# Project notes

## TODO list:
* [ ] Implement parallel K-means++
* [ ] Add clustering evaluation metrics (e.g., silhouette score)
* [ ] Implement different distance metrics (e.g., Manhattan, Cosine)
* [ ] Add an interesting dataset to work on
* [ ] Add some kind of data visualization
* [x] Add cluster datasets

## Project structure

C++ source of the project is in `src/` folder, and is built through CMake. The main executable is `kmeans`, which can be built by running the following commands in the project root directory:

```bash
mkdir build
cd build
cmake ..
make
```

The built executable can be run from the `build/` directory.

CLion configuration files are included in the project for ease of development. Some of them are run configurations which can be used to test the project against common datasets, like `iris` and `wine`.

## Sources

Some useful materials for K-means clustering can be found at the following sources:
* https://cse.buffalo.edu/faculty/miller/Courses/CSE633/Chandramohan-Fall-2012-CSE633.pdf

