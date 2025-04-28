# cpp_numerical_optimizers
This project implements a line optimization simulation using the Ceres Solver and Eigen libraries.

## Project Structure
```
cpp_numerical_optimizers
├── src
│   └── LineOptimizationSim.cpp      # Main implementation of the line optimization simulation
├── include
│   └── CMakeLists.txt                # CMake configuration for include directory (if needed)
├── CMakeLists.txt                    # Main CMake configuration file
└── README.md                          # Documentation for building and running the simulation
```

## Building the Project
To build the project, follow these steps:

1. Navigate to the project directory:
   ```
   cd cpp_numerical_optimizers
   ```

2. Create a build directory and navigate into it:
   ```
   mkdir build
   cd build
   ```

3. Run CMake to configure the project:
   ```
   cmake ..
   ```

4. Compile the project:
   ```
   make
   ```

## Running the Simulation
After building the project, you can run the simulation with the following command:
```
./line_sim
```

## Dependencies
This project requires the following libraries:
- Ceres Solver
- Eigen

Make sure to install these libraries before building the project.