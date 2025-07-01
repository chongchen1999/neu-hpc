cmake_minimum_required(VERSION 3.10)
project(Word2Vec)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add compiler flags for optimization and debugging
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Add executable
add_executable(word2vec main.cpp)

# If you want to add more source files later, you can do:
# add_executable(word2vec 
#     main.cpp
#     matrix.cpp
#     word2vec.cpp
# )

# Optional: Add include directories if you have header files
# target_include_directories(word2vec PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Optional: Link libraries if needed (e.g., for plotting)
# find_package(Threads REQUIRED)
# target_link_libraries(word2vec Threads::Threads)