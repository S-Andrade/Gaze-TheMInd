# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/sandra/Gaze/train/cmake-3.29.2/Bootstrap.cmk/cmake

# The command to remove a file.
RM = /home/sandra/Gaze/train/cmake-3.29.2/Bootstrap.cmk/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sandra/Gaze/train/cmake-3.29.2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sandra/Gaze/train/cmake-3.29.2

# Include any dependencies generated for this target.
include Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/compiler_depend.make

# Include the progress variables for this target.
include Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/progress.make

# Include the compile flags for this target's objects.
include Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/flags.make

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/content_stream.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/flags.make
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/content_stream.cpp.o: Utilities/cmcppdap/src/content_stream.cpp
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/content_stream.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/sandra/Gaze/train/cmake-3.29.2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/content_stream.cpp.o"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/content_stream.cpp.o -MF CMakeFiles/cmcppdap.dir/src/content_stream.cpp.o.d -o CMakeFiles/cmcppdap.dir/src/content_stream.cpp.o -c /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/content_stream.cpp

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/content_stream.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cmcppdap.dir/src/content_stream.cpp.i"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/content_stream.cpp > CMakeFiles/cmcppdap.dir/src/content_stream.cpp.i

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/content_stream.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cmcppdap.dir/src/content_stream.cpp.s"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/content_stream.cpp -o CMakeFiles/cmcppdap.dir/src/content_stream.cpp.s

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/io.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/flags.make
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/io.cpp.o: Utilities/cmcppdap/src/io.cpp
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/io.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/sandra/Gaze/train/cmake-3.29.2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/io.cpp.o"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/io.cpp.o -MF CMakeFiles/cmcppdap.dir/src/io.cpp.o.d -o CMakeFiles/cmcppdap.dir/src/io.cpp.o -c /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/io.cpp

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/io.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cmcppdap.dir/src/io.cpp.i"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/io.cpp > CMakeFiles/cmcppdap.dir/src/io.cpp.i

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/io.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cmcppdap.dir/src/io.cpp.s"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/io.cpp -o CMakeFiles/cmcppdap.dir/src/io.cpp.s

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/jsoncpp_json_serializer.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/flags.make
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/jsoncpp_json_serializer.cpp.o: Utilities/cmcppdap/src/jsoncpp_json_serializer.cpp
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/jsoncpp_json_serializer.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/sandra/Gaze/train/cmake-3.29.2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/jsoncpp_json_serializer.cpp.o"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/jsoncpp_json_serializer.cpp.o -MF CMakeFiles/cmcppdap.dir/src/jsoncpp_json_serializer.cpp.o.d -o CMakeFiles/cmcppdap.dir/src/jsoncpp_json_serializer.cpp.o -c /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/jsoncpp_json_serializer.cpp

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/jsoncpp_json_serializer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cmcppdap.dir/src/jsoncpp_json_serializer.cpp.i"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/jsoncpp_json_serializer.cpp > CMakeFiles/cmcppdap.dir/src/jsoncpp_json_serializer.cpp.i

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/jsoncpp_json_serializer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cmcppdap.dir/src/jsoncpp_json_serializer.cpp.s"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/jsoncpp_json_serializer.cpp -o CMakeFiles/cmcppdap.dir/src/jsoncpp_json_serializer.cpp.s

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/network.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/flags.make
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/network.cpp.o: Utilities/cmcppdap/src/network.cpp
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/network.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/sandra/Gaze/train/cmake-3.29.2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/network.cpp.o"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/network.cpp.o -MF CMakeFiles/cmcppdap.dir/src/network.cpp.o.d -o CMakeFiles/cmcppdap.dir/src/network.cpp.o -c /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/network.cpp

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/network.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cmcppdap.dir/src/network.cpp.i"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/network.cpp > CMakeFiles/cmcppdap.dir/src/network.cpp.i

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/network.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cmcppdap.dir/src/network.cpp.s"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/network.cpp -o CMakeFiles/cmcppdap.dir/src/network.cpp.s

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/null_json_serializer.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/flags.make
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/null_json_serializer.cpp.o: Utilities/cmcppdap/src/null_json_serializer.cpp
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/null_json_serializer.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/sandra/Gaze/train/cmake-3.29.2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/null_json_serializer.cpp.o"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/null_json_serializer.cpp.o -MF CMakeFiles/cmcppdap.dir/src/null_json_serializer.cpp.o.d -o CMakeFiles/cmcppdap.dir/src/null_json_serializer.cpp.o -c /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/null_json_serializer.cpp

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/null_json_serializer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cmcppdap.dir/src/null_json_serializer.cpp.i"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/null_json_serializer.cpp > CMakeFiles/cmcppdap.dir/src/null_json_serializer.cpp.i

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/null_json_serializer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cmcppdap.dir/src/null_json_serializer.cpp.s"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/null_json_serializer.cpp -o CMakeFiles/cmcppdap.dir/src/null_json_serializer.cpp.s

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_events.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/flags.make
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_events.cpp.o: Utilities/cmcppdap/src/protocol_events.cpp
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_events.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/sandra/Gaze/train/cmake-3.29.2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_events.cpp.o"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_events.cpp.o -MF CMakeFiles/cmcppdap.dir/src/protocol_events.cpp.o.d -o CMakeFiles/cmcppdap.dir/src/protocol_events.cpp.o -c /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/protocol_events.cpp

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_events.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cmcppdap.dir/src/protocol_events.cpp.i"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/protocol_events.cpp > CMakeFiles/cmcppdap.dir/src/protocol_events.cpp.i

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_events.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cmcppdap.dir/src/protocol_events.cpp.s"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/protocol_events.cpp -o CMakeFiles/cmcppdap.dir/src/protocol_events.cpp.s

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_requests.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/flags.make
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_requests.cpp.o: Utilities/cmcppdap/src/protocol_requests.cpp
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_requests.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/sandra/Gaze/train/cmake-3.29.2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_requests.cpp.o"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_requests.cpp.o -MF CMakeFiles/cmcppdap.dir/src/protocol_requests.cpp.o.d -o CMakeFiles/cmcppdap.dir/src/protocol_requests.cpp.o -c /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/protocol_requests.cpp

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_requests.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cmcppdap.dir/src/protocol_requests.cpp.i"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/protocol_requests.cpp > CMakeFiles/cmcppdap.dir/src/protocol_requests.cpp.i

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_requests.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cmcppdap.dir/src/protocol_requests.cpp.s"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/protocol_requests.cpp -o CMakeFiles/cmcppdap.dir/src/protocol_requests.cpp.s

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_response.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/flags.make
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_response.cpp.o: Utilities/cmcppdap/src/protocol_response.cpp
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_response.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/sandra/Gaze/train/cmake-3.29.2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_response.cpp.o"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_response.cpp.o -MF CMakeFiles/cmcppdap.dir/src/protocol_response.cpp.o.d -o CMakeFiles/cmcppdap.dir/src/protocol_response.cpp.o -c /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/protocol_response.cpp

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_response.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cmcppdap.dir/src/protocol_response.cpp.i"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/protocol_response.cpp > CMakeFiles/cmcppdap.dir/src/protocol_response.cpp.i

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_response.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cmcppdap.dir/src/protocol_response.cpp.s"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/protocol_response.cpp -o CMakeFiles/cmcppdap.dir/src/protocol_response.cpp.s

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_types.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/flags.make
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_types.cpp.o: Utilities/cmcppdap/src/protocol_types.cpp
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_types.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/sandra/Gaze/train/cmake-3.29.2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_types.cpp.o"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_types.cpp.o -MF CMakeFiles/cmcppdap.dir/src/protocol_types.cpp.o.d -o CMakeFiles/cmcppdap.dir/src/protocol_types.cpp.o -c /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/protocol_types.cpp

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_types.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cmcppdap.dir/src/protocol_types.cpp.i"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/protocol_types.cpp > CMakeFiles/cmcppdap.dir/src/protocol_types.cpp.i

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_types.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cmcppdap.dir/src/protocol_types.cpp.s"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/protocol_types.cpp -o CMakeFiles/cmcppdap.dir/src/protocol_types.cpp.s

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/session.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/flags.make
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/session.cpp.o: Utilities/cmcppdap/src/session.cpp
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/session.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/sandra/Gaze/train/cmake-3.29.2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/session.cpp.o"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/session.cpp.o -MF CMakeFiles/cmcppdap.dir/src/session.cpp.o.d -o CMakeFiles/cmcppdap.dir/src/session.cpp.o -c /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/session.cpp

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/session.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cmcppdap.dir/src/session.cpp.i"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/session.cpp > CMakeFiles/cmcppdap.dir/src/session.cpp.i

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/session.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cmcppdap.dir/src/session.cpp.s"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/session.cpp -o CMakeFiles/cmcppdap.dir/src/session.cpp.s

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/socket.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/flags.make
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/socket.cpp.o: Utilities/cmcppdap/src/socket.cpp
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/socket.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/sandra/Gaze/train/cmake-3.29.2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/socket.cpp.o"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/socket.cpp.o -MF CMakeFiles/cmcppdap.dir/src/socket.cpp.o.d -o CMakeFiles/cmcppdap.dir/src/socket.cpp.o -c /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/socket.cpp

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/socket.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cmcppdap.dir/src/socket.cpp.i"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/socket.cpp > CMakeFiles/cmcppdap.dir/src/socket.cpp.i

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/socket.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cmcppdap.dir/src/socket.cpp.s"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/socket.cpp -o CMakeFiles/cmcppdap.dir/src/socket.cpp.s

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/typeinfo.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/flags.make
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/typeinfo.cpp.o: Utilities/cmcppdap/src/typeinfo.cpp
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/typeinfo.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/sandra/Gaze/train/cmake-3.29.2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/typeinfo.cpp.o"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/typeinfo.cpp.o -MF CMakeFiles/cmcppdap.dir/src/typeinfo.cpp.o.d -o CMakeFiles/cmcppdap.dir/src/typeinfo.cpp.o -c /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/typeinfo.cpp

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/typeinfo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cmcppdap.dir/src/typeinfo.cpp.i"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/typeinfo.cpp > CMakeFiles/cmcppdap.dir/src/typeinfo.cpp.i

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/typeinfo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cmcppdap.dir/src/typeinfo.cpp.s"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/typeinfo.cpp -o CMakeFiles/cmcppdap.dir/src/typeinfo.cpp.s

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/typeof.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/flags.make
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/typeof.cpp.o: Utilities/cmcppdap/src/typeof.cpp
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/typeof.cpp.o: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/sandra/Gaze/train/cmake-3.29.2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/typeof.cpp.o"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/typeof.cpp.o -MF CMakeFiles/cmcppdap.dir/src/typeof.cpp.o.d -o CMakeFiles/cmcppdap.dir/src/typeof.cpp.o -c /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/typeof.cpp

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/typeof.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cmcppdap.dir/src/typeof.cpp.i"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/typeof.cpp > CMakeFiles/cmcppdap.dir/src/typeof.cpp.i

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/typeof.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cmcppdap.dir/src/typeof.cpp.s"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/src/typeof.cpp -o CMakeFiles/cmcppdap.dir/src/typeof.cpp.s

# Object files for target cmcppdap
cmcppdap_OBJECTS = \
"CMakeFiles/cmcppdap.dir/src/content_stream.cpp.o" \
"CMakeFiles/cmcppdap.dir/src/io.cpp.o" \
"CMakeFiles/cmcppdap.dir/src/jsoncpp_json_serializer.cpp.o" \
"CMakeFiles/cmcppdap.dir/src/network.cpp.o" \
"CMakeFiles/cmcppdap.dir/src/null_json_serializer.cpp.o" \
"CMakeFiles/cmcppdap.dir/src/protocol_events.cpp.o" \
"CMakeFiles/cmcppdap.dir/src/protocol_requests.cpp.o" \
"CMakeFiles/cmcppdap.dir/src/protocol_response.cpp.o" \
"CMakeFiles/cmcppdap.dir/src/protocol_types.cpp.o" \
"CMakeFiles/cmcppdap.dir/src/session.cpp.o" \
"CMakeFiles/cmcppdap.dir/src/socket.cpp.o" \
"CMakeFiles/cmcppdap.dir/src/typeinfo.cpp.o" \
"CMakeFiles/cmcppdap.dir/src/typeof.cpp.o"

# External object files for target cmcppdap
cmcppdap_EXTERNAL_OBJECTS =

Utilities/cmcppdap/libcmcppdap.a: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/content_stream.cpp.o
Utilities/cmcppdap/libcmcppdap.a: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/io.cpp.o
Utilities/cmcppdap/libcmcppdap.a: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/jsoncpp_json_serializer.cpp.o
Utilities/cmcppdap/libcmcppdap.a: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/network.cpp.o
Utilities/cmcppdap/libcmcppdap.a: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/null_json_serializer.cpp.o
Utilities/cmcppdap/libcmcppdap.a: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_events.cpp.o
Utilities/cmcppdap/libcmcppdap.a: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_requests.cpp.o
Utilities/cmcppdap/libcmcppdap.a: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_response.cpp.o
Utilities/cmcppdap/libcmcppdap.a: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/protocol_types.cpp.o
Utilities/cmcppdap/libcmcppdap.a: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/session.cpp.o
Utilities/cmcppdap/libcmcppdap.a: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/socket.cpp.o
Utilities/cmcppdap/libcmcppdap.a: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/typeinfo.cpp.o
Utilities/cmcppdap/libcmcppdap.a: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/src/typeof.cpp.o
Utilities/cmcppdap/libcmcppdap.a: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/build.make
Utilities/cmcppdap/libcmcppdap.a: Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/sandra/Gaze/train/cmake-3.29.2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Linking CXX static library libcmcppdap.a"
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && $(CMAKE_COMMAND) -P CMakeFiles/cmcppdap.dir/cmake_clean_target.cmake
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cmcppdap.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/build: Utilities/cmcppdap/libcmcppdap.a
.PHONY : Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/build

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/clean:
	cd /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap && $(CMAKE_COMMAND) -P CMakeFiles/cmcppdap.dir/cmake_clean.cmake
.PHONY : Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/clean

Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/depend:
	cd /home/sandra/Gaze/train/cmake-3.29.2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sandra/Gaze/train/cmake-3.29.2 /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap /home/sandra/Gaze/train/cmake-3.29.2 /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : Utilities/cmcppdap/CMakeFiles/cmcppdap.dir/depend
