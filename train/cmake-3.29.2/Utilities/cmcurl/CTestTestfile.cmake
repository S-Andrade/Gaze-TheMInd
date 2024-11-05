# CMake generated Testfile for 
# Source directory: /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcurl
# Build directory: /home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcurl
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[curl]=] "curltest" "http://open.cdash.org/user.php")
set_tests_properties([=[curl]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcurl/CMakeLists.txt;1657;add_test;/home/sandra/Gaze/train/cmake-3.29.2/Utilities/cmcurl/CMakeLists.txt;0;")
subdirs("lib")
