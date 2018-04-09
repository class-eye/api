include_directories(${CMAKE_CURRENT_LIST_DIR})
set(SRC ${CMAKE_CURRENT_LIST_DIR}/student.cpp
		${CMAKE_CURRENT_LIST_DIR}/pose.cpp
		${CMAKE_CURRENT_LIST_DIR}/rfcn.cpp 	
		${CMAKE_CURRENT_LIST_DIR}/yolov3.cpp
        ${CMAKE_CURRENT_LIST_DIR}/hand.cpp
		${CMAKE_CURRENT_LIST_DIR}/face.cpp
		${CMAKE_CURRENT_LIST_DIR}/facefeature.cpp
        ${CMAKE_CURRENT_LIST_DIR}/jfda.cpp 
		${CMAKE_CURRENT_LIST_DIR}/fs.cpp  
		${CMAKE_CURRENT_LIST_DIR}/Timer.cpp 
		${CMAKE_CURRENT_LIST_DIR}/behavior.cpp  
		${CMAKE_CURRENT_LIST_DIR}/functions.cpp  
		${CMAKE_CURRENT_LIST_DIR}/student/functions.hpp
		${CMAKE_CURRENT_LIST_DIR}/student/rfcn.hpp
		${CMAKE_CURRENT_LIST_DIR}/student/yolov3.hpp
		${CMAKE_CURRENT_LIST_DIR}/student/behavior.hpp
		${CMAKE_CURRENT_LIST_DIR}/student/student.hpp
		${CMAKE_CURRENT_LIST_DIR}/student/fs.hpp
		${CMAKE_CURRENT_LIST_DIR}/student/Timer.hpp
		${CMAKE_CURRENT_LIST_DIR}/student/facefeature.hpp
		${CMAKE_CURRENT_LIST_DIR}/student/pose.hpp
		${CMAKE_CURRENT_LIST_DIR}/student/hand.hpp
		${CMAKE_CURRENT_LIST_DIR}/student/face.hpp
        ${CMAKE_CURRENT_LIST_DIR}/student/jfda.hpp)	
set(STUDENT_INCLUDE ${CMAKE_CURRENT_LIST_DIR})
set(STUDENT_LIBRARY student)
add_library(student STATIC ${SRC})
target_link_libraries(student ${OpenCV_LIBS})