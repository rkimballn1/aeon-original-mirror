set(SRC
    iterator.cpp
)

add_executable(speech_iterator ${SRC})
target_link_libraries(speech_iterator aeon pthread sox ${CURL_LIBRARIES} opencv_imgproc opencv_highgui opencv_core ${PYTHON_LIBRARIES})
add_dependencies(speech_iterator aeon)
