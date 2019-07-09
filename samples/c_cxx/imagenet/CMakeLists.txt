# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(FS_SOURCES local_filesystem.h sync_api.h controller.h controller.cc)
if(WIN32)
  LIST(APPEND FS_SOURCES local_filesystem_win.cc sync_api_win.cc)
else()
  LIST(APPEND FS_SOURCES local_filesystem_posix.cc sync_api_posix.cc)
endif()
add_library(slim_fs_lib ${FS_SOURCES})
if(WIN32)
  target_compile_definitions(slim_fs_lib PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
endif()

add_executable(image_classifier main.cc jpeg_handle.cc jpeg_handle.h jpeg_mem.cc jpeg_mem.h
        async_ring_buffer.h image_loader.cc image_loader.h cached_interpolation.h  single_consumer.h)
target_include_directories(image_classifier PRIVATE ${PROJECT_SOURCE_DIR}/include ${JPEG_INCLUDE_DIR})
if(WIN32)
  target_compile_definitions(image_classifier PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
endif()
target_link_libraries(image_classifier PRIVATE onnxruntime slim_fs_lib ${JPEG_LIBRARIES})

if(PNG_FOUND)
	add_executable(resize_image resize_image_cmd.cc jpeg_handle.cc jpeg_handle.h jpeg_mem.cc jpeg_mem.h
			image_loader.cc image_loader.h cached_interpolation.h)
	if(WIN32)
	  target_compile_definitions(resize_image PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
	endif()
	target_include_directories(resize_image PRIVATE ${JPEG_INCLUDE_DIR})
	target_link_libraries(resize_image PRIVATE slim_fs_lib ${JPEG_LIBRARIES})
endif()


