//
// RT64
//

#pragma once

// Blue noise tile set, 64 tiles of 64x64 laid out as an 8x8 grid, uploaded as one 512x512
// BGRA8 texture by Application::setup (hle/rt64_application.cpp:387), which takes both the
// pointer and sizeof() of this array.
//
// Declared here and defined in LDR_64_64_64_RGB1.cpp so the megabyte of data is compiled
// once rather than in every translation unit that samples it.

#define LDR_64_64_64_RGB1_WIDTH  512
#define LDR_64_64_64_RGB1_HEIGHT 512

extern const unsigned char LDR_64_64_64_RGB1_BGRA8[LDR_64_64_64_RGB1_WIDTH * LDR_64_64_64_RGB1_HEIGHT * 4];
