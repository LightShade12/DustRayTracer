#pragma once

#ifdef DEBUG
	#define printToConsole(format, ...) printf(format, __VA_ARGS__);
#else
	#define printToConsole
#endif // DEBUG


