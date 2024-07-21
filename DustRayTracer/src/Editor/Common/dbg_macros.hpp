#pragma once

#ifdef DEBUG
	#define printToConsole(format, ...) printf(format, __VA_ARGS__);
#else
	#define printToConsole
#endif // DEBUG

#ifdef CUSTOM_MACRO_CASING

#endif // CUSTOM_MACRO_CASING

