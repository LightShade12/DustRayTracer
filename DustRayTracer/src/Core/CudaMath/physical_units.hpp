#pragma once
/*
* using SI units
*/

//float operator "" _km(long double x) { return x * 1000; }
//float operator "" _cm(long double x) { return x / 100; }
//
//float operator "" _minutes(long double x) { return x * 60; };
//float operator "" _hours(long double x) { return x * 3600; };

__constant__ const float TRIANGLE_EPSILON = 0.000001;
__constant__ const float PI = 3.141592;
__constant__ const float HIT_EPSILON = 0.001;
__constant__ const float MAT_MIN_ROUGHNESS = 0.045f;
//TODO: add system wide math constants here that are not std supplied
//-Infinity, PI, EPSILON etc