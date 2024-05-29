#pragma once
/*
* using SI units
*/

float operator "" _km(long double x) { return x * 1000; }
float operator "" _cm(long double x) { return x / 100; }

float operator "" _minutes(long double x) { return x * 60; };
float operator "" _hours(long double x) { return x * 3600; };