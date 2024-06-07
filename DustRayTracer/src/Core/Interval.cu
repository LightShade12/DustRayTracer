#include "Interval.cuh"

const Interval Interval::empty = Interval(FLT_MAX, -FLT_MAX);//TODO: lookup why this is so
const Interval Interval::universe = Interval(-FLT_MAX, FLT_MAX);