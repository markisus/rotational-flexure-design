// ticktock.cc
//
// Supplies implementation of the class TickTock to measure wall-time
//
// Ramses van Zon, 2015-2016
//

#include "ticktock.h"
#include <iostream>
#include <iomanip>


#include <chrono>
// define a reference point:
const static std::chrono::steady_clock::time_point reference_time = std::chrono::steady_clock::now();


static void print(const char* prefix, const double dt)
{
    // helper function to print elapsed time	
    if (prefix!=0)
        std::cout << prefix << "\t" << std::setprecision(4) << dt << " sec\n";
    else
        std::cout << "    \t" << std::setprecision(4) << dt << " sec\n";
}

static double elapsed_time()
{
    std::chrono::steady_clock::time_point this_time = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(this_time - reference_time).count();
    return dt;
}

void TickTock::tick()
{
    // start measuring time
    tick_ = elapsed_time();
}

void TickTock::tock(const char* prefix) const
{
    // done measuring time; print elapsed seconds
    print(prefix, silent_tock()); 
}

double TickTock::silent_tock() const
{
    // done measuring time; give elapsed seconds
    double tock = elapsed_time();
    double dt = tock-tick_;
    return dt;
}
