#include <vector>
#include "gtest/gtest.h"
#include "polyscope/polyscope.h"

int main(int argc, char** argv)
{
    polyscope::init();
    polyscope::show();
    return 0;
}