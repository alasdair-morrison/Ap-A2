#include "../include/sklearn_cpp/linear_model/utils.hpp"

int main(){
    const Dataset test{"/home/joseph/University/AP/Ap-A2/data/concrete.csv"};

    // std::vector<double> y = {3, -0.5, 2, 7};
    // std::vector<double> y2 = {2.5, 0.0, 2, 8};

    std::vector<double> y{1.2,2,3};
    std::vector<double> y2{1,2,3};

    double r2 = R_squared(y, y2);

    std::cout << "R^2 = " << r2 << std::endl;
}