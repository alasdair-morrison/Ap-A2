#include "../include/sklearn_cpp/linear_model/utils.hpp"

int main(){
    //check Dataset class works
    const Dataset test{"/home/joseph/University/AP/Ap-A2/data/concrete.csv"};

    //check R^2 method works
    std::vector<double> y = {3, -0.5, 2, 7};
    std::vector<double> y2 = {2.5, 0.0, 2, 8};
    double r2 = R_squared(y, y2);
    std::cout << "R^2 = " << r2 << std::endl;

    //check accuracy_score method works
    std::vector<double> Y = {1,2,3,4,5};
    std::vector<double> Y2 = {1,2,3,4,0};
    double acc = accuracy_score(Y, Y2);
    std::cout << "accuracy = " << acc << std::endl;
}