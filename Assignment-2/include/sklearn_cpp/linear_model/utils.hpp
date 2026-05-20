#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include<numeric>
#include <cmath>

class Dataset
{
private:
    std::vector<std::vector<double>> labels;
    std::vector<double> data;

    //identifies numbers, used to detect header in csv file.
    bool isNumber(const std::string& s){
        std::stringstream ss(s);
        double d;
        char c;
        return (ss >> d) && !(ss >> c);
    }

    void populate(const std::string& filename)
    {
        std::ifstream file(filename);

        if (!file.is_open())
        {
            throw std::runtime_error("Could not open file");
        }

        std::string line;
        bool firstLine = true;

        while (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::string value;

            std::vector<std::string> raw;
            std::vector<double> row;

            while (std::getline(ss, value, ','))
            {
                raw.push_back(value);
            }

            // detect header
            if (firstLine){
                bool header = false;
                for (const auto& v : raw){
                    if (!isNumber(v)){    
                        header = true;
                        break;
                    }
                }

                firstLine = false;

                if (header)
                    continue;
            }

            std::vector<double> features;

            for (size_t i = 0; i < raw.size(); ++i){
            double num = std::stod(raw[i]);

            //pushes the last piece of data into data vector
            if (i == raw.size() - 1){
                data.push_back(num);
            }

            //pushes other numbers into vector
            else{
                features.push_back(num);
            }
            }
            //pushes vector of features into vec of vecs
            labels.push_back(features);
        }
    }

public:
    // Constructor for file name
    Dataset(const std::string& filename)
    {
        populate(filename);
        std::cout<<"Dataset constructed using filename \n";
    }

    // Constructor for vector of vectors
    Dataset(const std::vector<std::vector<double>>& labels, const std::vector<double>& data):
    labels{labels},
    data{data}
    {std::cout<<"Dataset constructed using vectors \n";}

    //Allows user to use data
    const std::vector<double>& getData() const{ return data; }
    const std::vector<std::vector<double>>& getLabels() const{ return labels; }
    //Allows user to print data
    void print() const{
    for (const auto& row : labels){
        for (const auto& value : row){
            std::cout << value << " ";
        }
        std::cout << "\n";
    }
    }   

    void printResult() const{
        for (const auto& n : data){
            std::cout << n << "\n";
        }
    }
};

void VectorSizeChecker(const std::vector<double>& v1, const std::vector<double>& v2){
    if(v1.size() != v2.size()){
    throw std::invalid_argument("Vectors must be same size");
    }
}

//R squared calculator
double const R_squared(const std::vector<double>& Y_actual, const std::vector<double>& Y_predicted){
    
    VectorSizeChecker(Y_actual, Y_predicted);
    
    //Get the mean value of the actual value, used to calculate S_total
    double Y_mean{std::accumulate(Y_actual.begin(),Y_actual.end(),0)/static_cast<double>(Y_actual.size())};

    double S_residual{0};
    double S_total{0};

    //Calculate the S_residual and S_total terms
    for(int i{0}; i < Y_actual.size(); i++){
        S_residual += std::pow(Y_actual[i] - Y_predicted[i],2);
        S_total += std::pow(Y_actual[i] - Y_mean,2);
    }

    //If all Y values are the same, avoid division by zero.
    if (S_total == 0.0){return 0.0;}

    //calculate R squared term
    double R_squared{1 - S_residual/S_total};

    return R_squared;
}

//Accuracy checker for logistic regression accuracy analysis
double const accuracy_score(const std::vector<double>& Y_actual, const std::vector<double>& Y_predicted){

    VectorSizeChecker(Y_actual, Y_predicted);

    int correctPrediction{0};

    for(int i{0}; i < Y_actual.size(); i++){
        if(Y_actual[i] == Y_predicted[i]){
            correctPrediction += 1;
        }
    }

    double accuracy{correctPrediction/static_cast<double>(Y_actual.size())};
    
    return accuracy;
}