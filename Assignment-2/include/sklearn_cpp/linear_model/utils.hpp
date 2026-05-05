#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>

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

        //pushes vector of features into vec of vecs
        labels.push_back(features);
        }
    }
}


public:
    // Constructor for file name
    Dataset(const std::string& filename)
    {
        populate(filename);
    }

    // Constructor for vector of vectors
    Dataset(const std::vector<std::vector<double>>& labels, const std::vector<double>& data):
    labels{labels},
    data{data}{}

    //Allows user to use data
    const std::vector<double>& getData() const{ return data; }

    //Allows user to print data
    void print() const{
    for (const auto& row : labels){
        for (const auto& value : row){
            std::cout << value << " ";
        }
        std::cout << "\n";
    }
}
};