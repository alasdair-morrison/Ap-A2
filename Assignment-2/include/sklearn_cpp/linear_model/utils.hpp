#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>

class Dataset
{
private:
    std::vector<std::vector<double>> data;

    //identifies numbers, used to detect header in csv file.
    bool isNumber(const std::string& s){
        std::stringstream ss(s);
        double d;
        char c;
        return (ss >> d) && !(ss >> c);
    }

public:
    // Constructor 
    Dataset(const std::string& filename)
    {
        populate(filename);
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
            if (firstLine)
            {
                bool header = false;
                for (const auto& v : raw)
                {
                    if (!isNumber(v))
                    {
                        header = true;
                        break;
                    }
                }

                firstLine = false;

                if (header)
                    continue;
            }

            for (const auto& v : raw)
            {
                if (!v.empty())
                    row.push_back(std::stod(v));
            }

            data.push_back(row);
        }
    }

    //Allows user to use data
    const std::vector<std::vector<double>>& getData() const{ return data; }

    //Allows user to print data a certain amount of data
    void print() const{
    for (const auto& row : data){
        for (const auto& value : row){
            std::cout << value << " ";
        }
        std::cout << "\n";
    }
}
};
#endif