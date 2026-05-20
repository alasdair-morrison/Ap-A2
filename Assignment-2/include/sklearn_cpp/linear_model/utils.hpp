#pragma once
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <string>
class Dataset {
    private:
        std::vector<std::vector<double>> features;
        std::vector<double> labels;

    public:
        Dataset(std::string filename) {
            std::ifstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open file: " + filename);
            }

            std::string line;
            while (std::getline(file, line)) {
                if (line.empty()) {
                    continue;
                }

                std::vector<double> row;
                std::stringstream ss(line);
                std::string cell;
                bool valid_row = true;

                while (std::getline(ss, cell, ',')) {
                    if (!cell.empty() && cell.back() == '\r') {
                        cell.pop_back();
                    }

                    try {
                        row.push_back(std::stod(cell));
                    } catch (...) {
                        valid_row = false;
                        break;
                    }
                }

                if (!valid_row || row.empty()) {
                    continue;
                }

                labels.push_back(row.back());
                row.pop_back();
                features.push_back(row);
            }
        }

        Dataset() {
            // Default constructor
        }

        Dataset(std::vector<std::vector<double>> features, std::vector<double> labels) {
            this->features = features;
            this->labels = labels;

        }

        const std::vector<std::vector<double>>& getData() const {
            return features;
        }

        const std::vector<double>& getLabels() const {
            return labels;
        }
};