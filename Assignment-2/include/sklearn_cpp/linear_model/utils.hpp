#pragma once
#include <vector>
#include <string>
class Database {
    private:
        std::vector<std::vector<double>> features;
        std::vector<int> labels;

    public:
        Database(std::string filename) {
            // Load data from file and populate features and labels
        }

        Database() {
            // Default constructor
        }

        Database(std::vector<std::vector<double>> features, std::vector<int> labels) {
            this->features = features;
            this->labels = labels;

        }

        std::vector<std::vector<double>> getData() {
            return features;
        }

        std::vector<int> getLabels() {
            return labels;
        }
};