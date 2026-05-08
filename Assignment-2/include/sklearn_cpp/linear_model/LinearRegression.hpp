#pragma once
 
#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <iomanip>
 
namespace sklearn_cpp {
namespace utils {
 

    // STANDARD SCALER CLASS
    // Normalizes features to mean=0, std=1 so gradient descent works.
    // Without this, features with large values (cement ~500) dominate
    // over features with small values (superplastic ~5).
 
    class StandardScaler {
    private:
        std::vector<double> featureMeans;
        std::vector<double> featureStdDevs;
        bool isFitted;
 
    public:
        StandardScaler() : isFitted(false) {}
 
        void fit(const std::vector<std::vector<double>>& xData) {
            if (xData.empty()) {
                throw std::runtime_error("StandardScaler::fit: Empty dataset.");
            }
 
            int numSamples = static_cast<int>(xData.size());
            int numFeatures = static_cast<int>(xData[0].size());
 
            featureMeans.assign(numFeatures, 0.0);
            featureStdDevs.assign(numFeatures, 0.0);
 
            // Compute means
            for (int i = 0; i < numSamples; ++i) {
                for (int j = 0; j < numFeatures; ++j) {
                    featureMeans[j] += xData[i][j];
                }
            }
            for (int j = 0; j < numFeatures; ++j) {
                featureMeans[j] /= numSamples;
            }
 
            // Compute standard deviations
            for (int i = 0; i < numSamples; ++i) {
                for (int j = 0; j < numFeatures; ++j) {
                    double diff = xData[i][j] - featureMeans[j];
                    featureStdDevs[j] += diff * diff;
                }
            }
            for (int j = 0; j < numFeatures; ++j) {
                featureStdDevs[j] = std::sqrt(featureStdDevs[j] / numSamples);
                if (featureStdDevs[j] < 1e-12) {
                    featureStdDevs[j] = 1.0;
                }
            }
 
            isFitted = true;
        }
 
        std::vector<std::vector<double>> transform(
            const std::vector<std::vector<double>>& xData) const
        {
            if (!isFitted) {
                throw std::runtime_error(
                    "StandardScaler::transform: Not fitted. Call fit() first.");
            }
 
            int numSamples = static_cast<int>(xData.size());
            int numFeatures = static_cast<int>(xData[0].size());
            std::vector<std::vector<double>> xScaled(
                numSamples, std::vector<double>(numFeatures));
 
            for (int i = 0; i < numSamples; ++i) {
                for (int j = 0; j < numFeatures; ++j) {
                    xScaled[i][j] = (xData[i][j] - featureMeans[j]) 
                                    / featureStdDevs[j];
                }
            }
 
            return xScaled;
        }
 
        std::vector<std::vector<double>> fitTransform(
            const std::vector<std::vector<double>>& xData)
        {
            fit(xData);
            return transform(xData);
        }
 
        const std::vector<double>& getMeans() const { return featureMeans; }
        const std::vector<double>& getStdDevs() const { return featureStdDevs; }
    };
 
    // R² SCORE FUNCTION
    // Evaluates model performance. 1.0 = perfect, 0.0 = bad.
 
    inline double r2Score(const std::vector<double>& yTrue,
                          const std::vector<double>& yPred) {
        if (yTrue.size() != yPred.size() || yTrue.empty()) {
            throw std::invalid_argument("r2Score: Invalid input sizes");
        }
 
        int numSamples = static_cast<int>(yTrue.size());
 
        double yMean = 0.0;
        for (int i = 0; i < numSamples; ++i) {
            yMean += yTrue[i];
        }
        yMean /= numSamples;
 
        double sResidual = 0.0;
        double sTotal = 0.0;
        for (int i = 0; i < numSamples; ++i) {
            double residual = yTrue[i] - yPred[i];
            double deviation = yTrue[i] - yMean;
            sResidual += residual * residual;
            sTotal += deviation * deviation;
        }
 
        if (sTotal < 1e-15) {
            return (sResidual < 1e-15) ? 1.0 : 0.0;
        }
 
        return 1.0 - (sResidual / sTotal);
    }
 
} // namespace utils
 
namespace linear_model {
 
    // LINEAR REGRESSION CLASS
    // Implements gradient descent to learn weights for multi-feature
    // linear regression: y = w1*x1 + w2*x2 + ... + wn*xn + b
 
    class LinearRegression {
    private:
        std::vector<double> weights;
        double bias;
        double learningRate;
        int maxIterations;
        double tolerance;
        bool verbose;
        int printInterval;
        std::vector<double> lossHistory;
        bool isTrained;
 
        double dotProduct(const std::vector<double>& vecA,
                          const std::vector<double>& vecB) const {
            double result = 0.0;
            for (size_t i = 0; i < vecA.size(); ++i) {
                result += vecA[i] * vecB[i];
            }
            return result;
        }
 
        double computeLoss(const std::vector<std::vector<double>>& xData,
                           const std::vector<double>& yData) const {
            int numSamples = static_cast<int>(xData.size());
            double totalLoss = 0.0;
 
            for (int i = 0; i < numSamples; ++i) {
                double yHat = dotProduct(weights, xData[i]) + bias;
                double error = yHat - yData[i];
                totalLoss += error * error;
            }
 
            return totalLoss / numSamples;
        }
 
    public:
        LinearRegression(double learningRate = 0.01,
                         int maxIterations = 1000,
                         double tolerance = 1e-8,
                         bool verbose = false,
                         int printInterval = 100)
            : bias(0.0),
              learningRate(learningRate),
              maxIterations(maxIterations),
              tolerance(tolerance),
              verbose(verbose),
              printInterval(printInterval),
              isTrained(false) {}
 
        /*
         * fit() - Train the model using gradient descent
         * 
         * Implements the update rule from assignment 03-maths:
         *   wJ = wJ - alpha * (2/m) * SUM_i [(yHat - y) * xJ]
         *   b  = b  - alpha * (2/m) * SUM_i [(yHat - y)]
         
         */
        void fit(const std::vector<std::vector<double>>& xTrain,
                 const std::vector<double>& yTrain) {
 
            if (xTrain.empty() || yTrain.empty()) {
                throw std::invalid_argument(
                    "LinearRegression::fit: Training data must not be empty.");
            }
            if (xTrain.size() != yTrain.size()) {
                throw std::invalid_argument(
                    "LinearRegression::fit: xTrain and yTrain size mismatch.");
            }
 
            int numSamples = static_cast<int>(xTrain.size());
            int numFeatures = static_cast<int>(xTrain[0].size());
 
            weights.assign(numFeatures, 0.0);
            bias = 0.0;
            lossHistory.clear();
 
            // Gradient descent loop
            for (int iter = 0; iter < maxIterations; ++iter) {
 
                std::vector<double> gradWeights(numFeatures, 0.0);
                double gradBias = 0.0;
 
                // Compute gradients over all samples
                for (int i = 0; i < numSamples; ++i) {
                    double yHat = dotProduct(weights, xTrain[i]) + bias;
                    double error = yHat - yTrain[i];
 
                    for (int j = 0; j < numFeatures; ++j) {
                        gradWeights[j] += error * xTrain[i][j];
                    }
                    gradBias += error;
                }
 
                // Scale by 2/m (from derivative of MSE)
                double scaleFactor = 2.0 / numSamples;
                for (int j = 0; j < numFeatures; ++j) {
                    gradWeights[j] *= scaleFactor;
                }
                gradBias *= scaleFactor;
 
                // Update weights and bias
                for (int j = 0; j < numFeatures; ++j) {
                    weights[j] -= learningRate * gradWeights[j];
                }
                bias -= learningRate * gradBias;
 
                // Track loss
                double currentLoss = computeLoss(xTrain, yTrain);
                lossHistory.push_back(currentLoss);
 
                if (verbose && (iter % printInterval == 0 
                                || iter == maxIterations - 1)) {
                    std::cout << "  Iteration " << std::setw(6) << iter
                              << " | Loss: " << std::fixed
                              << std::setprecision(6) << currentLoss
                              << std::endl;
                }
 
                // Early stopping
                if (iter > 0) {
                    double lossDiff = std::abs(
                        lossHistory[iter] - lossHistory[iter - 1]);
                    if (lossDiff < tolerance) {
                        if (verbose) {
                            std::cout << "  Converged at iteration " << iter
                                      << std::endl;
                        }
                        break;
                    }
                }
            }
 
            isTrained = true;
        }
 
        /*
         * predict() - Make predictions on new data
         * 
         * Returns: vector of predicted values
         */
        std::vector<double> predict(
            const std::vector<std::vector<double>>& xData) const {
 
            if (!isTrained) {
                throw std::runtime_error(
                    "LinearRegression::predict: Model not trained. "
                    "Call fit() first.");
            }
 
            std::vector<double> predictions;
            predictions.reserve(xData.size());
 
            for (const auto& sample : xData) {
                double yHat = dotProduct(weights, sample) + bias;
                predictions.push_back(yHat);
            }
 
            return predictions;
        }
 
        double predictSingle(const std::vector<double>& xSample) const {
            if (!isTrained) {
                throw std::runtime_error(
                    "LinearRegression::predictSingle: Model not trained.");
            }
            return dotProduct(weights, xSample) + bias;
        }
 
        // Getters
        const std::vector<double>& getWeights() const { return weights; }
        double getBias() const { return bias; }
        const std::vector<double>& getLossHistory() const { return lossHistory; }
        bool getIsTrained() const { return isTrained; }
    };
 
} // namespace linear_model
} // namespace sklearn_cpp