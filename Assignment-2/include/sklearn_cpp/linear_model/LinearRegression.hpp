 
#ifndef SKLEARN_CPP_LINEAR_MODEL_LINEAR_REGRESSION_HPP
#define SKLEARN_CPP_LINEAR_MODEL_LINEAR_REGRESSION_HPP
 
#include <vector>
#include <cmath>       // For std::abs
#include <iostream>    // For printing training progress
#include <stdexcept>   // For error handling
#include <iomanip>     // For formatting printed numbers
 
namespace sklearn_cpp {
namespace linear_model {
 
    class LinearRegression {
    private:
        // ==============================================================
        // LEARNABLE PARAMETERS (what the model learns during training)
        // ==============================================================
 
        // The weight for each feature. If we have 8 features,
        // this vector has 8 elements: [w1, w2, ..., w8]
        std::vector<double> weights;
 
        // The bias term (also called "intercept").
        // This is the constant added to the prediction.
        double bias;
 
        // ==============================================================
        // HYPERPARAMETERS (settings WE choose before training)
        // ==============================================================
 
        // Learning rate (alpha): controls how big each gradient step is.
        // Too big → model explodes. Too small → takes forever.
        // Typical values: 0.001 to 0.1
        double learningRate;
 
        // Maximum number of times we loop through the update step.
        int maxIterations;
 
        // If the loss changes by less than this between iterations,
        // we consider the model "converged" and stop early.
        double tolerance;
 
        // If true, print the loss value during training so you can
        // watch it decrease (useful for debugging).
        bool verbose;
 
        // How often to print (every N iterations). Printing every
        // single iteration would flood the terminal.
        int printInterval;
 
        // ==============================================================
        // TRAINING HISTORY
        // ==============================================================
 
        // Stores the loss value at each iteration. Useful for checking
        // if the model is actually improving.
        std::vector<double> lossHistory;
 
        // Has the model been trained yet?
        bool isTrained;
 
        // ==============================================================
        // PRIVATE HELPER FUNCTIONS
        // ==============================================================
 
        /*
         * dotProduct
         * ----------
         * Computes the dot product of two vectors:
         *   result = a[0]*b[0] + a[1]*b[1] + ... + a[n-1]*b[n-1]
         *
         * This is the key operation in linear regression.
         * When we do dotProduct(weights, x) + bias, we get our prediction.
         *
         * Example:
         *   weights = [0.5, -0.3]
         *   x       = [10, 20]
         *   dot     = 0.5*10 + (-0.3)*20 = 5 - 6 = -1
         */
        double dotProduct(const std::vector<double>& vecA,
                          const std::vector<double>& vecB) const {
            double result = 0.0;
            for (size_t i = 0; i < vecA.size(); ++i) {
                result += vecA[i] * vecB[i];
            }
            return result;
        }
 
        /*
         * computeLoss
         * -----------
         * Calculates the Mean Squared Error (MSE):
         *   L = (1/m) * SUM_i (prediction_i - actual_i)^2
         *
         * This tells us "on average, how far off are our predictions?"
         * Lower is better. Zero means perfect predictions.
         */
        double computeLoss(const std::vector<std::vector<double>>& xData,
                           const std::vector<double>& yData) const {
            int numSamples = static_cast<int>(xData.size());
            double totalLoss = 0.0;
 
            for (int i = 0; i < numSamples; ++i) {
                double yHat = dotProduct(weights, xData[i]) + bias;
                double error = yHat - yData[i];
                totalLoss += error * error;   // Square the error
            }
 
            return totalLoss / numSamples;    // Average over all samples
        }
 
    public:
        // ==============================================================
        // CONSTRUCTOR
        // ==============================================================
 
        /*
         * Creates a new LinearRegression model.
         *
         * You can customize the hyperparameters, or just use the defaults:
         *   LinearRegression model;                 // All defaults
         *   LinearRegression model(0.01, 5000);     // Custom rate & iterations
         */
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
 
        // ==============================================================
        // fit() — THE TRAINING FUNCTION
        // ==============================================================
 
        /*
         * fit(xTrain, yTrain)
         * -------------------
         * This is where the magic happens. The model looks at your data
         * and learns the best weights and bias.
         *
         * Parameters:
         *   xTrain - 2D vector of features. Each row is one sample.
         *            Example: xTrain[0] = {540.0, 0.0, 0.0, 162.0, ...}
         *   yTrain - 1D vector of labels (what we want to predict).
         *            Example: yTrain[0] = 79.99  (concrete strength)
         *
         * IMPORTANT: You should scale your features first using
         *   StandardScaler before calling fit(). See scaler.hpp.
         */
        void fit(const std::vector<std::vector<double>>& xTrain,
                 const std::vector<double>& yTrain) {
 
            // --- Safety checks ---
            if (xTrain.empty() || yTrain.empty()) {
                throw std::invalid_argument(
                    "LinearRegression::fit: Training data must not be empty.");
            }
            if (xTrain.size() != yTrain.size()) {
                throw std::invalid_argument(
                    "LinearRegression::fit: xTrain and yTrain must have the "
                    "same number of samples.");
            }
 
            int numSamples = static_cast<int>(xTrain.size());    // m
            int numFeatures = static_cast<int>(xTrain[0].size()); // n
 
            // --- Initialize all weights to 0, bias to 0 ---
            weights.assign(numFeatures, 0.0);
            bias = 0.0;
            lossHistory.clear();
 
            // ===================================================
            // GRADIENT DESCENT LOOP
            // ===================================================
            // Each iteration: compute gradients, update parameters
            for (int iter = 0; iter < maxIterations; ++iter) {
 
                // These will accumulate the gradient values
                std::vector<double> gradWeights(numFeatures, 0.0);
                double gradBias = 0.0;
 
                // --- Loop over every training sample ---
                for (int i = 0; i < numSamples; ++i) {
 
                    // 1. Make a prediction with current weights
                    //    y_hat = w1*x1 + w2*x2 + ... + b
                    double yHat = dotProduct(weights, xTrain[i]) + bias;
 
                    // 2. Compute the error (how far off we are)
                    //    error = y_hat - y_true
                    double error = yHat - yTrain[i];
 
                    // 3. Accumulate gradients
                    //    For each weight j:
                    //      dL/dw_j += error * x_j
                    //    For bias:
                    //      dL/db += error * 1
                    for (int j = 0; j < numFeatures; ++j) {
                        gradWeights[j] += error * xTrain[i][j];
                    }
                    gradBias += error;
                }
 
                // 4. Scale gradients by 2/m
                //    (The "2" comes from the derivative of the squared term.
                //     The assignment uses 1/m instead of 1/2m in the loss,
                //     so the gradient has 2/m instead of 1/m.)
                double scaleFactor = 2.0 / numSamples;
                for (int j = 0; j < numFeatures; ++j) {
                    gradWeights[j] *= scaleFactor;
                }
                gradBias *= scaleFactor;
 
                // 5. UPDATE: nudge each parameter in the opposite
                //    direction of its gradient
                //
                //    w_j = w_j - learningRate * gradient_w_j
                //    b   = b   - learningRate * gradient_b
                for (int j = 0; j < numFeatures; ++j) {
                    weights[j] -= learningRate * gradWeights[j];
                }
                bias -= learningRate * gradBias;
 
                // --- Track the loss ---
                double currentLoss = computeLoss(xTrain, yTrain);
                lossHistory.push_back(currentLoss);
 
                // --- Print progress (if verbose) ---
                if (verbose && (iter % printInterval == 0
                                || iter == maxIterations - 1)) {
                    std::cout << "  Iteration " << std::setw(6) << iter
                              << " | Loss: " << std::fixed
                              << std::setprecision(6) << currentLoss
                              << std::endl;
                }
 
                // --- Early stopping ---
                // If the loss barely changed, we've converged — stop early
                if (iter > 0) {
                    double lossDiff = std::abs(
                        lossHistory[iter] - lossHistory[iter - 1]);
                    if (lossDiff < tolerance) {
                        if (verbose) {
                            std::cout << "  Converged at iteration " << iter
                                      << " (loss change: " << lossDiff << ")"
                                      << std::endl;
                        }
                        break;
                    }
                }
            }
 
            isTrained = true;
        }
 
        // ==============================================================
        // predict() — USE THE TRAINED MODEL
        // ==============================================================
 
        /*
         * predict(xData)
         * ---------------
         * After training, use this to predict outputs for new data.
         *
         * For each sample x:
         *   prediction = w1*x1 + w2*x2 + ... + wn*xn + b
         *
         * IMPORTANT: If you scaled the training data, you must scale
         * the prediction data the same way (using the same scaler).
         *
         * Parameters:
         *   xData - 2D vector of features to predict on
         *
         * Returns:
         *   Vector of predictions (one per sample)
         */
        std::vector<double> predict(
            const std::vector<std::vector<double>>& xData) const {
 
            if (!isTrained) {
                throw std::runtime_error(
                    "LinearRegression::predict: Model not trained yet! "
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
 
        /*
         * predictSingle(xSample)
         * ----------------------
         * Predict for just ONE sample (convenience function).
         */
        double predictSingle(const std::vector<double>& xSample) const {
            if (!isTrained) {
                throw std::runtime_error(
                    "LinearRegression::predictSingle: Model not trained.");
            }
            return dotProduct(weights, xSample) + bias;
        }
 
        // ==============================================================
        // GETTERS — access the model's internals
        // ==============================================================
 
        const std::vector<double>& getWeights() const { return weights; }
        double getBias() const { return bias; }
        const std::vector<double>& getLossHistory() const { return lossHistory; }
        bool getIsTrained() const { return isTrained; }
    };
 
} // namespace linear_model
} // namespace sklearn_cpp
 
#endif // SKLEARN_CPP_LINEAR_MODEL_LINEAR_REGRESSION_HPP