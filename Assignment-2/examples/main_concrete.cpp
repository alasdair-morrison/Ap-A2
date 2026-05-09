#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <random>

#include "../include/sklearn_cpp/linear_model/utils.hpp"
#include "../include/sklearn_cpp/linear_model/LinearRegression.hpp"

using sklearn_cpp::linear_model::LinearRegression;
using sklearn_cpp::utils::StandardScaler;

// Shuffle data and labels together using a fixed seed for reproducibility
void shuffleData(
    std::vector<std::vector<double>>& xData,
    std::vector<double>& yData,
    unsigned int seed = 42)
{
    if (xData.size() != yData.size()) {
        throw std::invalid_argument("xData and yData size mismatch");
    }

    int totalSamples = static_cast<int>(xData.size());
    std::vector<int> indices(totalSamples);
    for (int i = 0; i < totalSamples; ++i) indices[i] = i;

    std::mt19937 generator(seed);
    std::shuffle(indices.begin(), indices.end(), generator);

    std::vector<std::vector<double>> shuffledX(totalSamples);
    std::vector<double> shuffledY(totalSamples);

    for (int i = 0; i < totalSamples; ++i) {
        shuffledX[i] = xData[indices[i]];
        shuffledY[i] = yData[indices[i]];
    }

    xData = shuffledX;
    yData = shuffledY;
}

// Split data into train/test sets (sequential after shuffling)
void trainTestSplit(
    const std::vector<std::vector<double>>& xData,
    const std::vector<double>& yData,
    std::vector<std::vector<double>>& xTrain,
    std::vector<double>& yTrain,
    std::vector<std::vector<double>>& xTest,
    std::vector<double>& yTest,
    double testSize = 0.2)
{
    if (xData.size() != yData.size()) {
        throw std::invalid_argument("xData and yData size mismatch");
    }

    int totalSamples = static_cast<int>(xData.size());
    int trainSize    = static_cast<int>(totalSamples * (1.0 - testSize));

    for (int i = 0; i < trainSize; ++i) {
        xTrain.push_back(xData[i]);
        yTrain.push_back(yData[i]);
    }
    for (int i = trainSize; i < totalSamples; ++i) {
        xTest.push_back(xData[i]);
        yTest.push_back(yData[i]);
    }
}

double calculateMSE(
    const std::vector<double>& yActual,
    const std::vector<double>& yPredicted)
{
    if (yActual.size() != yPredicted.size()) {
        throw std::invalid_argument("Size mismatch");
    }
    double sum = 0.0;
    int n = static_cast<int>(yActual.size());
    for (int i = 0; i < n; ++i) {
        double e = yPredicted[i] - yActual[i];
        sum += e * e;
    }
    return sum / n;
}

double calculateMAE(
    const std::vector<double>& yActual,
    const std::vector<double>& yPredicted)
{
    if (yActual.size() != yPredicted.size()) {
        throw std::invalid_argument("Size mismatch");
    }
    double sum = 0.0;
    int n = static_cast<int>(yActual.size());
    for (int i = 0; i < n; ++i) {
        sum += std::abs(yPredicted[i] - yActual[i]);
    }
    return sum / n;
}

// Degree-2 polynomial feature expansion applied to raw (unscaled) features.
//
// For n original features this produces:
//   n   original features      x_i
//   n   squared terms          x_i^2
//   n*(n-1)/2 interactions     x_i * x_j  (i < j)
// ──────────────────────────────────────────────────
//   n*(n+3)/2 total            (for n=8 → 44 features)
//
// Expansion is done BEFORE StandardScaler so the scaler normalises every
// new feature independently — the same pattern as sklearn's Pipeline.
//
// This is STILL linear regression: the model learns one weight per
// expanded feature.  The non-linearity lives in the input space, not in
// the model.  The fit() and predict() calls are identical to before.
std::vector<std::vector<double>> addPolynomialFeatures(
    const std::vector<std::vector<double>>& xData)
{
    std::vector<std::vector<double>> xPoly;
    xPoly.reserve(xData.size());

    for (const auto& row : xData) {
        int n = static_cast<int>(row.size());
        std::vector<double> newRow = row;          // keep all original features

        // Squared terms: x_i^2
        for (int i = 0; i < n; ++i) {
            newRow.push_back(row[i] * row[i]);
        }

        // Pairwise interaction terms: x_i * x_j (i < j)
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                newRow.push_back(row[i] * row[j]);
            }
        }

        xPoly.push_back(newRow);
    }

    return xPoly;
}

int main() {
    try {
        // ============================================================
        // 1. Load dataset
        // ============================================================
        std::cout << "Loading concrete.csv..." << std::endl;
        Database dataset("data/concrete.csv");

        auto xData = dataset.getData();
        auto yData = dataset.getLabels();

        int totalSamples = static_cast<int>(xData.size());
        int numFeatures  = static_cast<int>(xData[0].size());

        std::cout << "Loaded " << totalSamples << " samples with "
                  << numFeatures << " features" << std::endl;

        // ============================================================
        // 2. Shuffle then split — shared for both models
        //    (seed=42 for reproducibility)
        // ============================================================
        shuffleData(xData, yData, 42);

        std::vector<std::vector<double>> xTrain, xTest;
        std::vector<double> yTrain, yTest;
        trainTestSplit(xData, yData, xTrain, yTrain, xTest, yTest, 0.2);

        int numTrain = static_cast<int>(xTrain.size());
        int numTest  = static_cast<int>(xTest.size());

        double trainYMean = 0.0, testYMean = 0.0;
        for (double v : yTrain) trainYMean += v;
        for (double v : yTest)  testYMean  += v;
        trainYMean /= numTrain;
        testYMean  /= numTest;

        std::cout << "Split  -> Train: " << numTrain << " | Test: " << numTest << std::endl;
        std::cout << "y-mean -> Train: " << std::fixed << std::setprecision(2) << trainYMean
                  << " | Test: " << testYMean
                  << "  (should be close)" << std::endl << std::endl;

        // ============================================================
        // 3. MODEL 1 — Baseline: 8 original features
        // ============================================================
        std::cout << "============================================================" << std::endl;
        std::cout << "  MODEL 1: Linear Regression  (" << numFeatures << " features)" << std::endl;
        std::cout << "============================================================" << std::endl;

        StandardScaler scalerLinear;
        auto xTrainLinearScaled = scalerLinear.fitTransform(xTrain);
        auto xTestLinearScaled  = scalerLinear.transform(xTest);

        LinearRegression modelLinear(0.01, 100000, 1e-8, true, 5000);
        modelLinear.fit(xTrainLinearScaled, yTrain);

        auto predLinear  = modelLinear.predict(xTestLinearScaled);
        double mseLinear = calculateMSE(yTest, predLinear);
        double maeLinear = calculateMAE(yTest, predLinear);
        double r2Linear  = sklearn_cpp::utils::r2Score(yTest, predLinear);

        std::cout << "Bias: " << std::fixed << std::setprecision(4)
                  << modelLinear.getBias() << std::endl;
        std::cout << "MSE:  " << std::fixed << std::setprecision(4) << mseLinear << std::endl;
        std::cout << "MAE:  " << maeLinear << std::endl;
        std::cout << "R2:   " << r2Linear  << std::endl << std::endl;

        // ============================================================
        // 4. MODEL 2 — Polynomial degree-2 expansion (still LinearRegression)
        //
        //    WHY THIS HELPS:
        //    Concrete strength is not a purely linear function of its
        //    ingredients.  For example, the water-to-cement ratio is a
        //    ratio (non-linear), and strength grows roughly with log(age).
        //    Degree-2 polynomial features let the same linear weight
        //    equation capture those curved relationships without changing
        //    LinearRegression at all — we just give it richer inputs.
        //
        //    WHY IT IS STILL LINEAR REGRESSION:
        //    The model remains  y = w1*phi1 + w2*phi2 + ... + wk*phik + b
        //    where each phi_i is a (possibly non-linear) function of the
        //    raw inputs.  The model is linear in the WEIGHTS, which is the
        //    definition of linear regression.  fit() and predict() are
        //    identical; only the feature matrix handed to them changes.
        // ============================================================
        std::cout << "============================================================" << std::endl;
        std::cout << "  MODEL 2: Linear Regression + Polynomial Features (deg 2)" << std::endl;
        std::cout << "============================================================" << std::endl;

        // Expand raw features — expansion must happen before scaling
        auto xTrainPoly = addPolynomialFeatures(xTrain);
        auto xTestPoly  = addPolynomialFeatures(xTest);

        int numPolyFeatures = static_cast<int>(xTrainPoly[0].size());
        int numSquared      = numFeatures;
        int numInteractions = numPolyFeatures - 2 * numFeatures;

        std::cout << "Feature expansion: " << numFeatures << " original"
                  << "  +  " << numSquared << " squared"
                  << "  +  " << numInteractions << " interactions"
                  << "  =  " << numPolyFeatures << " total" << std::endl;

        // Fit scaler ONLY on training poly features to avoid data leakage
        StandardScaler scalerPoly;
        auto xTrainPolyScaled = scalerPoly.fitTransform(xTrainPoly);
        auto xTestPolyScaled  = scalerPoly.transform(xTestPoly);

        LinearRegression modelPoly(0.01, 100000, 1e-8, true, 5000);
        modelPoly.fit(xTrainPolyScaled, yTrain);

        auto predPoly  = modelPoly.predict(xTestPolyScaled);
        double msePoly = calculateMSE(yTest, predPoly);
        double maePoly = calculateMAE(yTest, predPoly);
        double r2Poly  = sklearn_cpp::utils::r2Score(yTest, predPoly);

        std::cout << "Bias: " << std::fixed << std::setprecision(4)
                  << modelPoly.getBias() << std::endl;
        std::cout << "MSE:  " << std::fixed << std::setprecision(4) << msePoly << std::endl;
        std::cout << "MAE:  " << maePoly << std::endl;
        std::cout << "R2:   " << r2Poly  << std::endl << std::endl;

        // ============================================================
        // 5. Comparison table
        // ============================================================
        std::cout << "============================================================" << std::endl;
        std::cout << "  COMPARISON" << std::endl;
        std::cout << "============================================================" << std::endl;
        std::cout << std::left
                  << std::setw(32) << "Model"
                  << std::right
                  << std::setw(10) << "Features"
                  << std::setw(10) << "MSE"
                  << std::setw(10) << "MAE"
                  << std::setw(9)  << "R2" << std::endl;
        std::cout << std::string(71, '-') << std::endl;

        std::cout << std::left  << std::setw(32) << "Linear Regression"
                  << std::right << std::setw(10) << numFeatures
                  << std::setw(10) << std::fixed << std::setprecision(2) << mseLinear
                  << std::setw(10) << maeLinear
                  << std::setw(9)  << std::setprecision(4) << r2Linear << std::endl;

        std::cout << std::left  << std::setw(32) << "Linear Reg + Poly deg-2"
                  << std::right << std::setw(10) << numPolyFeatures
                  << std::setw(10) << std::setprecision(2) << msePoly
                  << std::setw(10) << maePoly
                  << std::setw(9)  << std::setprecision(4) << r2Poly << std::endl;

        double r2Gain  = r2Poly  - r2Linear;
        double mseGain = mseLinear - msePoly;
        std::cout << std::endl;
        std::cout << "R2 improvement:  +" << std::fixed << std::setprecision(4) << r2Gain  << std::endl;
        std::cout << "MSE reduction:   -" << std::fixed << std::setprecision(2) << mseGain << std::endl;
        std::cout << std::endl;

        // ============================================================
        // 6. Sample predictions — polynomial model (first 10 test samples)
        // ============================================================
        std::cout << "Sample predictions — polynomial model (first 10 test samples):" << std::endl;
        std::cout << std::setw(8)  << "Sample"
                  << std::setw(12) << "Predicted"
                  << std::setw(12) << "Actual"
                  << std::setw(10) << "Error" << std::endl;
        std::cout << std::string(42, '-') << std::endl;

        int numToShow = std::min(10, numTest);
        for (int i = 0; i < numToShow; ++i) {
            double err = predPoly[i] - yTest[i];
            std::cout << std::setw(8)  << i + 1
                      << std::setw(12) << std::fixed << std::setprecision(2) << predPoly[i]
                      << std::setw(12) << yTest[i]
                      << std::setw(10) << err << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
