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

// ============================================================
// Utility functions (unchanged)
// ============================================================

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

double calculateMSE(const std::vector<double>& yActual,
                    const std::vector<double>& yPredicted)
{
    if (yActual.size() != yPredicted.size())
        throw std::invalid_argument("Size mismatch");
    double sum = 0.0;
    int n = static_cast<int>(yActual.size());
    for (int i = 0; i < n; ++i) {
        double e = yPredicted[i] - yActual[i];
        sum += e * e;
    }
    return sum / n;
}

double calculateMAE(const std::vector<double>& yActual,
                    const std::vector<double>& yPredicted)
{
    if (yActual.size() != yPredicted.size())
        throw std::invalid_argument("Size mismatch");
    double sum = 0.0;
    int n = static_cast<int>(yActual.size());
    for (int i = 0; i < n; ++i)
        sum += std::abs(yPredicted[i] - yActual[i]);
    return sum / n;
}

// ============================================================
// Feature engineering functions
// ============================================================

// Degree-2 polynomial expansion (generic — works for any feature count).
//
// For n input features produces n + n + n*(n-1)/2 = n*(n+3)/2 features:
//   original | squared x_i^2 | interactions x_i*x_j (i<j)
//
// Applied BEFORE StandardScaler so every expanded feature is
// normalised independently.  The model remains:
//   y = w1*phi1 + w2*phi2 + ... + wk*phik + b
// which is linear in the weights — still Linear Regression.
std::vector<std::vector<double>> addPolynomialFeatures(
    const std::vector<std::vector<double>>& xData)
{
    std::vector<std::vector<double>> xPoly;
    xPoly.reserve(xData.size());
    for (const auto& row : xData) {
        int n = static_cast<int>(row.size());
        std::vector<double> newRow = row;
        for (int i = 0; i < n; ++i)
            newRow.push_back(row[i] * row[i]);
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                newRow.push_back(row[i] * row[j]);
        xPoly.push_back(newRow);
    }
    return xPoly;
}

// Domain-specific feature engineering for concrete compressive strength.
//
// Raw features (column order from CSV):
//   0=cement  1=slag  2=ash  3=water  4=superplastic
//   5=coarseagg  6=fineagg  7=age
//
// Added features and their physical justification:
//
//  8  log(age)
//       Concrete strength grows ~linearly with the LOGARITHM of curing
//       time (Powers' maturity law).  A degree-2 expansion gives age^2,
//       which captures the wrong curvature — old concrete (365 days) is
//       extrapolated to be quadratically stronger than 28-day concrete,
//       whereas in reality the gains taper off.  log(age) fixes this.
//
//  9  water / cement  (w/c ratio)
//       Abrams' law (1919): f_c ≈ A / B^(w/c).  This is a RATIO, which
//       a polynomial expansion of raw w and c cannot represent — it would
//       need to learn 1/cement from {cement, cement^2, cement*water, ...},
//       which requires an inverse, not available in degree-2 space.
//       Providing the ratio directly gives the model the key predictor.
//
// 10  cement + slag + ash  (total binder)
//       The combined cementitious content drives long-term hydration and
//       strength.  The linear model already uses the three columns
//       separately; making their sum explicit helps the model when slag
//       and ash partially replace cement (blended mix designs).
//
// 11  (cement + slag + ash) / water  (binder-to-water ratio)
//       A generalisation of Abrams' law that accounts for supplementary
//       cementitious materials.  More binder per unit water → stronger
//       concrete.  This is the strongest single predictor in blended mixes.
//
// Result: 8 raw → 12 base features, then degree-2 → 90 features.
std::vector<std::vector<double>> addDomainFeatures(
    const std::vector<std::vector<double>>& xData)
{
    std::vector<std::vector<double>> result;
    result.reserve(xData.size());

    for (const auto& row : xData) {
        double cement       = row[0];
        double slag         = row[1];
        double ash          = row[2];
        double water        = row[3];
        double age          = row[7];

        std::vector<double> newRow = row;               // keep original 8

        // Feature 8: log(age) — maturity law
        // age >= 1 in this dataset so log(age) >= 0; safe, no +1 needed.
        newRow.push_back(std::log(age));

        // Feature 9: w/c ratio — Abrams' law
        // cement >= 102 in this dataset; division is always safe.
        newRow.push_back(water / cement);

        // Feature 10: total binder
        double binder = cement + slag + ash;
        newRow.push_back(binder);

        // Feature 11: binder-to-water ratio
        // water >= 121.8 in this dataset; division is always safe.
        newRow.push_back(binder / water);

        result.push_back(newRow);
    }

    return result;
}

// ============================================================
// Main: three-model comparison
// ============================================================
int main() {
    try {
        // --------------------------------------------------------
        // 1. Load and verify dataset
        // --------------------------------------------------------
        std::cout << "Loading concrete.csv..." << std::endl;
        Database dataset("data/concrete.csv");

        auto xData = dataset.getData();
        auto yData = dataset.getLabels();

        int totalSamples = static_cast<int>(xData.size());
        int numRawFeatures = static_cast<int>(xData[0].size());

        std::cout << "Loaded " << totalSamples << " samples, "
                  << numRawFeatures << " raw features" << std::endl;

        // --------------------------------------------------------
        // 2. Shuffle + split — identical split used by all models
        // --------------------------------------------------------
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
        std::cout << "y-mean -> Train: " << std::fixed << std::setprecision(2)
                  << trainYMean << " | Test: " << testYMean
                  << "  (should be close)" << std::endl << std::endl;

        // --------------------------------------------------------
        // MODEL 1 — Baseline: 8 raw features, linear
        // --------------------------------------------------------
        std::cout << "============================================================" << std::endl;
        std::cout << "  MODEL 1: Linear Regression  (" << numRawFeatures << " features)" << std::endl;
        std::cout << "============================================================" << std::endl;

        StandardScaler scalerM1;
        auto xTrainM1 = scalerM1.fitTransform(xTrain);
        auto xTestM1  = scalerM1.transform(xTest);

        // verbose=false: baseline is well-understood, no need to show iterations
        LinearRegression modelM1(0.01, 100000, 1e-8, false, 5000);
        modelM1.fit(xTrainM1, yTrain);

        auto predM1  = modelM1.predict(xTestM1);
        double mseM1 = calculateMSE(yTest, predM1);
        double maeM1 = calculateMAE(yTest, predM1);
        double r2M1  = sklearn_cpp::utils::r2Score(yTest, predM1);

        std::cout << "Bias:  " << std::fixed << std::setprecision(4) << modelM1.getBias() << std::endl;
        std::cout << "MSE:   " << mseM1 << std::endl;
        std::cout << "MAE:   " << maeM1 << std::endl;
        std::cout << "R2:    " << r2M1  << std::endl << std::endl;

        // --------------------------------------------------------
        // MODEL 2 — Degree-2 poly on 8 raw features (44 features)
        // --------------------------------------------------------
        std::cout << "============================================================" << std::endl;
        std::cout << "  MODEL 2: Linear Reg + Poly deg-2  (44 features)" << std::endl;
        std::cout << "============================================================" << std::endl;

        auto xTrainM2raw = addPolynomialFeatures(xTrain);
        auto xTestM2raw  = addPolynomialFeatures(xTest);

        StandardScaler scalerM2;
        auto xTrainM2 = scalerM2.fitTransform(xTrainM2raw);
        auto xTestM2  = scalerM2.transform(xTestM2raw);

        // verbose=false: previously demonstrated; skip the 100k-iteration log
        LinearRegression modelM2(0.01, 100000, 1e-8, false, 5000);
        modelM2.fit(xTrainM2, yTrain);

        auto predM2  = modelM2.predict(xTestM2);
        double mseM2 = calculateMSE(yTest, predM2);
        double maeM2 = calculateMAE(yTest, predM2);
        double r2M2  = sklearn_cpp::utils::r2Score(yTest, predM2);

        std::cout << "Bias:  " << std::fixed << std::setprecision(4) << modelM2.getBias() << std::endl;
        std::cout << "MSE:   " << mseM2 << std::endl;
        std::cout << "MAE:   " << maeM2 << std::endl;
        std::cout << "R2:    " << r2M2  << std::endl << std::endl;

        // --------------------------------------------------------
        // MODEL 3 — Domain engineering + degree-2 poly (90 features)
        //
        // Pipeline:
        //   raw 8 → addDomainFeatures() → 12 base
        //         → addPolynomialFeatures() → 90 features
        //         → StandardScaler (fit on train only)
        //         → LinearRegression (unchanged fit/predict API)
        //
        // This is still Linear Regression: y = w · phi(x) + b
        // The model is linear in the weights.  phi(x) encodes the
        // domain knowledge; the weights are still learned by gradient
        // descent exactly as before.
        // --------------------------------------------------------
        std::cout << "============================================================" << std::endl;
        std::cout << "  MODEL 3: Domain Features + Poly deg-2  (90 features)" << std::endl;
        std::cout << "============================================================" << std::endl;
        std::cout << "  Engineered features added before polynomial expansion:" << std::endl;
        std::cout << "    feat 8  log(age)              — maturity law curvature" << std::endl;
        std::cout << "    feat 9  water/cement           — Abrams w/c ratio" << std::endl;
        std::cout << "    feat 10 cement+slag+ash        — total binder content" << std::endl;
        std::cout << "    feat 11 binder/water           — blended Abrams ratio" << std::endl;
        std::cout << "  8 raw + 4 domain = 12 base  -->  degree-2 --> 90 total" << std::endl;
        std::cout << std::endl;

        // Step 1: domain features on raw (unscaled) data
        auto xTrainDomain = addDomainFeatures(xTrain);
        auto xTestDomain  = addDomainFeatures(xTest);

        // Step 2: degree-2 expansion on the 12-feature set
        auto xTrainM3raw = addPolynomialFeatures(xTrainDomain);
        auto xTestM3raw  = addPolynomialFeatures(xTestDomain);

        int numM3Features = static_cast<int>(xTrainM3raw[0].size());
        std::cout << "  Total features after expansion: " << numM3Features << std::endl;

        // Step 3: scale — ONLY fit on training data
        StandardScaler scalerM3;
        auto xTrainM3 = scalerM3.fitTransform(xTrainM3raw);
        auto xTestM3  = scalerM3.transform(xTestM3raw);

        // Step 4: train — verbose=true so we can watch convergence
        LinearRegression modelM3(0.01, 100000, 1e-8, true, 10000);
        modelM3.fit(xTrainM3, yTrain);

        auto predM3  = modelM3.predict(xTestM3);
        double mseM3 = calculateMSE(yTest, predM3);
        double maeM3 = calculateMAE(yTest, predM3);
        double r2M3  = sklearn_cpp::utils::r2Score(yTest, predM3);

        std::cout << "Bias:  " << std::fixed << std::setprecision(4) << modelM3.getBias() << std::endl;
        std::cout << "MSE:   " << mseM3 << std::endl;
        std::cout << "MAE:   " << maeM3 << std::endl;
        std::cout << "R2:    " << r2M3  << std::endl << std::endl;

        // --------------------------------------------------------
        // Comparison table
        // --------------------------------------------------------
        std::cout << "============================================================" << std::endl;
        std::cout << "  COMPARISON" << std::endl;
        std::cout << "============================================================" << std::endl;
        std::cout << std::left  << std::setw(36) << "Model"
                  << std::right << std::setw(10) << "Features"
                  << std::setw(10) << "MSE"
                  << std::setw(10) << "MAE"
                  << std::setw(9)  << "R2" << std::endl;
        std::cout << std::string(75, '-') << std::endl;

        std::cout << std::left  << std::setw(36) << "M1: Linear Regression"
                  << std::right << std::setw(10) << numRawFeatures
                  << std::setw(10) << std::fixed << std::setprecision(2) << mseM1
                  << std::setw(10) << maeM1
                  << std::setw(9)  << std::setprecision(4) << r2M1 << std::endl;

        std::cout << std::left  << std::setw(36) << "M2: + Poly deg-2"
                  << std::right << std::setw(10) << 44
                  << std::setw(10) << std::setprecision(2) << mseM2
                  << std::setw(10) << maeM2
                  << std::setw(9)  << std::setprecision(4) << r2M2 << std::endl;

        std::cout << std::left  << std::setw(36) << "M3: + Domain + Poly deg-2"
                  << std::right << std::setw(10) << numM3Features
                  << std::setw(10) << std::setprecision(2) << mseM3
                  << std::setw(10) << maeM3
                  << std::setw(9)  << std::setprecision(4) << r2M3 << std::endl;

        std::cout << std::endl;
        std::cout << "M1 -> M2 gain:  R2 +" << std::fixed << std::setprecision(4)
                  << (r2M2 - r2M1) << "   MSE -" << std::setprecision(2) << (mseM1 - mseM2) << std::endl;
        std::cout << "M2 -> M3 gain:  R2 +" << std::fixed << std::setprecision(4)
                  << (r2M3 - r2M2) << "   MSE -" << std::setprecision(2) << (mseM2 - mseM3) << std::endl;
        std::cout << "M1 -> M3 total: R2 +" << std::fixed << std::setprecision(4)
                  << (r2M3 - r2M1) << "   MSE -" << std::setprecision(2) << (mseM1 - mseM3) << std::endl;
        std::cout << std::endl;

        // --------------------------------------------------------
        // Sample predictions — Model 3 (best model)
        // --------------------------------------------------------
        std::cout << "Sample predictions — Model 3 (first 10 test samples):" << std::endl;
        std::cout << std::setw(8)  << "Sample"
                  << std::setw(12) << "Predicted"
                  << std::setw(12) << "Actual"
                  << std::setw(10) << "Error" << std::endl;
        std::cout << std::string(42, '-') << std::endl;

        int numToShow = std::min(10, numTest);
        for (int i = 0; i < numToShow; ++i) {
            double err = predM3[i] - yTest[i];
            std::cout << std::setw(8)  << i + 1
                      << std::setw(12) << std::fixed << std::setprecision(2) << predM3[i]
                      << std::setw(12) << yTest[i]
                      << std::setw(10) << err << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
