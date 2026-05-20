#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
 
// These are our header-only library files.
// Because we used -I ./include when compiling, the compiler knows
// to look inside ./include for these paths.
#include <sklearn_cpp/utils/csv_reader.hpp>
#include <sklearn_cpp/utils/scaler.hpp>
#include <sklearn_cpp/utils/metrics.hpp>
#include <sklearn_cpp/linear_model/linear_regression.hpp>
 
int main() {
 
    std::cout << "=======================================" << std::endl;
    std::cout << " Linear Regression: Concrete Strength  " << std::endl;
    std::cout << "=======================================" << std::endl;
 
    // ==========================================================
    // STEP 1: LOAD THE DATA
    // ==========================================================
    // concrete.csv has 1030 rows of data (plus a header).
    // Each row has 9 numbers:
    //   8 features: cement, slag, ash, water, superplastic,
    //               coarseagg, fineagg, age
    //   1 target:   strength (what we want to predict)
 
    std::string dataPath = "../data/concrete.csv";
    std::cout << "\n[Step 1] Loading data from: " << dataPath << std::endl;
 
    std::vector<std::vector<double>> rawData;
    try {
        // readCsv returns a 2D vector: 1030 rows x 9 columns
        // The 'true' means "the first line is a header, skip it"
        rawData = sklearn_cpp::utils::readCsv(dataPath, true);
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
 
    std::cout << "    Loaded " << rawData.size() << " samples." << std::endl;
 
    // ==========================================================
    // STEP 2: SPLIT INTO FEATURES (X) AND LABELS (y)
    // ==========================================================
    // X = first 8 columns (the inputs)
    // y = last column (strength — what we predict)
 
    std::vector<std::vector<double>> xData;  // 1030 x 8
    std::vector<double> yData;               // 1030 x 1
    sklearn_cpp::utils::splitFeaturesAndLabels(rawData, xData, yData);
 
    int numSamples = static_cast<int>(xData.size());
    int numFeatures = static_cast<int>(xData[0].size());
    std::cout << "    Samples: " << numSamples
              << " | Features: " << numFeatures << std::endl;
 
    // ==========================================================
    // STEP 3: SCALE THE FEATURES
    // ==========================================================
    // WHY? Cement values are ~100-500, age values are ~1-365.
    // If we don't scale, gradient descent will struggle because
    // the gradient will be huge for some features and tiny for others.
    //
    // After scaling, every feature has mean=0 and std=1.
 
    std::cout << "\n[Step 2] Scaling features..." << std::endl;
 
    sklearn_cpp::utils::StandardScaler scaler;
    std::vector<std::vector<double>> xScaled = scaler.fitTransform(xData);
 
    std::cout << "    Done. All features now have mean=0, std=1." << std::endl;
 
    // ==========================================================
    // STEP 4: CREATE AND TRAIN THE MODEL
    // ==========================================================
    // We create a LinearRegression object and call fit().
    //
    // Hyperparameters:
    //   learningRate = 0.01  (step size — not too big, not too small)
    //   maxIterations = 5000 (try up to 5000 gradient steps)
    //   tolerance = 1e-8     (stop if loss barely changes)
    //   verbose = true       (print progress)
    //   printInterval = 1000 (print every 1000 iterations)
 
    std::cout << "\n[Step 3] Training the model..." << std::endl;
 
    sklearn_cpp::linear_model::LinearRegression model(
        0.01,    // learningRate
        5000,    // maxIterations
        1e-8,    // tolerance
        true,    // verbose (print progress)
        1000     // printInterval
    );
 
    // This is where the model actually learns!
    // It loops through the data thousands of times, adjusting
    // the weights a tiny bit each time to reduce the error.
    model.fit(xScaled, yData);
 
    // ==========================================================
    // STEP 5: EVALUATE THE MODEL
    // ==========================================================
    // We use R² score to see how good the model is.
    // R² close to 1.0 = great, close to 0 = bad.
 
    std::cout << "\n[Step 4] Evaluating..." << std::endl;
 
    // Use the trained model to predict on the same data we trained on
    std::vector<double> yPredicted = model.predict(xScaled);
 
    // Compare predictions to actual values
    double r2Train = sklearn_cpp::utils::r2Score(yData, yPredicted);
 
    std::cout << "    Training R2-score: " << std::fixed << std::setprecision(4)
              << r2Train << std::endl;
    std::cout << "    (For reference, Python sklearn gets ~0.61 on this data)"
              << std::endl;
 
    // ==========================================================
    // STEP 6: SHOW THE LEARNED WEIGHTS
    // ==========================================================
    // These weights tell us how much each ingredient affects strength.
    // Positive weight = more of it → higher strength
    // Negative weight = more of it → lower strength
 
    std::cout << "\n[Step 5] Learned weights:" << std::endl;
 
    const std::vector<double>& learnedWeights = model.getWeights();
    std::vector<std::string> featureNames = {
        "cement", "slag", "ash", "water",
        "superplastic", "coarseagg", "fineagg", "age"
    };
 
    for (int j = 0; j < numFeatures; ++j) {
        std::cout << "    w[" << std::setw(12) << featureNames[j] << "] = "
                  << std::fixed << std::setprecision(4)
                  << learnedWeights[j] << std::endl;
    }
    std::cout << "    " << std::setw(16) << "bias" << "  = "
              << std::fixed << std::setprecision(4)
              << model.getBias() << std::endl;
 
    std::cout << "\n=======================================" << std::endl;
    std::cout << " Done!" << std::endl;
    std::cout << "=======================================" << std::endl;
 
    return 0;
}