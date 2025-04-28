#include <iostream>
#include <ceres/ceres.h>
#include <Eigen/Dense>

// Define a simple cost function for optimization
struct CostFunction {
    CostFunction(double x, double y) : x_(x), y_(y) {}

    template <typename T>
    bool operator()(const T* const m, const T* const b, T* residual) const {
        residual[0] = T(y_) - (m[0] * T(x_) + b[0]);
        return true;
    }

    private:
        const double x_;
        const double y_;
};

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    // Sample data points
    std::vector<double> x_data = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> y_data = {1.0, 2.0, 3.0, 4.0};

    // Initial parameters for the line: y = mx + b
    double m = 0.0; // slope
    double b = 0.0; // intercept

    ceres::Problem problem;

    // Add residuals for each data point
    for (size_t i = 0; i < x_data.size(); ++i) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CostFunction, 1, 1, 1>(
                new CostFunction(x_data[i], y_data[i])),
            nullptr, &m, &b);
    }

    // Configure the solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << "\n";
    std::cout << "Estimated m: " << m << ", b: " << b << "\n";

    return 0;
}