#include <iostream>
#include <ceres/ceres.h>
#include <Eigen/Dense>

struct Camera {
  Matrix3d K;           // 内参
  Matrix3d R_wc;        // 世界到相机的旋转（world to cam）
  Vector3d p_wc;        // 相机在世界坐标系的位置
};

template <typename T>
Eigen::Matrix<T, 3, 1> ProjectPluckerToImageLine(const Eigen::Matrix3d& K,
                                                 const Eigen::Matrix3d& R_wc,
                                                 const Eigen::Vector3d& p_wc,
                                                 const Eigen::Matrix<T, 3, 1>& n,
                                                 const Eigen::Matrix<T, 3, 1>& v) {
    // Transform Plücker line to camera frame
    Eigen::Matrix<T, 3, 1> n_cam = R_wc.cast<T>() * n + p_wc.cast<T>().cross(R_wc.cast<T>() * v);
    Eigen::Matrix<T, 3, 1> v_cam = R_wc.cast<T>() * v;

    // Project to image space
    Eigen::Matrix<T, 3, 1> line = K.cast<T>().transpose() * n_cam;
    return line;
}

template <typename T>
T PointLineDistance(const Eigen::Matrix<T, 3, 1>& line, const Eigen::Matrix<T, 2, 1>& point) {
    // Line equation: ax + by + c = 0
    T a = line[0];
    T b = line[1];
    T c = line[2];

    // Point-line distance formula
    T distance = (a * point.x() + b * point.y() + c) / sqrt(a * a + b * b);
    return distance;
}

// Define a simple cost function for optimization
struct LineReprojectionError {
  LineReprojectionError(const Eigen::Vector2d& p1_obs, const Eigen::Vector2d& p2_obs,
                        const Eigen::Matrix3d& K, const Eigen::Matrix3d& R_wc, const Eigen::Vector3d& p_wc)
      : p1_obs_(p1_obs), p2_obs_(p2_obs), K_(K), R_wc_(R_wc), p_wc_(p_wc) {}

  template <typename T>
  bool operator()(const T* const plucker, T* residuals) const {
      // Extract Plücker line components
      Eigen::Matrix<T, 3, 1> n(plucker[0], plucker[1], plucker[2]);
      Eigen::Matrix<T, 3, 1> v(plucker[3], plucker[4], plucker[5]);

      // Project Plücker line to image space
      Eigen::Matrix<T, 3, 1> line = ProjectPluckerToImageLine<T>(K_, R_wc_, p_wc_, n, v);

      // Compute reprojection errors for the two endpoints
      residuals[0] = PointLineDistance<T>(line, p1_obs_.cast<T>());
      residuals[1] = PointLineDistance<T>(line, p2_obs_.cast<T>());

      return true;
  }

  // Observations and camera parameters
  const Eigen::Vector2d p1_obs_;
  const Eigen::Vector2d p2_obs_;
  const Eigen::Matrix3d K_;
  const Eigen::Matrix3d R_wc_;
  const Eigen::Vector3d p_wc_;
};

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    // Camera intrinsic parameter
    Matrix3d K, K_line;
    K << 500, 0, 320,
         0, 500, 240,
         0,   0,   1;

    double fu = 500, fv = 500, cu = 320, cv = 240;
    K_line << fv,   0,    0,
            0,    fu,   0,
            -fv*cu, -fu*cv, fu*fv;
            
    // A line in Plücker
    Vector3d P1(-30, -20, 40), P2(10, 60, 80);
    Vector3d v = (P2 - P1).normalized();
    Vector3d n = P1.cross(v);
    Vector6d plucker;
    plucker.head<3>() = n;
    plucker.tail<3>() = v;
    std::cout << "Plücker line (n | v): " << plucker.transpose() << std::endl;
    // camera pose
    int num_cams = 10;
    vector<Camera> cams = GenerateCameras(num_cams, K);
    std::vector<std::pair<Vector2d, Vector2d>> line_observations;
    // error define
    double total_error = 0;
    for (int i = 0; i < num_cams; ++i) {
        Vector2d p1_proj = ProjectPoint(K, cams[i].R_wc, cams[i].p_wc, P1);
        Vector2d p2_proj = ProjectPoint(K, cams[i].R_wc, cams[i].p_wc, P2);

        // Vector2d p1_obs = AddNoise(p1_proj);
        // Vector2d p2_obs = AddNoise(p2_proj);

        Vector2d p1_obs = p1_proj;
        Vector2d p2_obs = p2_proj;

        line_observations.push_back(std::make_pair(p1_obs, p2_obs));

        Vector3d line = ProjectPluckerToImageLine(K_line, cams[i].R_wc, cams[i].p_wc, plucker);

        double e1 = PointLineDistance(line, p1_obs);
        double e2 = PointLineDistance(line, p2_obs);
        // cout << "Frame " << i << ": error1 = " << e1 << ", error2 = " << e2 << endl;

        total_error += e1 * e1 + e2 * e2;
    }

    cout << "Initial total reprojection error (squared sum): " << total_error << endl;
    
    // Optimization Initialization (with error)
    Eigen::Matrix<double, 3, 2> A;
    A.col(0) = n;
    A.col(1) = v;
    Eigen::HouseholderQR<Eigen::Matrix<double, 3, 2>> qr(A);
    Eigen::Matrix3d U = qr.householderQ();
    Eigen::Matrix<double, 3, 2> R = qr.matrixQR().triangularView<Eigen::Upper>();
    double w1, w2;
    w1 = R(0, 0);
    w2 = R(1, 1);
    Eigen::Matrix<double, 2, 2> W;
    W << w1, -w2, w2, w1;
    Eigen::Matrix<double, 4, 1> dx;
    dx << 0.001, 0.002, 0.002, 0.001;
    UpdateOrthonomal(dx, U, W, U, W);
    Eigen::Vector3d v_est, n_est, u1, u2, u3;
    u1 = U.col(0);
    u2 = U.col(1);
    u3 = U.col(2);
    w1 = W(0,0);
    w2 = W(1,0);
    n_est = w1 * u1;
    v_est = w2 * u2;

    //********* Own LM Optimizaton ************/
    // int runs = 1000;
    // LMOptimization(n_est, v_est, line_observations, cams, K_line, runs, errors);

    //********* Ceres Optimization ************/
    ceres::Problem problem;

    // Combine v_est and n_est into a single parameter block (Plücker line)
    double plucker[6] = {n_est.x(), n_est.y(), n_est.z(), v_est.x(), v_est.y(), v_est.z()};
    
    // Add the LineManifold to ensure the Plücker line constraints are respected
    problem.AddParameterBlock(plucker, 6, new ceres::LineManifold());
    
    // Add residual blocks for each observation
    for (size_t i = 0; i < line_observations.size(); ++i) {
        const auto& obs = line_observations[i];
        const Camera& cam = cams[i];
    
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<LineReprojectionError, 2, 6>(
                new LineReprojectionError(obs.first, obs.second, K, cam.R_wc, cam.p_wc)),
            nullptr,  // No loss function
            plucker);
    }
    
    // Configure the solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    // Output the results
    std::cout << summary.FullReport() << "\n";
    std::cout << "Optimized Plücker line (n | v): "
              << plucker[0] << " " << plucker[1] << " " << plucker[2] << " | "
              << plucker[3] << " " << plucker[4] << " " << plucker[5] << "\n";


    // std::ofstream error_file("/home/itadmin/MINS/catkin_ws/src/MINS/VIW_Fusion/Simulation/error_log.txt");
    // for (size_t i = 0; i < errors.size(); ++i) {
    //     error_file << i << " " << errors[i] << std::endl;
    // }
    // error_file.close();

    return 0;
}