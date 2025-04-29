#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <iostream>

struct Camera {
  Eigen::Matrix3d K;    // 内参
  Eigen::Matrix3d R_wc; // 世界到相机的旋转（world to cam）
  Eigen::Vector3d p_wc; // 相机在世界坐标系的位置
};

template <typename T>
Eigen::Matrix<T, 3, 3> SkewSymmetric(const Eigen::Matrix<T, 3, 1> &v) {
  Eigen::Matrix<T, 3, 3> skew;
  skew << T(0), -v.z(), v.y(), v.z(), T(0), -v.x(), -v.y(), v.x(), T(0);
  return skew;
}

template <typename T>
Eigen::Matrix<T, 3, 1>
ProjectPluckerToImageLine(const Eigen::Matrix3d &K, const Eigen::Matrix3d &R_wc,
                          const Eigen::Vector3d &p_wc,
                          const Eigen::Matrix<T, 3, 1> &n,
                          const Eigen::Matrix<T, 3, 1> &v) {
  Eigen::Matrix<T, 3, 3> skew;
  skew << T(0), -p_wc.cast<T>().z(), p_wc.cast<T>().y(), p_wc.cast<T>().z(),
      T(0), -p_wc.cast<T>().x(), -p_wc.cast<T>().y(), p_wc.cast<T>().x(), T(0);

  // Transform Plücker line to camera frame
  Eigen::Matrix<T, 3, 1> n_cam = R_wc.cast<T>() * n - R_wc.cast<T>() * skew * v;
  Eigen::Matrix<T, 3, 1> v_cam = R_wc.cast<T>() * v;

  // Project to image space
  Eigen::Matrix<T, 3, 1> line = K.cast<T>() * n_cam;
  return line;
}

template <typename T>
T PointLineDistance(const Eigen::Matrix<T, 3, 1> &line,
                    const Eigen::Matrix<T, 2, 1> &pt) {
  // Line equation: ax + by + c = 0
  T a = line[0];
  T b = line[1];
  T c = line[2];

  // Homogeneous point
  Eigen::Matrix<T, 3, 1> p_s_h(pt[0], pt[1], T(1.0));

  // Point-line distance formula
  T denom = sqrt(a * a + b * b);
  T distance = p_s_h.dot(line) / denom;
  return abs(distance);
}

// Define a simple cost function for optimization
struct LineReprojectionError {
  LineReprojectionError(const Eigen::Vector2d &p1_obs,
                        const Eigen::Vector2d &p2_obs, const Eigen::Matrix3d &K,
                        const Eigen::Matrix3d &R_wc,
                        const Eigen::Vector3d &p_wc)
      : p1_obs_(p1_obs), p2_obs_(p2_obs), K_(K), R_wc_(R_wc), p_wc_(p_wc) {}

  template <typename T>
  bool operator()(const T *const plucker, T *residuals) const {
    // Extract Plücker line components
    Eigen::Matrix<T, 3, 1> n(plucker[0], plucker[1], plucker[2]);
    Eigen::Matrix<T, 3, 1> v(plucker[3], plucker[4], plucker[5]);

    // Project Plücker line to image space
    Eigen::Matrix<T, 3, 1> line =
        ProjectPluckerToImageLine<T>(K_, R_wc_, p_wc_, n, v);

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

void UpdateOrthonomal(const Eigen::Matrix<double, 4, 1> &dx, Eigen::Matrix3d &U,
                      Eigen::Matrix<double, 2, 2> &W, Eigen::Matrix3d &U_temp,
                      Eigen::Matrix<double, 2, 2> &W_temp) {
  Eigen::Vector3d du = dx.head<3>();
  double dw = dx(3);

  // update U
  Eigen::Matrix3d dU = Eigen::Matrix3d::Identity();
  double c1 = cos(du(0)), s1 = sin(du(0));
  double c2 = cos(du(1)), s2 = sin(du(1));
  double c3 = cos(du(2)), s3 = sin(du(2));

  Eigen::Matrix3d Rx;
  Rx << 1, 0, 0, 0, c1, -s1, 0, s1, c1;

  Eigen::Matrix3d Ry;
  Ry << c2, 0, -s2, 0, 1, 0, s2, 0, c2;

  Eigen::Matrix3d Rz;
  Rz << c3, -s3, 0, s3, c3, 0, 0, 0, 1;
  U_temp = Rx * Ry * Rz * U;
  // U_temp = U + U * SkewSymmetric(du);

  // update W
  Eigen::Matrix2d dW_matrix;
  dW_matrix << cos(dw), -sin(dw), sin(dw), cos(dw);
  W_temp = dW_matrix * W;
  // dW_matrix << 0, -dw,
  //             dw, 0;
  // W_temp = W + W * dW_matrix;
}

std::vector<Camera> GenerateCameras(int N, const Eigen::Matrix3d &K) {
  std::vector<Camera> cams;
  for (int i = 0; i < N; ++i) {
    Camera cam;
    cam.K = K;

    double theta = 0.08 * i;
    Eigen::Matrix3d R =
        Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitY()).toRotationMatrix();
    Eigen::Vector3d t(0.1 * i, 0.2 * i, 0.3 * i);

    cam.R_wc = R;
    cam.p_wc = t;
    cams.push_back(cam);
  }
  return cams;
}

// world line Pw to camera line Pc
Eigen::Vector3d WorldToCamera(const Eigen::Matrix3d &R_wc,
                              const Eigen::Vector3d &p_wc,
                              const Eigen::Vector3d &Pw) {
  return R_wc * (Pw - p_wc); // T_cw * Pw
}

Eigen::Vector2d ProjectPoint(const Eigen::Matrix3d &K,
                             const Eigen::Matrix3d &R_wc,
                             const Eigen::Vector3d &p_wc,
                             const Eigen::Vector3d &Pw) {
  Eigen::Vector3d Pc = WorldToCamera(R_wc, p_wc, Pw);
  Eigen::Vector3d p = K * Pc;
  return Eigen::Vector2d(p[0] / p[2], p[1] / p[2]);
}

Eigen::Matrix<double, 6, 1> ComputePluckerLine(const Eigen::Vector3d &P1,
                                               const Eigen::Vector3d &P2) {
  Eigen::Matrix<double, 6, 1> plucker;

  // Direction vector (v): normalized vector from P1 to P2
  Eigen::Vector3d v = P2 - P1;

  // Moment vector (n): cross product of P1 and v
  Eigen::Vector3d n = P1.cross(P2);

  // Combine n and v into the Plücker line representation
  plucker.head<3>() = n;
  plucker.tail<3>() = v;

  return plucker;
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  // Camera intrinsic parameters
  double fu = 500, fv = 500, cu = 320, cv = 240;
  Eigen::Matrix3d K, K_line;
  K << fu, 0, cu, 0, fv, cv, 0, 0, 1;
  K_line << fv, 0, 0, 0, fu, 0, -fv * cu, -fu * cv, fu * fv;

  // Define a line in Plücker coordinates
  Eigen::Vector3d P1(-30, -20, 40), P2(10, 60, 80);
  Eigen::Matrix<double, 6, 1> plucker =
      ComputePluckerLine(P1, P2); // Plücker line representation
  Eigen::Vector3d v = plucker.tail<3>();
  Eigen::Vector3d n = plucker.head<3>();

  std::cout << "Initial Plücker line (n | v): " << plucker.transpose()
            << std::endl;

  // Generate camera poses
  int num_cams = 10;
  std::vector<Camera> cams = GenerateCameras(num_cams, K);
  std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> line_observations;

  // Compute initial reprojection errors
  double total_error = 0;
  for (int i = 0; i < num_cams; ++i) {
    Eigen::Vector2d p1_proj = ProjectPoint(K, cams[i].R_wc, cams[i].p_wc, P1);
    Eigen::Vector2d p2_proj = ProjectPoint(K, cams[i].R_wc, cams[i].p_wc, P2);

    Eigen::Vector2d p1_obs = p1_proj; // Simulated observation
    Eigen::Vector2d p2_obs = p2_proj;

    line_observations.emplace_back(p1_obs, p2_obs);

    Eigen::Matrix<double, 3, 1> line = ProjectPluckerToImageLine<double>(
        K_line, cams[i].R_wc, cams[i].p_wc, n, v);

    double e1 = PointLineDistance(line, p1_obs);
    double e2 = PointLineDistance(line, p2_obs);

    total_error += e1 * e1 + e2 * e2;
  }

  std::cout << "Initial total reprojection error (squared sum): " << total_error
            << std::endl;

  // Initialize optimization variables
  Eigen::Matrix<double, 3, 2> A;
  A.col(0) = n;
  A.col(1) = v;
  Eigen::HouseholderQR<Eigen::Matrix<double, 3, 2>> qr(A);
  Eigen::Matrix3d U = qr.householderQ();
  Eigen::Matrix<double, 3, 2> R = qr.matrixQR().triangularView<Eigen::Upper>();
  double w1 = R(0, 0), w2 = R(1, 1);
  Eigen::Matrix<double, 2, 2> W;
  W << w1, -w2, w2, w1;

  Eigen::Matrix<double, 4, 1> dx;
  dx << 0.001, 0.002, 0.002, 0.001;

  UpdateOrthonomal(dx, U, W, U, W);

  Eigen::Vector3d v_est, n_est;
  n_est = W(0, 0) * U.col(0);
  v_est = W(1, 0) * U.col(1);

  //********* Ceres Optimization ************/
  ceres::Problem problem;

  // Combine v_est and n_est into a single parameter block (Plücker line)
  double plucker_params[6] = {n_est.x(), n_est.y(), n_est.z(),
                              v_est.x(), v_est.y(), v_est.z()};

  // Add the LineManifold to ensure the Plücker line constraints are respected
  problem.AddParameterBlock(plucker_params, 6, new ceres::LineManifold<3>());

  // Add residual blocks for each observation
  for (size_t i = 0; i < line_observations.size(); ++i) {
    const auto &obs = line_observations[i];
    const Camera &cam = cams[i];

    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<LineReprojectionError, 2, 6>(
            new LineReprojectionError(obs.first, obs.second, K_line, cam.R_wc,
                                      cam.p_wc)),
        nullptr, // No loss function
        plucker_params);
  }

  // Configure the solver
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 1000;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // Output the results
  std::cout << summary.FullReport() << "\n";
  std::cout << "Optimized Plücker line (n | v): " << plucker_params[0] << " "
            << plucker_params[1] << " " << plucker_params[2] << " | "
            << plucker_params[3] << " " << plucker_params[4] << " "
            << plucker_params[5] << "\n";

  return 0;
}