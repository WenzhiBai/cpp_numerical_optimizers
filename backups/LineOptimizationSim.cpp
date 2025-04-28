#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <random>
#include <chrono> 
#include <fstream>
#include <unsupported/Eigen/MatrixFunctions>
using namespace std;
using namespace Eigen;

typedef Eigen::Matrix<double, 6, 1> Vector6d;

struct Camera {
    Matrix3d K;           // 内参
    Matrix3d R_wc;        // 世界到相机的旋转（world to cam）
    Vector3d p_wc;        // 相机在世界坐标系的位置
};

std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
std::normal_distribution<double> noise(0.0, 1.0);

Eigen::Matrix3d SkewSymmetric(const Eigen::Vector3d& v) {
    Eigen::Matrix3d skew;
    skew <<     0, -v.z(),  v.y(),
             v.z(),     0, -v.x(),
            -v.y(),  v.x(),     0;
    return skew;
}

// add observation noise
Vector2d AddNoise(const Vector2d& pt) {
    return Vector2d(pt(0) + noise(generator), pt(1) + noise(generator));
    // return pt;
}

// world line Pw to camera line Pc
Vector3d WorldToCamera(const Matrix3d& R_wc, const Vector3d& p_wc, const Vector3d& Pw) {
    return R_wc * (Pw - p_wc);  // T_cw * Pw
}

Vector2d ProjectPoint(const Matrix3d& K, const Matrix3d& R_wc, const Vector3d& p_wc, const Vector3d& Pw) {
    Vector3d Pc = WorldToCamera(R_wc, p_wc, Pw);
    Vector3d p = K * Pc;
    return Vector2d(p[0] / p[2], p[1] / p[2]);
}

// Plücker线段投影为图像中的直线 ax + by + c = 0
Vector3d ProjectPluckerToImageLine(const Matrix3d& K, const Matrix3d& R_wc, const Vector3d& p_wc, const Vector6d& plucker) {
    Vector3d n = plucker.head(3);
    Vector3d v = plucker.tail(3);

    // Vector3d t_cw = -R_wc * p_wc;
    // Vector3d n_cam = R_wc * n + SkewSymmetric(t_cw) * R_wc * v;
    Vector3d n_cam = R_wc * n - R_wc * SkewSymmetric(p_wc) * v;
    Vector3d v_cam = R_wc * v;

    return K * n_cam;
}


// 生成仿真相机轨迹
vector<Camera> GenerateCameras(int N, const Matrix3d& K) {
    vector<Camera> cams;
    for (int i = 0; i < N; ++i) {
        Camera cam;
        cam.K = K;

        double theta = 0.08 * i;
        Matrix3d R = AngleAxisd(theta, Vector3d::UnitY()).toRotationMatrix();
        Vector3d t(0.1 * i, 0.2 * i, 0.3 *i);

        cam.R_wc = R;
        cam.p_wc = t;
        cams.push_back(cam);
    }
    return cams;
}

double PointLineDistance(const Vector3d& line, const Vector2d& pt) {
    Eigen::Vector3d p_s_h(pt[0], pt[1], 1.0);
    double denom = std::sqrt(line[0]*line[0] + line[1]*line[1]);
    double d = p_s_h.dot(line) / denom;
    return std::abs(d);
}

double compute_error(const Eigen::Matrix3d &U, const Eigen::Matrix<double, 2, 2> &W, const vector<std::pair<Vector2d, Vector2d>> &line_observations,
    const vector<Camera> &cams, const Matrix3d &K_line) {
    double error = 0;

    // convert orthonomal to PLucker
    double w1, w2;
    Eigen::Vector3d u1, u2, n, v;
    u1 = U.col(0);
    u2 = U.col(1);
    w1 = W(0,0);
    w2 = W(1,0);
    n = w1 * u1;
    v = w2 * u2;
    Vector6d plucker;
    plucker.head<3>() = n;
    plucker.tail<3>() = v;
    // std::cout << "Plücker line (n | v): " << plucker.transpose() << std::endl;
    
    for (int i = 0; i < cams.size(); ++i) {
        Vector3d line = ProjectPluckerToImageLine(K_line, cams[i].R_wc, cams[i].p_wc, plucker);
        
        double e1 = PointLineDistance(line, line_observations[i].first);
        double e2 = PointLineDistance(line, line_observations[i].second);
        Eigen::Matrix<double, 2, 1> res;
        res << e1, e2;
        // Append to our summation variables
        error += pow(res.norm(), 2);
    }
    return error;
}

void UpdateOrthonomal(const Eigen::Matrix<double, 4, 1> &dx, Eigen::Matrix3d &U, Eigen::Matrix<double, 2, 2> &W, 
    Eigen::Matrix3d &U_temp, Eigen::Matrix<double, 2, 2> &W_temp) {
    Eigen::Vector3d du = dx.head<3>();
    double dw = dx(3);

    // update U
    Eigen::Matrix3d dU = Eigen::Matrix3d::Identity();
    double c1 = cos(du(0)), s1 = sin(du(0));
    double c2 = cos(du(1)), s2 = sin(du(1));
    double c3 = cos(du(2)), s3 = sin(du(2));

    Eigen::Matrix3d Rx;
    Rx << 1, 0, 0,
          0, c1, -s1,
          0, s1, c1;

    Eigen::Matrix3d Ry;
    Ry << c2, 0, -s2,
          0, 1, 0,
          s2, 0, c2;

    Eigen::Matrix3d Rz;
    Rz << c3, -s3, 0,
          s3, c3, 0,
          0, 0, 1;
    U_temp = Rx * Ry * Rz * U;
    // U_temp = U + U * SkewSymmetric(du);

    // update W
    Eigen::Matrix2d dW_matrix;
    dW_matrix << cos(dw), -sin(dw),
                 sin(dw), cos(dw);
    W_temp = dW_matrix * W;
    // dW_matrix << 0, -dw,
    //             dw, 0;
    // W_temp = W + W * dW_matrix;
}

void LMOptimization(const Vector3d &n, const Vector3d &v, const vector<std::pair<Vector2d, Vector2d>> &line_observations, 
    const vector<Camera> &cams, const Matrix3d &K_line, int runs, std::vector<double> &errors) {
    
    // Initilization
    cout << "The initial value before optimization is " << n.transpose() << v.transpose() << endl;
    Eigen::Matrix<double, 3, 2> A;
    A.col(0) = n;
    A.col(1) = v;
    Eigen::HouseholderQR<Eigen::Matrix<double, 3, 2>> qr(A);
    Eigen::Matrix3d U = qr.householderQ();
    Eigen::Matrix<double, 3, 2> R = qr.matrixQR().triangularView<Eigen::Upper>();
    
    double w1, w2;
    // w1 = R(0, 0) / std::sqrt(R(0, 0) * R(0, 0) + R(1, 1) * R(1, 1));
    // w2 = R(1, 1) / std::sqrt(R(0, 0) * R(0, 0) + R(1, 1) * R(1, 1));
    w1 = R(0, 0);
    w2 = R(1, 1);
    Eigen::Matrix<double, 2, 2> W;
    W << w1, -w2, w2, w1;
    
    // Optimization parameters
    double lam = 1e-4;
    double eps = 10000;
    int run_num = 0;

    // Variables used in the optimization
    bool recompute = true;
    Eigen::Matrix<double, 4, 4> Hess = Eigen::Matrix<double, 4, 4>::Zero();
    Eigen::Matrix<double, 4, 1> grad = Eigen::Matrix<double, 4, 1>::Zero();

    // Cost at the last iteration
    double cost_old = compute_error(U, W, line_observations, cams, K_line);
    errors.push_back(cost_old);
    cout << "Initial reprojection error (squared sum): " << cost_old << endl;
    
    //while (run_num < runs && lam < 1e10 && eps > 1e-6) {
    while (run_num < runs) {
        // Triggers a recomputation of jacobians/information/gradients
        if (recompute) {
            Hess.setZero();
            grad.setZero();
  
            double err = 0;

            // convert orthonomal to PLucker
            Eigen::Vector3d n_tmp, v_tmp, u1, u2, u3;
            u1 = U.col(0);
            u2 = U.col(1);
            u3 = U.col(2);
            w1 = W(0,0);
            w2 = W(1,0);
            n_tmp = u1 * w1;
            v_tmp = u2 * w2;
            Vector6d plucker;
            plucker.head<3>() = n_tmp;
            plucker.tail<3>() = v_tmp;
            // cout << "The value before is " << n_tmp.transpose() << v_tmp.transpose() << endl;

            // Loop through each camera for this feature
            for (int i = 0; i < cams.size(); ++i) {
                // Get the position of this clone in the global
                const Eigen::Matrix<double, 3, 3> R_wc = cams[i].R_wc;
                const Eigen::Matrix<double, 3, 1> p_wc = cams[i].p_wc;
                
                // Get the position of line in image frame
                Vector3d line_dist = ProjectPluckerToImageLine(K_line, R_wc, p_wc, plucker);
                
                // get the observation
                Vector3d uv_s, uv_e;
                uv_s << line_observations[i].first(0), line_observations[i].first(1), 1;
                uv_e << line_observations[i].second(0), line_observations[i].second(1), 1;

                //=========================================================================
                // Jacobian
                //=========================================================================
                // Observation error in respective to line 
                MatrixXd dz_l = MatrixXd::Identity(2, 3);
                double ln_2 = line_dist(0) * line_dist(0) + line_dist(1) * line_dist(1);
                dz_l(0, 0) = uv_s(0) - ((line_dist(0) * uv_s.transpose() * line_dist)(0) / (ln_2));
                dz_l(0, 1) = uv_s(1) - ((line_dist(1) * uv_s.transpose() * line_dist)(0) / (ln_2));
                dz_l(1, 0) = uv_e(0) - ((line_dist(0) * uv_e.transpose() * line_dist)(0) / (ln_2));
                dz_l(1, 1) = uv_e(1) - ((line_dist(1) * uv_e.transpose() * line_dist)(0) / (ln_2));
                dz_l *= 1 / sqrt(ln_2);

                // Compute Jacobians in respect to normalized image coordinates 
                MatrixXd dln_lc = MatrixXd::Zero(3, 6);
                dln_lc.block(0, 0, 3, 3) = K_line;

                // Nomalized observation in respect to cam frame 
                MatrixXd dlc_dlw = MatrixXd::Zero(6, 6);
                dlc_dlw.block(0, 0, 3, 3) = R_wc;
                dlc_dlw.block(0, 3, 3, 3) = -R_wc * SkewSymmetric(p_wc);
                dlc_dlw.block(3, 3, 3, 3) = R_wc;

                // Plucker representation to Orthogonal coodirnate
                MatrixXd dlc_dlo = MatrixXd::Zero(6, 4);
                dlc_dlo.block(0, 1, 3, 1) = - w1 * U.col(2);
                dlc_dlo.block(0, 2, 3, 1) = w1 * U.col(1);
                dlc_dlo.block(0, 3, 3, 1) = -w2 * U.col(0);
                dlc_dlo.block(3, 0, 3, 1) = w2 * U.col(2);
                dlc_dlo.block(3, 2, 3, 1) = -w2 * U.col(0);
                dlc_dlo.block(3, 3, 3, 1) = w1 * U.col(1);

                // whole joccobian
                Eigen::Matrix<double, 2, 4> H;
                H = dz_l * dln_lc * dlc_dlw * dlc_dlo;

                // calculate residual
                Eigen::VectorXd error_s, error_e;
            
                double e1  = PointLineDistance(line_dist, line_observations[i].first);
                double e2  = PointLineDistance(line_dist, line_observations[i].second);
                Eigen::Matrix<double, 2, 1> res;
                res << e1, e2;

                //=====================================================================================
                //=====================================================================================
                // cout << "Point line distance is " << res.norm() << endl;
                
                // Append to our summation variables
                err += std::pow(res.norm(), 2);
                grad.noalias() += H.transpose() * res.cast<double>();
                Hess.noalias() += H.transpose() * H;

                // whole joccobian (to plucker coordinate) 
                // Eigen::Matrix<double, 2, 6> H_p;
                // H_p = dz_l * dln_lc * dlc_dlw;
                // err += std::pow(res.norm(), 2);
                // grad.noalias() += H_p.transpose() * res.cast<double>();
                // Hess.noalias() += H_p.transpose() * H;
            }
        }

        // Solve Levenberg iteration
        Eigen::Matrix<double, 4, 4> Hess_l = Hess;
        for (size_t r = 0; r < (size_t)Hess_l.rows(); r++) {
            Hess_l(r, r) *= (1.0 + lam);
        }

        Eigen::Matrix<double, 4, 1> dx = Hess.colPivHouseholderQr().solve(grad);
        dx = 0.05 * dx;
        // cout << "The update value is " << dx.transpose() << endl;
        
        // Check if error has gone down
        Eigen::Matrix3d U_temp;
        Eigen::Matrix<double, 2, 2> W_temp;
        UpdateOrthonomal(dx, U, W, U_temp, W_temp);

        double cost = compute_error(U_temp, W_temp, line_observations, cams, K_line);

        // debug info 
        cout << "The " << run_num << "th iteration cost is " << cost << endl;
        errors.push_back(cost);

        // // Check if converged
        // if (cost <= cost_old && (cost_old - cost) / cost_old < 1e-6) {
        // // if (cost <= cost_old && cost < 1e-6) {
        //     // UpdateOrthonomal(dx, U, W, U, W);
        //     U = U_temp;
        //     W = W_temp;
        //     eps = 0;
        //     cout << "Optimization success " << endl;
        //     break;
        // }

        // If cost is lowered, accept step
        // Else inflate lambda (try to make more stable)

        // LM optimization
        // if (cost <= cost_old) {
        //     recompute = true;
        //     cost_old = cost;
        //     // UpdateOrthonomal(dx, U, W, U, W);
        //     U = U_temp;
        //     W = W_temp;
        //     run_num++;
        //     lam = lam / 10;
        //     eps = dx.norm();
        // } else {
        //     recompute = false;
        //     lam = lam * 10;
        //     continue;
        // }

        // Gasson Newdon 
        // if (dx.transpose() * dx < 1e-10) {
        //     U = U_temp;
        //     W = W_temp;
        //     eps = 0;
        //     cout << "Optimization success " << endl;
        //     break;
        // }
        U = U_temp;
        W = W_temp;
        run_num++;
    }

    cout << "The optimization time is " << run_num << endl;
    Eigen::Vector3d n_tmp, v_tmp, u1, u2, u3;
    u1 = U.col(0);
    u2 = U.col(1);
    u3 = U.col(2);
    w1 = W(0,0);
    w2 = W(1,0);
    n_tmp = w1 * u1;
    v_tmp = w2 * u2;
    cout << "The result after optimization re-define is " << n_tmp.transpose() << v_tmp.transpose() << endl;
}

int main() {
    // Camera intrinsic parameter
    Matrix3d K, K_line;
    K << 500, 0, 320,
         0, 500, 240,
         0,   0,   1;

    double fu = 500, fv = 500, cu = 320, cv = 240;
    K_line << fv,   0,    0,
            0,    fu,   0,
            -fv*cu, -fu*cv, fu*fv;

    std::vector<double> errors;
            
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
    
    // Optimization simulation
    // Initial Estimization (with error)
    // Vector3d v_est = v + Vector3d(0.01, -0.1, 0.1);
    // v_est.normalize();
    // Vector3d n_est = n + Vector3d(-0.1, -0.1, -0.1);
    

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

    //********* Own LM Optimizaton ************/7
    int runs = 1000;
    LMOptimization(n_est, v_est, line_observations, cams, K_line, runs, errors);

    std::ofstream error_file("/home/itadmin/MINS/catkin_ws/src/MINS/VIW_Fusion/Simulation/error_log.txt");
    for (size_t i = 0; i < errors.size(); ++i) {
        error_file << i << " " << errors[i] << std::endl;
    }
    error_file.close();

    //********* Ceres Slover ************/

    return 0;
}