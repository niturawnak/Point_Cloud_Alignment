                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            #include "../nanoflann.hpp"
#include "../nanoflann.hpp"
#include <numeric>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <Eigen/Dense>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "PC_ReaderWriter.h"

using namespace Eigen;
using namespace std;
using namespace nanoflann;
using namespace cv;


typedef KDTreeEigenMatrixAdaptor<MatrixXd> my_kd_tree_t;
vector<int> indices;
vector<double> distances;


// template <typename Der>
// void generateRandomPointCloud(Eigen::MatrixBase<Der> &mat, const size_t N,
//                               const size_t dim,
//                               const typename Der::Scalar max_range = 10) {
//   std::cout << "Generating " << N << " random points...";
//   mat.resize(N, dim);
//   for (size_t i = 0; i < N; i++)
//     for (size_t d = 0; d < dim; d++)
//       mat(i, d) = max_range * (rand() % 1000) / typename Der::Scalar(1000);
//   std::cout << "done\n";
// }

void PointCloudtoMatrix(MatrixXd &mat1, MatrixXd &mat2, PC_ReaderWriter& pc1, PC_ReaderWriter& pc2) {

    mat1.resize(pc1.no_of_rows, pc1.no_of_cols);
    for (size_t i = 0; i < pc1.no_of_rows; i++)
        for (size_t d = 0; d < pc1.no_of_cols; d++)
            mat1(i, d) = pc1.points[pc1.no_of_cols * i + d];

    mat2.resize(pc2.no_of_rows, pc2.no_of_cols);
    for (size_t i = 0; i < pc2.no_of_rows; i++)
        for (size_t d = 0; d < pc2.no_of_cols; d++)
            mat2(i, d) = pc2.points[pc2.no_of_cols * i + d];
}


void Get_Nearest_Neighbour(Mat mat1, const my_kd_tree_t& mat_index, const size_t no_of_rows, const size_t no_of_cols) {
  indices.clear();
  distances.clear();
  for (int i = 0; i < no_of_rows; ++i) {
    // Query point:
    vector<double> query_pt(no_of_cols);
    for (size_t d = 0; d < no_of_cols; d++) {
        query_pt[d] = mat1.at<double>(i, d);
    }
    // do a knn search
    size_t ret_index;
    double out_dist_sqr;

    nanoflann::KNNResultSet<double> resultSet(1);
    resultSet.init(&ret_index, &out_dist_sqr);

    mat_index.index->findNeighbors(resultSet, &query_pt[0],nanoflann::SearchParams(10));

    indices.push_back(int(ret_index));
    distances.push_back(out_dist_sqr);
  }
}


typedef struct Transformations {
    Mat rotation;
    Mat translation;
} Transformations ;


Transformations best_fit_transform(const Mat& pointcloud_1, const Mat& pointcloud_2) {
    Point3d center1, center2;
    Mat R, T;
    R = Mat::zeros(3, 3, CV_64F);
    T = Mat::zeros(3, 1, CV_64F);

    Mat covariance = Mat::zeros(3, 3, CV_64F);
    vector<Point3d> cent_pcl1, cent_pcl2;
    Transformations TR;

    int pcl1_rows = pointcloud_1.rows, pcl2_rows = pointcloud_2.rows;
    
    // Calculate the center of pcl1
    for (int i = 0; i < pcl1_rows; i++) {
        center1.x += pointcloud_1.at<double>(i, 0);
        center1.y += pointcloud_1.at<double>(i, 1);
        center1.z += pointcloud_1.at<double>(i, 2);
    }
    center1 = center1 * (1.0 / pcl1_rows);

    // Calculate the center of pcl2
    for (int i = 0; i < pcl2_rows; i++) {
        center2.x += pointcloud_2.at<double>(i, 0);
        center2.y += pointcloud_2.at<double>(i, 1);
        center2.z += pointcloud_2.at<double>(i, 2);
    }
    center2 = center2 * (1.0 / pcl2_rows);

    // Move pcl1 to the center
    for (int i = 0; i < pcl1_rows; i++) {
        Point3d pt;
        pt.x = pointcloud_1.at<double>(i, 0) - center1.x;
        pt.y = pointcloud_1.at<double>(i, 1) - center1.y;
        pt.z = pointcloud_1.at<double>(i, 2) - center1.z;
        cent_pcl1.emplace_back(pt);
    }

    // Move pcl2 to the center
    for (int i = 0; i < pcl2_rows; i++) {
        Point3d pt;
        pt.x = pointcloud_2.at<double>(i, 0) - center2.x;
        pt.y = pointcloud_2.at<double>(i, 1) - center2.y;
        pt.z = pointcloud_2.at<double>(i, 2) - center2.z;
        cent_pcl2.emplace_back(pt);
    }

    // Calculate covariance matrix
    for (int i = 0; i < pcl1_rows; i++) {
        covariance.at<double>(0, 0) += cent_pcl1[i].x * cent_pcl2[i].x;
        covariance.at<double>(0, 1) += cent_pcl1[i].x * cent_pcl2[i].y;
        covariance.at<double>(0, 2) += cent_pcl1[i].x * cent_pcl2[i].z;
        covariance.at<double>(1, 0) += cent_pcl1[i].y * cent_pcl2[i].x;
        covariance.at<double>(1, 1) += cent_pcl1[i].y * cent_pcl2[i].y;
        covariance.at<double>(1, 2) += cent_pcl1[i].y * cent_pcl2[i].z;
        covariance.at<double>(2, 0) += cent_pcl1[i].z * cent_pcl2[i].x;
        covariance.at<double>(2, 1) += cent_pcl1[i].z * cent_pcl2[i].y;
        covariance.at<double>(2, 2) += cent_pcl1[i].z * cent_pcl2[i].z;
    }
    covariance /= pcl1_rows;

    Mat w, u, vt;
    SVD::compute(covariance, w, u, vt, 0);

    // Calculate the rotation matrix
    R = vt.t() * u.t();
    if (determinant(R) < 0.) {
        vt.at<double>(2, 0) *= -1;
        vt.at<double>(2, 1) *= -1;
        vt.at<double>(2, 2) *= -1;
        R = vt.t() * u.t();
    }

    Mat centerOfMass1 = Mat::zeros(3, 1, CV_64F); 
    centerOfMass1.at<double>(0, 0) = center1.x;
    centerOfMass1.at<double>(1, 0) = center1.y;
    centerOfMass1.at<double>(2, 0) = center1.z;

    Mat centerOfMass2 = Mat::zeros(3, 1, CV_64F);
    centerOfMass2.at<double>(0, 0) = center2.x;
    centerOfMass2.at<double>(1, 0) = center2.y;
    centerOfMass2.at<double>(2, 0) = center2.z;

    // Calculate the translation matrix
    T = centerOfMass2 - (R * centerOfMass1);

    TR.rotation = R;
    TR.translation = T;
    return TR;
}


double calculateError(Mat mat1, Mat mat2, Mat rot, Mat transl) {

    double error = 0;

    for (int i = 0; i < mat2.rows ; i++) {

        Mat point1 = Mat::zeros(3, 1, CV_64F);
        Mat point2 = Mat::zeros(3, 1, CV_64F);
        point1.at<double>(0, 0) = mat1.at<double>(i, 0);
        point1.at<double>(1, 0) = mat1.at<double>(i, 1);
        point1.at<double>(2, 0) = mat1.at<double>(i, 2);
        point2.at<double>(0, 0) = mat2.at<double>(i, 0);
        point2.at<double>(1, 0) = mat2.at<double>(i, 1);
        point2.at<double>(2, 0) = mat2.at<double>(i, 2);
        Mat diff = (rot * point1 + transl) - point2;
        error += sqrt(diff.at<double>(0, 0) * diff.at<double>(0, 0) + diff.at<double>(1, 0) * diff.at<double>(1, 0) + diff.at<double>(2, 0) * diff.at<double>(2, 0));
    }
    error = error / mat2.rows;
    return error;
}



void ICP(MatrixXd mat1, MatrixXd mat2, int max_iteration_num, Mat R_GT) {
    auto time_start = std::chrono::high_resolution_clock::now();
    Transformations T;
    Mat cvSource, cvTarget;
    eigen2cv(mat1, cvSource);
    eigen2cv(mat2, cvTarget);

    double prev_error = 0.0;
    double mean_error = 0.0;
    double tolerance = 0.00001;

    int max_I = max_iteration_num;

    my_kd_tree_t mat_index(3, std::cref(mat2), 10 /* max leaf */);
    mat_index.index->buildIndex();

    Mat rotation_matrix = Mat::eye(3, 3, CV_64F);
    Mat translation_matrix = Mat::eye(3, 1, CV_64F);
    Mat t_GT = Mat::eye(3, 1, CV_64F);

    for (int it = 0; it< max_I; it++) {

        Get_Nearest_Neighbour(cvSource, mat_index, cvSource.rows, 3);
        Mat cvNewTarget = Mat::zeros(cvSource.rows, 3, CV_64F);

        for (int i = 0; i < cvSource.rows; i++) {
            cvNewTarget.at<double>(i, 0) = cvTarget.at<double>(indices[i], 0);
            cvNewTarget.at<double>(i, 1) = cvTarget.at<double>(indices[i], 1);
            cvNewTarget.at<double>(i, 2) = cvTarget.at<double>(indices[i], 2);
        }
        // Compute motion that minimises mean square error(MSE) between paired points.
        T = best_fit_transform(cvSource, cvNewTarget); 
        rotation_matrix *= T.rotation;
        translation_matrix += T.translation;

        // Apply motion to P and update MSE.
        for (int i = 0; i < cvSource.rows; i++) {     
            Mat point = Mat::zeros(3, 1, CV_64F);
            point.at<double>(0, 0) = cvSource.at<double>(i, 0);
            point.at<double>(1, 0) = cvSource.at<double>(i, 1);
            point.at<double>(2, 0) = cvSource.at<double>(i, 2);
            point = T.rotation * point;
            point += T.translation;
            cvSource.at<double>(i, 0) = point.at<double>(0, 0);
            cvSource.at<double>(i, 1) = point.at<double>(1, 0);
            cvSource.at<double>(i, 2) = point.at<double>(2, 0);
        }

        // Updating MSE
        mean_error = calculateError(cvSource, cvNewTarget, T.rotation, T.translation); 
        cout << mean_error << endl;
        if (abs(prev_error - mean_error) < tolerance) {
            break;
        }
        prev_error = mean_error;
    }
    auto time_finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = (time_finish - time_start);
    cout << "Time = " << duration.count() << " s" << endl;
    cout << "Rotation matrix: " << endl << rotation_matrix << endl;
    cout << "Translation matrix: " << endl << translation_matrix << endl;
    cout << "r_error: " << std::acos((cv::trace(rotation_matrix.t() * R_GT)[0] - 1) / 2) * 180.0 / M_PI << std::endl;
    cout << "t_error: " << std::sqrt(norm((translation_matrix - t_GT), NORM_L2SQR)) << std::endl;




    ofstream file("/home/nitu/Desktop/3D_Sensing/Assignment03/data/fountain_ICP.xyz");
    //saving output file
    if (file.is_open())
    {
        for (int i = 0; i < cvSource.rows; ++i) {
            file << cvSource.at<double>(i, 0) << " " << cvSource.at<double>(i, 1) << " " << cvSource.at<double>(i, 2) << endl;
        }
        for (int i = 0; i < cvTarget.rows; ++i) {
            file << cvTarget.at<double>(i, 0) << " " << cvTarget.at<double>(i, 1) << " " << cvTarget.at<double>(i, 2) << endl;
        }
    }
}

vector<tuple<int, int, double>> Indices_Distances_pair;

double calculateError_trICP(int NPo, vector<tuple<int, int, double>> Indices_Distances_pair) {
    double trimmedMSE = 0.;
    int length = NPo;
    for (int i = 0; i < length; i++) {
        trimmedMSE += get<2>(Indices_Distances_pair[i]);
    }
    trimmedMSE /= length;
    return trimmedMSE;
}

bool sortbythird(const tuple<int, int, double>& a, const tuple<int, int, double>& b){
    return (get<2>(a) < get<2>(b));
}


void TrICP(MatrixXd mat1, MatrixXd mat2, int max_iteration_num, Mat R_GT) {
    auto time_start = std::chrono::high_resolution_clock::now();
    Transformations  T;
    Mat cvSource, cvTarget;
    eigen2cv(mat1, cvSource);
    eigen2cv(mat2, cvTarget);
    double prev_error = 0.0;
    double mean_error = 0.0;
    double tolerance = 0.0001;
    int max_I= max_iteration_num;//100;
    my_kd_tree_t mat_index(3, std::cref(mat2), 10 /* max leaf */);

    Mat rotation_matrix = Mat::eye(3, 3, CV_64F);
    Mat translation_matrix = Mat::eye(3, 1, CV_64F);
    Mat t_GT = Mat::eye(3, 1, CV_64F);


    mat_index.index->buildIndex();

    for (int it = 0; it< max_I; it++) {
        Indices_Distances_pair.clear();
        Get_Nearest_Neighbour(cvSource, mat_index, cvSource.rows, 3); 

        for (int i = 0; i < cvSource.rows; i++) {
            Indices_Distances_pair.push_back(make_tuple(i, indices[i], distances[i]));
        }

        sort(Indices_Distances_pair.begin(), Indices_Distances_pair.end(), sortbythird);
        sort(distances.begin(), distances.end());
        int NPo = 0.6 * double(Indices_Distances_pair.size());
        Mat cvNewSource = Mat::zeros(NPo, 3, CV_64F);
        Mat cvNewTarget = Mat::zeros(NPo, 3, CV_64F);
        for (int i = 0; i < NPo; i++) {
            cvNewSource.at<double>(i, 0) = cvSource.at<double>(get<0>(Indices_Distances_pair[i]), 0);
            cvNewSource.at<double>(i, 1) = cvSource.at<double>(get<0>(Indices_Distances_pair[i]), 1);
            cvNewSource.at<double>(i, 2) = cvSource.at<double>(get<0>(Indices_Distances_pair[i]), 2);
            cvNewTarget.at<double>(i, 0) = cvTarget.at<double>(get<1>(Indices_Distances_pair[i]), 0);
            cvNewTarget.at<double>(i, 1) = cvTarget.at<double>(get<1>(Indices_Distances_pair[i]), 1);
            cvNewTarget.at<double>(i, 2) = cvTarget.at<double>(get<1>(Indices_Distances_pair[i]), 2);
        }

        T = best_fit_transform(cvNewSource, cvNewTarget); // For !!!! Npo selected pairs !!!!, compute optimal motion(R, t) that minimises STS
        rotation_matrix *= T.rotation;
        translation_matrix += T.translation;

        for (int i = 0; i < cvSource.rows; i++) {     // Apply motion to P and update MSE.

            Mat pont = Mat::zeros(3, 1, CV_64F);
            pont.at<double>(0, 0) = cvSource.at<double>(i, 0);
            pont.at<double>(1, 0) = cvSource.at<double>(i, 1);
            pont.at<double>(2, 0) = cvSource.at<double>(i, 2);
            pont = T.rotation * pont;
            pont += T.translation;
            cvSource.at<double>(i, 0) = pont.at<double>(0, 0);
            cvSource.at<double>(i, 1) = pont.at<double>(1, 0);
            cvSource.at<double>(i, 2) = pont.at<double>(2, 0);
        }
        // mean_error = calculateError(cvSource, cvNewTarget, T.rot, T.transl); // Updating MSE
        mean_error = calculateError_trICP(NPo, Indices_Distances_pair); // Updating MSE
        cout << "MSE: " << mean_error << endl;

        if (abs(prev_error - mean_error) < tolerance) {
            break;
        }
        prev_error = mean_error;
    }
    auto time_finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = (time_finish - time_start);
    cout << "Time = " << duration.count() << " s" << endl;
    cout << "Rotation matrix: " << endl << rotation_matrix << endl;
    cout << "Translation matrix: " << endl << translation_matrix << endl;
    cout << "r_error_trimcp: " << std::acos((cv::trace(rotation_matrix.t() * R_GT)[0] - 1) / 2) * 180.0 / M_PI << std::endl;
    cout << "t_error_trimcp: " << std::sqrt(norm((translation_matrix - t_GT), NORM_L2SQR)) << std::endl;


    // out << "t_error: " << std::sqrt((t_total - t_gt).squaredNorm()) << std::endl;
    ofstream file("/home/nitu/Desktop/3D_Sensing/Assignment03/data/fountain_trICP.xyz");
    if (file.is_open())
    {
        for (int i = 0; i < cvSource.rows; ++i) {
            file << cvSource.at<double>(i, 0) << " " << cvSource.at<double>(i, 1) << " " << cvSource.at<double>(i, 2) << endl;
        }
        for (int i = 0; i < cvTarget.rows; ++i) {
            file << cvTarget.at<double>(i, 0) << " " << cvTarget.at<double>(i, 1) << " " << cvTarget.at<double>(i, 2) << endl;
        }
    }
}

Matrix3d find_rotation_matrix_from_angle(vector<double>& theta)
{
    Matrix3d R_x = Matrix3d::Identity(3, 3);
    Matrix3d R_y = Matrix3d::Identity(3, 3);
    Matrix3d R_z = Matrix3d::Identity(3, 3);
    R_x <<
        1, 0, 0,
        0, cos(theta[0]), -sin(theta[0]),
        0, sin(theta[0]), cos(theta[0]);
    R_y <<
        cos(theta[1]), 0, sin(theta[1]),
        0, 1, 0,
        -sin(theta[1]), 0, cos(theta[1]);
    R_z <<
        cos(theta[2]), -sin(theta[2]), 0,
        sin(theta[2]), cos(theta[2]), 0,
        0, 0, 1;
    return R_z * R_y * R_x;
}


#define PI 3.14159265

int main(int argc, char **argv) {
  // Randomize Seed
  srand(static_cast<unsigned int>(time(nullptr)));

  if (argc < 4) {
      std::cerr << "Usage: " << argv[0] << " PC1 PC2 MAX_I ROTATION_ANGLE" << std::endl;
      return 1;
  }
  PC_ReaderWriter* pc1;
  pc1 = new PC_ReaderWriter(argv[1]);
  PC_ReaderWriter* pc2;
  pc2 = new PC_ReaderWriter(argv[2]);

  MatrixXd mat1(pc1->no_of_rows, pc1->no_of_cols);
  MatrixXd mat2(pc2->no_of_rows, pc2->no_of_cols);
  
  PointCloudtoMatrix(mat1, mat2, *pc1, *pc2);


  int max_iteration= atoi(argv[3]);
  float angle_to_rotate = atof(argv[4]);

  //std::cout << mat1 <<endl;

  Mat initial_rotation_matrix = Mat::eye(3, 3, CV_64F);

  vector<double> degrees = { 0, 0, angle_to_rotate * PI / 180.0 };
  
  Matrix3d initialRotation = find_rotation_matrix_from_angle(degrees);
  cout << initialRotation;

  eigen2cv(initialRotation, initial_rotation_matrix);

  //Applying Rotation on Point cloud 2
  #pragma omp parallel for
  for (int i = 0; i < mat2.rows(); i++) {      
     mat2.block<1, 3>(i, 0).transpose() << initialRotation * mat2.block<1, 3>(i, 0).transpose();
  }

  ofstream file("/home/nitu/Desktop/3D_Sensing/Assignment03/data/fountain_rotation_0.xyz");
  if (file.is_open())
  {
     file << mat2 << '\n';
  }
  // Iterative Closest Point Algorithm
  ICP(mat1, mat2, max_iteration, initial_rotation_matrix);

  // Trimmed Iterative Closest Point Algorithm
  TrICP(mat1, mat2, max_iteration, initial_rotation_matrix); // mat1 is data, mat2 is the model

  // kdtree_demo<float>(1000 /* samples */, SAMPLES_DIM /* dim */);

  // circle_demo();

  return 0;
}







