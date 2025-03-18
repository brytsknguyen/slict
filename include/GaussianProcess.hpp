#pragma once

#include <iostream>
#include <fstream>
#include <filesystem>

#include <algorithm>    // Include this header for std::max
#include <Eigen/Dense>

// Sophus
#include <sophus/se3.hpp>

typedef Sophus::SO3<double> SO3d;
typedef Sophus::SE3<double> SE3d;
typedef Vector3d Vec3;
typedef Matrix3d Mat3;

using namespace std;
using namespace Eigen;


/* #region Define the states for convenience in initialization and copying ------------------------------------------*/

#define STATE_DIM 18
template <class T = double>
class GPState
{
public:

    using SO3T  = Sophus::SO3<T>;
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Mat3T = Eigen::Matrix<T, 3, 3>;

    double t;
    SO3T  R;
    Vec3T O;
    Vec3T S;
    Vec3T P;
    Vec3T V;
    Vec3T A;

    // Destructor
    ~GPState(){};
    
    // Constructor
    GPState()
        : t(0), R(SO3T()), O(Vec3T(0, 0, 0)), S(Vec3T(0, 0, 0)), P(Vec3T(0, 0, 0)), V(Vec3T(0, 0, 0)), A(Vec3T(0, 0, 0)) {}
    
    GPState(double t_)
        : t(t_), R(SO3T()), O(Vec3T()), S(Vec3T()), P(Vec3T()), V(Vec3T()), A(Vec3T()) {}

    GPState(double t_, const SE3d &pose)
        : t(t_), R(pose.so3().cast<T>()), O(Vec3T(0, 0, 0)), S(Vec3T(0, 0, 0)), P(pose.translation().cast<T>()), V(Vec3T(0, 0, 0)), A(Vec3T(0, 0, 0)) {}

    GPState(double t_, const SO3d &R_, const Vec3 &O_, const Vec3 &S_, const Vec3 &P_, const Vec3 &V_, const Vec3 &A_)
        : t(t_), R(R_.cast<T>()), O(O_.cast<T>()), S(S_.cast<T>()), P(P_.cast<T>()), V(V_.cast<T>()), A(A_.cast<T>()) {}

    GPState(const GPState<T> &other)
        : t(other.t), R(other.R), O(other.O), S(other.S), P(other.P), V(other.V), A(other.A) {}

    GPState(double t_, const GPState<T> &other)
        : t(t_), R(other.R), O(other.O), S(other.S), P(other.P), V(other.V), A(other.A) {}
    
    GPState &operator=(const GPState &Xother)
    {
        this->t = Xother.t;
        this->R = Xother.R;
        this->O = Xother.O;
        this->S = Xother.S;
        this->P = Xother.P;
        this->V = Xother.V;
        this->A = Xother.A;
        return *this;
    }

    Matrix<double, STATE_DIM, 1> boxminus(const GPState &Xother) const
    {
        Matrix<double, STATE_DIM, 1> dX;
        dX << (Xother.R.inverse()*R).log(),
               O - Xother.O,
               S - Xother.S,
               P - Xother.P,
               V - Xother.V,
               A - Xother.A;
        return dX;
    }

    double yaw()
    {
        Eigen::Vector3d n = R.matrix().col(0);
        Eigen::Vector3d ypr(3);
        double y = atan2(n(1), n(0));
        return y / M_PI * 180.0;
    }

    double pitch()
    {
        Eigen::Vector3d n = R.matrix().col(0);
        Eigen::Vector3d ypr(3);
        double y = atan2(n(1), n(0));
        double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
        return p / M_PI * 180.0;
    }

    double roll()
    {
        Eigen::Vector3d n = R.matrix().col(0);
        Eigen::Vector3d o = R.matrix().col(1);
        Eigen::Vector3d a = R.matrix().col(2);
        Eigen::Vector3d ypr(3);
        double y = atan2(n(1), n(0));
        double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
        double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
        return r / M_PI * 180.0;
    }
};

/* #endregion Define the states for convenience in initialization and copying ---------------------------------------*/


/* #region Utility for propagation and interpolation matrices, elementary jacobians dXt/dXk, J_r, H_r, Hprime_r.. ---*/

class GPMixer
{
private:

    // Knot length
    double dt = 0.0;

    // identity matrix
    const Mat3 Eye = Mat3::Identity();

    // Covariance of angular jerk
    Mat3 SigGa = Eye;

    // Covariance of translational jerk
    Mat3 SigNu = Eye;

public:

    // Destructor
   ~GPMixer() {};

    // Constructor
    GPMixer(double dt_, const Mat3 SigGa_, const Mat3 SigNu_) : dt(dt_), SigGa(SigGa_), SigNu(SigNu_) {};

    double getDt() const { return dt; }
    Mat3   getSigGa() const { return SigGa; }
    Mat3   getSigNu() const { return SigNu; }

    template <typename MatrixType1, typename MatrixType2>
    MatrixXd kron(const MatrixType1& A, const MatrixType2& B) const
    {
        MatrixXd result(A.rows() * B.rows(), A.cols() * B.cols());
        for (int i = 0; i < A.rows(); ++i)
            for (int j = 0; j < A.cols(); ++j)
                result.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;

        return result;
    }

    void setSigGa(const Mat3 &m)
    {
        SigGa = m;
    }

    void setSigNu(const Mat3 &m)
    {
        SigNu = m;
    }

    // Transition Matrix, PHI(tau, 0)
    MatrixXd Fbase(const double dtau, int N) const
    {
        std::function<int(int)> factorial = [&factorial](int n) -> int {return (n <= 1) ? 1 : n * factorial(n - 1);};

        MatrixXd Phi = MatrixXd::Identity(N, N);
        for(int n = 0; n < N; n++)
            for(int m = n + 1; m < N; m++)
                Phi(n, m) = pow(dtau, m-n)/factorial(m-n);

        return Phi;
    }

    // Gaussian Process covariance, Q = \int{Phi*F*SigNu*F'*Phi'}
    MatrixXd Qbase(const double dtau, int N) const 
    {
        std::function<int(int)> factorial = [&factorial](int n) -> int {return (n <= 1) ? 1 : n * factorial(n - 1);};
        
        MatrixXd Q(N, N);
        for(int n = 0; n < N; n++)
            for(int m = 0; m < N; m++)
                Q(n, m) = pow(dtau, 2*N-1-n-m)/double(2*N-1-n-m)/double(factorial(N-1-n))/double(factorial(N-1-m));
        // cout << "MyQ: " << Q << endl;
        return Q;
    }

    MatrixXd Qga(const double s, int N) const 
    {
        double dtau = s*dt;

        std::function<int(int)> factorial = [&factorial](int n) -> int {return (n <= 1) ? 1 : n * factorial(n - 1);};
        
        MatrixXd Q(N, N);
        for(int n = 0; n < N; n++)
            for(int m = 0; m < N; m++)
                Q(n, m) = pow(dtau, 2*N-1-n-m)/double(2*N-1-n-m)/double(factorial(N-1-n))/double(factorial(N-1-m));

        return kron(Qbase(dt, 3), SigGa);
    }

    MatrixXd Qnu(const double s, int N) const 
    {
        double dtau = s*dt;

        std::function<int(int)> factorial = [&factorial](int n) -> int {return (n <= 1) ? 1 : n * factorial(n - 1);};
        
        MatrixXd Q(N, N);
        for(int n = 0; n < N; n++)
            for(int m = 0; m < N; m++)
                Q(n, m) = pow(dtau, 2*N-1-n-m)/double(2*N-1-n-m)/double(factorial(N-1-n))/double(factorial(N-1-m));

        return kron(Qbase(dt, 3), SigNu);
    }

    Matrix<double, STATE_DIM, STATE_DIM> PropagateFullCov(Matrix<double, STATE_DIM, STATE_DIM> P0) const
    {
        Matrix<double, STATE_DIM, STATE_DIM> F; F.setZero();
        Matrix<double, STATE_DIM, STATE_DIM> Q; Q.setZero();
        
        F.block<9, 9>(0, 0) = kron(Fbase(dt, 3), Eye);
        F.block<9, 9>(9, 9) = kron(Fbase(dt, 3), Eye);

        Q.block<9, 9>(0, 0) = kron(Qbase(dt, 3), SigGa);
        Q.block<9, 9>(9, 9) = kron(Qbase(dt, 3), SigNu);

        return F*P0*F.transpose() + Q;
    }

    MatrixXd PSI(const double dtau, const Mat3 &Q) const
    {
        if (dtau < 1e-4)
            return kron(MatrixXd::Zero(3, 3), Eye);

        MatrixXd Phidtaubar = kron(Fbase(dt - dtau, 3), Eye);
        MatrixXd Qdtau = kron(Qbase(dtau, 3), Q);
        MatrixXd Qdt = kron(Qbase(dt, 3), Q);

        return Qdtau*Phidtaubar.transpose()*Qdt.inverse();
    }

    MatrixXd PSI_ROS(const double dtau) const
    {
        return PSI(dtau, SigGa);
    }

    MatrixXd PSI_PVA(const double dtau) const
    {
        return PSI(dtau, SigNu);
    }

    MatrixXd LAMDA(const double dtau, const Mat3 &Q) const
    {
        MatrixXd PSIdtau = PSI(dtau, Q);
        MatrixXd Fdtau = kron(Fbase(dtau, 3), Eye);
        MatrixXd Fdt = kron(Fbase(dt, 3), Eye);

        return Fdtau - PSIdtau*Fdt;
    }

    MatrixXd LAMDA_ROS(const double dtau) const
    {
        return LAMDA(dtau, SigGa);
    }

    MatrixXd LAMDA_PVA(const double dtau) const
    {
        return LAMDA(dtau, SigNu);
    }

    template <class T = double>
    static Eigen::Matrix<T, 3, 3> Jr(const Eigen::Matrix<T, 3, 1> &phi)
    {
        if (phi.norm() < 1e-4)
            return Eigen::Matrix<T, 3, 3>::Identity() - Sophus::SO3<T>::hat(phi);

        Eigen::Matrix<T, 3, 3> Jr;
        Sophus::rightJacobianSO3(phi, Jr);
        return Jr;
    }

    template <class T = double>
    static Eigen::Matrix<T, 3, 3> JrInv(const Eigen::Matrix<T, 3, 1> &phi)
    {
        if (phi.norm() < 1e-4)
            return Eigen::Matrix<T, 3, 3>::Identity() + Sophus::SO3<T>::hat(phi);

        Eigen::Matrix<T, 3, 3> JrInv;
        Sophus::rightJacobianInvSO3(phi, JrInv);
        return JrInv;
    }

    template <class T = double>
    void MapParamToState(T const *const *parameters, int base, GPState<T> &X) const
    {
        X.R = Eigen::Map<Sophus::SO3<T> const>(parameters[base + 0]);
        X.O = Eigen::Map<Eigen::Matrix<T, 3, 1> const>(parameters[base + 1]);
        X.S = Eigen::Map<Eigen::Matrix<T, 3, 1> const>(parameters[base + 2]);
        X.P = Eigen::Map<Eigen::Matrix<T, 3, 1> const>(parameters[base + 3]);
        X.V = Eigen::Map<Eigen::Matrix<T, 3, 1> const>(parameters[base + 4]);
        X.A = Eigen::Map<Eigen::Matrix<T, 3, 1> const>(parameters[base + 5]);
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> Fu(const Eigen::Matrix<T, 3, 1> &U, const Eigen::Matrix<T, 3, 1> &V) const
    {
        // Extract the elements of input
        T ux = U(0); T uy = U(1); T uz = U(2);
        T vx = V(0); T vy = V(1); T vz = V(2);

        Eigen::Matrix<T, 3, 3> Fu_;
        Fu_ << uy*vy +     uz*vz, ux*vy - 2.0*uy*vx, ux*vz - 2.0*uz*vx,
               uy*vx - 2.0*ux*vy, ux*vx +     uz*vz, uy*vz - 2.0*uz*vy,
               uz*vx - 2.0*ux*vz, uz*vy - 2.0*uy*vz, ux*vx +     uy*vy;
        return Fu_; 
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> Fv(const Eigen::Matrix<T, 3, 1> &U, const Eigen::Matrix<T, 3, 1> &V) const
    {
        return Sophus::SO3<T>::hat(U)*Sophus::SO3<T>::hat(U);
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> Fuu(const Eigen::Matrix<T, 3, 1> &U, const Eigen::Matrix<T, 3, 1> &V, const Eigen::Matrix<T, 3, 1> &A) const
    {
        // Extract the elements of input
        // T ux = U(0); T uy = U(1); T uz = U(2);
        T vx = V(0); T vy = V(1); T vz = V(2);
        T ax = A(0); T ay = A(1); T az = A(2);

        Eigen::Matrix<T, 3, 3> Fuu_;
        Fuu_ << ay*vy +     az*vz, ax*vy - 2.0*ay*vx, ax*vz - 2.0*az*vx,
                ay*vx - 2.0*ax*vy, ax*vx +     az*vz, ay*vz - 2.0*az*vy,
                az*vx - 2.0*ax*vz, az*vy - 2.0*ay*vz, ax*vx +     ay*vy; 
        return Fuu_; 
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> Fuv(const Eigen::Matrix<T, 3, 1> &U, const Eigen::Matrix<T, 3, 1> &V, const Eigen::Matrix<T, 3, 1> &A) const
    {
        // Extract the elements of input
        T ux = U(0); T uy = U(1); T uz = U(2);
        // T vx = V(0); T vy = V(1); T vz = V(2);
        T ax = A(0); T ay = A(1); T az = A(2);

        Eigen::Matrix<T, 3, 3> Fuv_;
        Fuv_ << -2.0*ay*uy - 2.0*az*uz,      ax*uy +     ay*ux,      ax*uz +     az*ux,
                     ax*uy +     ay*ux, -2.0*ax*ux - 2.0*az*uz,      ay*uz +     az*uy,
                     ax*uz +     az*ux,      ay*uz +     az*uy, -2.0*ax*ux - 2.0*ay*uy;
        return Fuv_; 
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> DJrXV_DX(const Eigen::Matrix<T, 3, 1> &X, const Eigen::Matrix<T, 3, 1> &V) const
    {
        using SO3T  = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        T Xn = X.norm();

        if(Xn < 1e-4)
            return 0.5*SO3T::hat(V);

        T Xnp2 = Xn*Xn;
        T Xnp3 = Xnp2*Xn;
        T Xnp4 = Xnp3*Xn;

        T sXn = sin(Xn);
        // T sXnp2 = sXn*sXn;
        
        T cXn = cos(Xn);
        // T cXnp2 = cXn*cXn;
        
        T gXn = (1.0 - cXn)/Xnp2;
        T DgXn_DXn = sXn/Xnp2 - 2.0*(1.0 - cXn)/Xnp3;

        T hXn = (Xn - sXn)/Xnp3;
        T DhXn_DXn = (1.0 - cXn)/Xnp3 - 3.0*(Xn - sXn)/Xnp4;

        Vec3T Xb = X/Xn;
        
        Vec3T XsksqV = SO3T::hat(X)*SO3T::hat(X)*V;
        Mat3T DXsksqV_DX = Fu<T>(X, V);

        return SO3T::hat(V)*gXn + SO3T::hat(V)*X*DgXn_DXn*Xb.transpose() + DXsksqV_DX*hXn + XsksqV*DhXn_DXn*Xb.transpose();
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> DDJrXVA_DXDX(const Eigen::Matrix<T, 3, 1> &X, const Eigen::Matrix<T, 3, 1> &V, const Eigen::Matrix<T, 3, 1> &A) const
    {
        using SO3T = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        T Xn = X.norm();

        if(Xn < 1e-4)
            return Fuu(X, V, A)/6.0;

        T Xnp2 = Xn*Xn;
        T Xnp3 = Xnp2*Xn;
        T Xnp4 = Xnp3*Xn;
        T Xnp5 = Xnp4*Xn;

        T sXn = sin(Xn);
        // T sXnp2 = sXn*sXn;
        
        T cXn = cos(Xn);
        // T cXnp2 = cXn*cXn;
        
        // T gXn = (1.0 - cXn)/Xnp2;
        T DgXn_DXn = sXn/Xnp2 - 2.0*(1.0 - cXn)/Xnp3;
        T DDgXn_DXnDXn = cXn/Xnp2 - 4.0*sXn/Xnp3 + 6.0*(1.0 - cXn)/Xnp4;

        T hXn = (Xn - sXn)/Xnp3;
        T DhXn_DXn = (1.0 - cXn)/Xnp3 - 3.0*(Xn - sXn)/Xnp4;
        T DDhXn_DXnDXn = 6.0/Xnp4 + sXn/Xnp3 + 6.0*cXn/Xnp4 - 12.0*sXn/Xnp5;

        Vec3T Xb = X/Xn;
        Mat3T DXb_DX = 1.0/Xn*(Mat3T::Identity(3, 3) - Xb*Xb.transpose());

        Vec3T XsksqV = SO3T::hat(X)*SO3T::hat(X)*V;
        Mat3T DXsksqV_DX = Fu(X, V);
        Mat3T DDXsksqVA_DXDX = Fuu(X, V, A);

        Mat3T Vsk = SO3T::hat(V);
        T AtpXb = A.transpose()*Xb;
        Eigen::Matrix<T, 1, 3> AtpDXb = A.transpose()*DXb_DX;

        return  Vsk*A*DgXn_DXn*Xb.transpose()

              + Vsk*AtpXb*DgXn_DXn
              + Vsk*X*AtpDXb*DgXn_DXn
              + Vsk*X*AtpXb*Xb.transpose()*DDgXn_DXnDXn

              + DDXsksqVA_DXDX*hXn
              + DXsksqV_DX*A*Xb.transpose()*DhXn_DXn

              + DXsksqV_DX*AtpXb*DhXn_DXn
              + XsksqV*AtpDXb*DhXn_DXn
              + XsksqV*AtpXb*Xb.transpose()*DDhXn_DXnDXn;
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> DDJrXVA_DXDV(const Eigen::Matrix<T, 3, 1> &X, const Eigen::Matrix<T, 3, 1> &V, const Eigen::Matrix<T, 3, 1> &A) const
    {
        using SO3T = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        T Xn = X.norm();

        if(Xn < 1e-4)
            return -0.5*SO3T::hat(A);

        T Xnp2 = Xn*Xn;
        T Xnp3 = Xnp2*Xn;
        T Xnp4 = Xnp3*Xn;
        // T Xnp5 = Xnp4*Xn;

        T sXn = sin(Xn);
        // T sXnp2 = sXn*sXn;
        
        T cXn = cos(Xn);
        // T cXnp2 = cXn*cXn;
        
        T gXn = (1.0 - cXn)/Xnp2;
        T DgXn_DXn = sXn/Xnp2 - 2.0*(1.0 - cXn)/Xnp3;
        // T DDgXn_DXnDXn = cXn/Xnp2 - 4.0*sXn/Xnp3 + 6.0*(1.0 - cXn)/Xnp4;

        T hXn = (Xn - sXn)/Xnp3;
        T DhXn_DXn = (1.0 - cXn)/Xnp3 - 3.0*(Xn - sXn)/Xnp4;
        // T DDhXn_DXnDXn = 6.0/Xnp4 + sXn/Xnp3 + 6.0*cXn/Xnp4 - 12*sXn/Xnp5;

        Vec3T Xb = X/Xn;
        // Mat3T DXb_DX = 1.0/Xn*(Mat3T::Identity(3, 3) - Xb*Xb.transpose());

        Mat3T DXsksqV_DV = Fv(X, V);
        Mat3T DDXsksqVA_DXDV = Fuv(X, V, A);

        T AtpXb = A.transpose()*Xb;

        return -SO3T::hat(A)*gXn - SO3T::hat(X)*DgXn_DXn*AtpXb + DDXsksqVA_DXDV*hXn + DXsksqV_DV*DhXn_DXn*AtpXb;
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> DJrInvXV_DX(const Eigen::Matrix<T, 3, 1> &X, const Eigen::Matrix<T, 3, 1> &V) const
    {
        using SO3T  = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        T Xn = X.norm();
        if(Xn < 1e-4)
            return -0.5*SO3T::hat(V);

        T Xnp2 = Xn*Xn;
        T Xnp3 = Xnp2*Xn;
        
        T sXn = sin(Xn);
        T sXnp2 = sXn*sXn;
        
        T cXn = cos(Xn);
        // T cXnp2 = cXn*cXn;
        
        T gXn = (1.0/Xnp2 - (1.0 + cXn)/(2.0*Xn*sXn));
        T DgXn_DXn = -2.0/Xnp3 + (Xn*sXnp2 + (sXn + Xn*cXn)*(1.0 + cXn))/(2.0*Xnp2*sXnp2);

        Vec3T Xb = X/Xn;

        Vec3T XsksqV = SO3T::hat(X)*SO3T::hat(X)*V;
        Mat3T DXsksqV_DX = Fu(X, V);

        return -0.5*SO3T::hat(V) + DXsksqV_DX*gXn + XsksqV*DgXn_DXn*Xb.transpose();
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> DDJrInvXVA_DXDX(const Eigen::Matrix<T, 3, 1> &X, const Eigen::Matrix<T, 3, 1> &V, const Eigen::Matrix<T, 3, 1> &A) const
    {
        using SO3T = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        T Xn = X.norm();

        if(Xn < 1e-4)
            return Fuu(X, V, A)/12.0;

        T Xnp2 = Xn*Xn;
        T Xnp3 = Xnp2*Xn;
        T Xnp4 = Xnp3*Xn;
        // T Xnp5 = Xnp4*Xn;

        T sXn = sin(Xn);
        T sXnp2 = sXn*sXn;
        // T s2Xn = sin(2.0*Xn);
        
        T cXn = cos(Xn);
        // T cXnp2 = cXn*cXn;
        T c2Xn = cos(2.0*Xn);
        
        T gXn = 1.0/Xnp2 - (1.0 + cXn)/(2.0*Xn*sXn);
        T DgXn_DXn = -2.0/Xnp3 + (sXn + Xn)*(1.0 + cXn)/(2.0*Xnp2*sXnp2);
        // T DDgXn_DXnDXn = 6.0/Xnp4 + (1.0 - c2Xn + Xnp2*cXn + 2.0*Xn*sXn + Xnp2)/(Xnp3*2.0*sXn*(cXn - 1.0));
        T DDgXn_DXnDXn = 6.0/Xnp4 + sXn/(Xnp3*(cXn - 1.0)) + (Xn*cXn + 2.0*sXn + Xn)/(2.0*Xnp2*sXn*(cXn - 1.0));

        Vec3T Xb = X/Xn;
        Mat3T DXb_DX = 1.0/Xn*(Mat3T::Identity(3, 3) - Xb*Xb.transpose());

        Vec3T XsksqV = SO3T::hat(X)*SO3T::hat(X)*V;
        Mat3T DXsksqV_DX = Fu(X, V);
        Mat3T DDXsksqVA_DXDX = Fuu(X, V, A);

        // Mat3T Vsk = SO3T::hat(V);
        T AtpXb = A.transpose()*Xb;
        Eigen::Matrix<T, 1, 3> AtpDXb = A.transpose()*DXb_DX;

        return   DDXsksqVA_DXDX*gXn
               + DXsksqV_DX*A*Xb.transpose()*DgXn_DXn

               + DXsksqV_DX*AtpXb*DgXn_DXn
               + XsksqV*AtpDXb*DgXn_DXn
               + XsksqV*AtpXb*Xb.transpose()*DDgXn_DXnDXn;
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> DDJrInvXVA_DXDV(const Eigen::Matrix<T, 3, 1> &X, const Eigen::Matrix<T, 3, 1> &V, const Eigen::Matrix<T, 3, 1> &A) const
    {
        using SO3T = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        T Xn = X.norm();

        if(Xn < 1e-4)
            return 0.5*SO3T::hat(A);

        T Xnp2 = Xn*Xn;
        T Xnp3 = Xnp2*Xn;
        // T Xnp4 = Xnp3*Xn;
        // T Xnp5 = Xnp4*Xn;

        T sXn = sin(Xn);
        T sXnp2 = sXn*sXn;
        // T s2Xn = sin(2*Xn);
        
        T cXn = cos(Xn);
        // T cXnp2 = cXn*cXn;
        // T c2Xn = cos(2*Xn);
        
        T gXn = (1.0/Xnp2 - (1.0 + cXn)/(2.0*Xn*sXn));
        T DgXn_DXn = -2.0/Xnp3 + (Xn*sXnp2 + (sXn + Xn*cXn)*(1.0 + cXn))/(2.0*Xnp2*sXnp2);
        // T DDgXn_DXnDXn = (Xn + 6.0*s2Xn - 12.0*sXn - Xn*c2Xn + Xnp3*cXn + 2.0*Xnp2*sXn + Xnp3)/(Xnp4*(s2Xn - 2.0*sXn));

        Vec3T Xb = X/Xn;
        // Mat3T DXb_DX = 1.0/Xn*(Mat3T::Identity(3, 3) - Xb*Xb.transpose());

        Mat3T DXsksqV_DV = Fv(X, V);
        // Mat3T DXsksqV_DX = Fu(X, V);
        Mat3T DDXsksqVA_DXDV = Fuv(X, V, A);

        T AtpXb = A.transpose()*Xb;

        return 0.5*SO3T::hat(A) + DDXsksqVA_DXDV*gXn + DXsksqV_DV*DgXn_DXn*AtpXb;
    }

    template <class T = double>
    void ComputeXtAndJacobians(const GPState<T> &Xa,
                               const GPState<T> &Xb,
                                     GPState<T> &Xt,
                               vector<vector<Eigen::Matrix<T, 3, 3>>> &DXt_DXa,
                               vector<vector<Eigen::Matrix<T, 3, 3>>> &DXt_DXb,
                               Eigen::Matrix<T, 9, 1> &gammaa_,
                               Eigen::Matrix<T, 9, 1> &gammab_,
                               Eigen::Matrix<T, 9, 1> &gammat_,
                               bool debug = false
                              ) const
    {
        using SO3T  = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Vec6T = Eigen::Matrix<T, 6, 1>;
        using Vec9T = Eigen::Matrix<T, 9, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        // Map the variables of the state
        double tau = Xt.t;
        SO3T   &Rt = Xt.R;
        Vec3T  &Ot = Xt.O;
        Vec3T  &St = Xt.S;
        Vec3T  &Pt = Xt.P;
        Vec3T  &Vt = Xt.V;
        Vec3T  &At = Xt.A;
        
        // Calculate the the mixer matrixes
        Matrix<T, Dynamic, Dynamic> LAM_ROSt = LAMDA(tau, SigGa).cast<T>();
        Matrix<T, Dynamic, Dynamic> PSI_ROSt = PSI(tau,   SigGa).cast<T>();
        Matrix<T, Dynamic, Dynamic> LAM_PVAt = LAMDA(tau, SigNu).cast<T>();
        Matrix<T, Dynamic, Dynamic> PSI_PVAt = PSI(tau,   SigNu).cast<T>();

        // Find the relative rotation
        SO3T Rab = Xa.R.inverse()*Xb.R;

        // Calculate the SO3 knots in relative form
        Vec3T Thead0 = Vec3T::Zero();
        Vec3T Thead1 = Xa.O;
        Vec3T Thead2 = Xa.S;

        Vec3T Theb = Rab.log();
        Mat3T JrInvTheb = JrInv(Theb);
        Mat3T HpTheb_ThebOb = DJrInvXV_DX(Theb, Xb.O);
        
        Vec3T Thebd0 = Theb;
        Vec3T Thebd1 = JrInvTheb*Xb.O;
        Vec3T Thebd2 = JrInvTheb*Xb.S + HpTheb_ThebOb*Thebd1;

        // Put them in vector form
        Vec9T gammaa; gammaa << Thead0, Thead1, Thead2;
        Vec9T gammab; gammab << Thebd0, Thebd1, Thebd2;

        // Calculate the knot euclid states and put them in vector form
        Vec9T pvaa; pvaa << Xa.P, Xa.V, Xa.A;
        Vec9T pvab; pvab << Xb.P, Xb.V, Xb.A;

        // Mix the knots to get the interpolated states
        Vec9T gammat = LAM_ROSt*gammaa + PSI_ROSt*gammab;
        Vec9T pvat   = LAM_PVAt*pvaa   + PSI_PVAt*pvab;

        // Retrive the interpolated SO3 in relative form
        Vec3T Thetd0 = gammat.block(0, 0, 3, 1);
        Vec3T Thetd1 = gammat.block(3, 0, 3, 1);
        Vec3T Thetd2 = gammat.block(6, 0, 3, 1);

        Mat3T JrThet  = Jr(Thetd0);
        SO3T  ExpThet = SO3T::exp(Thetd0);
        Mat3T HThet_ThetThetd1 = DJrXV_DX(Thetd0, Thetd1);

        // Assign the interpolated state
        Rt = Xa.R*ExpThet;
        Ot = JrThet*Thetd1;
        St = JrThet*Thetd2 + HThet_ThetThetd1*Thetd1;
        Pt = pvat.block(0, 0, 3, 1);
        Vt = pvat.block(3, 0, 3, 1);
        At = pvat.block(6, 0, 3, 1);

        // Calculate the Jacobian
        DXt_DXa = vector<vector<Mat3T>>(6, vector<Mat3T>(6, Mat3T::Zero()));
        DXt_DXb = vector<vector<Mat3T>>(6, vector<Mat3T>(6, Mat3T::Zero()));


        // Local index for the states in the state vector
        const int RIDX = 0;
        const int OIDX = 1;
        const int SIDX = 2;
        const int PIDX = 3;
        const int VIDX = 4;
        const int AIDX = 5;


        // Some reusable matrices
        SO3T ExpThetInv = ExpThet.inverse();
        Mat3T HpTheb_ThebSb = DJrInvXV_DX(Theb, Xb.S);
        Mat3T LpThebTheb_ThebObThebd1 = DDJrInvXVA_DXDX(Theb, Xb.O, Thebd1);
        Mat3T LpThebOb_ThebObThebd1 = DDJrInvXVA_DXDV(Theb, Xb.O, Thebd1);

        Mat3T HThet_ThetThetd2 = DJrXV_DX(Thetd0, Thetd2);
        Mat3T LThetThet_ThetThetd1Thetd1 = DDJrXVA_DXDX(Thetd0, Thetd1, Thetd1);
        Mat3T LThetThetd1_ThetThetd1Thetd1 = DDJrXVA_DXDV(Thetd0, Thetd1, Thetd1);
        

        // Jacobians from L1 to L0
        Mat3T JThead1Oa = Mat3T::Identity(); Mat3T JThead2Sa = Mat3T::Identity();

        Mat3T  JThebd0Ra = -JrInvTheb*Rab.inverse().matrix();
        Mat3T &JThebd0Rb =  JrInvTheb;

        Mat3T  JThebd1Ra = HpTheb_ThebOb*JThebd0Ra;
        Mat3T  JThebd1Rb = HpTheb_ThebOb*JThebd0Rb;
        Mat3T &JThebd1Ob = JrInvTheb;

        Mat3T  JThebd2Ra = HpTheb_ThebSb*JThebd0Ra + HpTheb_ThebOb*JThebd1Ra + LpThebTheb_ThebObThebd1*JThebd0Ra;
        Mat3T  JThebd2Rb = HpTheb_ThebSb*JThebd0Rb + HpTheb_ThebOb*JThebd1Rb + LpThebTheb_ThebObThebd1*JThebd0Rb;
        Mat3T  JThebd2Ob = LpThebOb_ThebObThebd1   + HpTheb_ThebOb*JThebd1Ob;
        Mat3T &JThebd2Sb = JrInvTheb;


        // Jacobians from L2 to L1
        Mat3T JThetd0Thead0 = LAM_ROSt.block(0, 0, 3, 3); Mat3T JThetd0Thead1 = LAM_ROSt.block(0, 3, 3, 3); Mat3T JThetd0Thead2 = LAM_ROSt.block(0, 6, 3, 3);
        Mat3T JThetd1Thead0 = LAM_ROSt.block(3, 0, 3, 3); Mat3T JThetd1Thead1 = LAM_ROSt.block(3, 3, 3, 3); Mat3T JThetd1Thead2 = LAM_ROSt.block(3, 6, 3, 3);
        Mat3T JThetd2Thead0 = LAM_ROSt.block(6, 0, 3, 3); Mat3T JThetd2Thead1 = LAM_ROSt.block(6, 3, 3, 3); Mat3T JThetd2Thead2 = LAM_ROSt.block(6, 6, 3, 3);

        Mat3T JThetd0Thebd0 = PSI_ROSt.block(0, 0, 3, 3); Mat3T JThetd0Thebd1 = PSI_ROSt.block(0, 3, 3, 3); Mat3T JThetd0Thebd2 = PSI_ROSt.block(0, 6, 3, 3);
        Mat3T JThetd1Thebd0 = PSI_ROSt.block(3, 0, 3, 3); Mat3T JThetd1Thebd1 = PSI_ROSt.block(3, 3, 3, 3); Mat3T JThetd1Thebd2 = PSI_ROSt.block(3, 6, 3, 3);
        Mat3T JThetd2Thebd0 = PSI_ROSt.block(6, 0, 3, 3); Mat3T JThetd2Thebd1 = PSI_ROSt.block(6, 3, 3, 3); Mat3T JThetd2Thebd2 = PSI_ROSt.block(6, 6, 3, 3);


        // Jacobians from L2 to L0
        Mat3T JThetd0Ra = JThetd0Thebd0*JThebd0Ra + JThetd0Thebd1*JThebd1Ra + JThetd0Thebd2*JThebd2Ra;
        Mat3T JThetd0Rb = JThetd0Thebd0*JThebd0Rb + JThetd0Thebd1*JThebd1Rb + JThetd0Thebd2*JThebd2Rb;
        Mat3T JThetd0Oa = JThetd0Thead1*JThead1Oa;
        Mat3T JThetd0Ob = JThetd0Thebd1*JThebd1Ob + JThetd0Thebd2*JThebd2Ob;
        Mat3T JThetd0Sa = JThetd0Thead2*JThead2Sa;
        Mat3T JThetd0Sb = JThetd0Thebd2*JThebd2Sb;

        Mat3T JThetd1Ra = JThetd1Thebd0*JThebd0Ra + JThetd1Thebd1*JThebd1Ra + JThetd1Thebd2*JThebd2Ra;
        Mat3T JThetd1Rb = JThetd1Thebd0*JThebd0Rb + JThetd1Thebd1*JThebd1Rb + JThetd1Thebd2*JThebd2Rb;
        Mat3T JThetd1Oa = JThetd1Thead1*JThead1Oa;
        Mat3T JThetd1Ob = JThetd1Thebd1*JThebd1Ob + JThetd1Thebd2*JThebd2Ob;
        Mat3T JThetd1Sa = JThetd1Thead2*JThead2Sa;
        Mat3T JThetd1Sb = JThetd1Thebd2*JThebd2Sb;

        Mat3T JThetd2Ra = JThetd2Thebd0*JThebd0Ra + JThetd2Thebd1*JThebd1Ra + JThetd2Thebd2*JThebd2Ra;
        Mat3T JThetd2Rb = JThetd2Thebd0*JThebd0Rb + JThetd2Thebd1*JThebd1Rb + JThetd2Thebd2*JThebd2Rb;
        Mat3T JThetd2Oa = JThetd2Thead1*JThead1Oa;
        Mat3T JThetd2Ob = JThetd2Thebd1*JThebd1Ob + JThetd2Thebd2*JThebd2Ob;
        Mat3T JThetd2Sa = JThetd2Thead2*JThead2Sa;
        Mat3T JThetd2Sb = JThetd2Thebd2*JThebd2Sb;


        // Jacobians from L3 to L2
        Mat3T &JRtThetd0 = JrThet;

        Mat3T &JOtThetd0 = HThet_ThetThetd1;
        Mat3T &JOtThetd1 = JrThet;

        Mat3T  JStThetd0 = HThet_ThetThetd2 + LThetThet_ThetThetd1Thetd1;
        Mat3T  JStThetd1 = LThetThetd1_ThetThetd1Thetd1 + HThet_ThetThetd1;
        Mat3T &JStThetd2 = JrThet;


        // DRt_DRa
        DXt_DXa[RIDX][RIDX] = ExpThetInv.matrix() + JRtThetd0*JThetd0Ra;
        // DRt_DOa
        DXt_DXa[RIDX][OIDX] = JRtThetd0*JThetd0Oa;
        // DRt_DSa
        DXt_DXa[RIDX][SIDX] = JRtThetd0*JThetd0Sa;
        // DRt_DPa DRt_DVa DRt_DAa are all zeros
        
        // DOt_Ra
        DXt_DXa[OIDX][RIDX] = JOtThetd0*JThetd0Ra + JOtThetd1*JThetd1Ra;
        // DOt_Oa
        DXt_DXa[OIDX][OIDX] = JOtThetd0*JThetd0Oa + JOtThetd1*JThetd1Oa;
        // DOt_Sa
        DXt_DXa[OIDX][SIDX] = JOtThetd0*JThetd0Sa + JOtThetd1*JThetd1Sa;
        // DOt_DPa DOt_DVa DOt_DAa are all zeros

        // DSt_Ra
        DXt_DXa[SIDX][RIDX] = JStThetd0*JThetd0Ra + JStThetd1*JThetd1Ra + JStThetd2*JThetd2Ra;
        // DSt_Oa
        DXt_DXa[SIDX][OIDX] = JStThetd0*JThetd0Oa + JStThetd1*JThetd1Oa + JStThetd2*JThetd2Oa;
        // DSt_Sa
        DXt_DXa[SIDX][SIDX] = JStThetd0*JThetd0Sa + JStThetd1*JThetd1Sa + JStThetd2*JThetd2Sa;
        // DSt_DPa DSt_DVa DSt_DAa are all zeros


        // DRt_DRb
        DXt_DXb[RIDX][RIDX] = JRtThetd0*JThetd0Rb;
        // DRt_DOb
        DXt_DXb[RIDX][OIDX] = JRtThetd0*JThetd0Ob;
        // DRt_DSb
        DXt_DXb[RIDX][SIDX] = JRtThetd0*JThetd0Sb;
        // DRt_DPb DRt_DVb DRt_DAb are all zeros
        
        // DOt_Rb
        DXt_DXb[OIDX][RIDX] = JOtThetd0*JThetd0Rb + JOtThetd1*JThetd1Rb;
        // DOt_Ob
        DXt_DXb[OIDX][OIDX] = JOtThetd0*JThetd0Ob + JOtThetd1*JThetd1Ob;
        // DOt_Sb
        DXt_DXb[OIDX][SIDX] = JOtThetd0*JThetd0Sb + JOtThetd1*JThetd1Sb;
        // DOt_DPb DOt_DVb DOt_DAb are all zeros

        // DSt_Rb
        DXt_DXb[SIDX][RIDX] = JStThetd0*JThetd0Rb + JStThetd1*JThetd1Rb + JStThetd2*JThetd2Rb;
        // DSt_Ob
        DXt_DXb[SIDX][OIDX] = JStThetd0*JThetd0Ob + JStThetd1*JThetd1Ob + JStThetd2*JThetd2Ob;
        // DSt_Sb
        DXt_DXb[SIDX][SIDX] = JStThetd0*JThetd0Sb + JStThetd1*JThetd1Sb + JStThetd2*JThetd2Sb;
        // DSt_DPb DSt_DVb DSt_DAb are all zeros




        // Extract the blocks of R3 states
        Mat3T LAM_PVA11 = LAM_PVAt.block(0, 0, 3, 3); Mat3T LAM_PVA12 = LAM_PVAt.block(0, 3, 3, 3); Mat3T LAM_PVA13 = LAM_PVAt.block(0, 6, 3, 3);
        Mat3T LAM_PVA21 = LAM_PVAt.block(3, 0, 3, 3); Mat3T LAM_PVA22 = LAM_PVAt.block(3, 3, 3, 3); Mat3T LAM_PVA23 = LAM_PVAt.block(3, 6, 3, 3);
        Mat3T LAM_PVA31 = LAM_PVAt.block(6, 0, 3, 3); Mat3T LAM_PVA32 = LAM_PVAt.block(6, 3, 3, 3); Mat3T LAM_PVA33 = LAM_PVAt.block(6, 6, 3, 3);

        Mat3T PSI_PVA11 = PSI_PVAt.block(0, 0, 3, 3); Mat3T PSI_PVA12 = PSI_PVAt.block(0, 3, 3, 3); Mat3T PSI_PVA13 = PSI_PVAt.block(0, 6, 3, 3);
        Mat3T PSI_PVA21 = PSI_PVAt.block(3, 0, 3, 3); Mat3T PSI_PVA22 = PSI_PVAt.block(3, 3, 3, 3); Mat3T PSI_PVA23 = PSI_PVAt.block(3, 6, 3, 3);
        Mat3T PSI_PVA31 = PSI_PVAt.block(6, 0, 3, 3); Mat3T PSI_PVA32 = PSI_PVAt.block(6, 3, 3, 3); Mat3T PSI_PVA33 = PSI_PVAt.block(6, 6, 3, 3);

        // DPt_DPa
        DXt_DXa[PIDX][PIDX] = LAM_PVA11;
        // DPt_DVa
        DXt_DXa[PIDX][VIDX] = LAM_PVA12;
        // DPt_DAa
        DXt_DXa[PIDX][AIDX] = LAM_PVA13;
        
        // DVt_DPa
        DXt_DXa[VIDX][PIDX] = LAM_PVA21;
        // DVt_DVa
        DXt_DXa[VIDX][VIDX] = LAM_PVA22;
        // DVt_DAa
        DXt_DXa[VIDX][AIDX] = LAM_PVA23;

        // DAt_DPa
        DXt_DXa[AIDX][PIDX] = LAM_PVA31;
        // DAt_DVa
        DXt_DXa[AIDX][VIDX] = LAM_PVA32;
        // DAt_DAa
        DXt_DXa[AIDX][AIDX] = LAM_PVA33;

        // DPt_DPb
        DXt_DXb[PIDX][PIDX] = PSI_PVA11;
        // DRt_DPb
        DXt_DXb[PIDX][VIDX] = PSI_PVA12;
        // DRt_DAb
        DXt_DXb[PIDX][AIDX] = PSI_PVA13;

        // DVt_DPb
        DXt_DXb[VIDX][PIDX] = PSI_PVA21;
        // DVt_DVb
        DXt_DXb[VIDX][VIDX] = PSI_PVA22;
        // DVt_DAb
        DXt_DXb[VIDX][AIDX] = PSI_PVA23;
        
        // DAt_DPb
        DXt_DXb[AIDX][PIDX] = PSI_PVA31;
        // DAt_DVb
        DXt_DXb[AIDX][VIDX] = PSI_PVA32;
        // DAt_DAb
        DXt_DXb[AIDX][AIDX] = PSI_PVA33;

        gammaa_ = gammaa;
        gammab_ = gammab;
        gammat_ = gammat;
    }

    GPMixer &operator=(const GPMixer &other)
    {
        this->dt = other.dt;
        this->SigGa = other.SigGa;
        this->SigNu = other.SigNu;
    }
};

// Define the shared pointer
typedef std::shared_ptr<GPMixer> GPMixerPtr;

/* #endregion Utility for propagation and interpolation matrices, elementary jacobians dXt/dXk, J_r, H_r, Hprime_r.. */


/* #region Managing control points: cration, extension, queries, ... ------------------------------------------------*/

class GaussianProcess
{
    using CovM = Eigen::Matrix<double, STATE_DIM, STATE_DIM>;

private:
    
    // The invalid covariance
    const CovM CovMZero = CovM::Zero();

    // Start time
    double t0 = 0;

    // Knot length
    double dt = 0.0;

    // Mixer
    GPMixerPtr gpm;

    // Set to true to maintain a covariance of each state
    bool keepCov = false;

    template <typename T>
    using aligned_deque = std::deque<T, Eigen::aligned_allocator<T>>;

    // Covariance
    aligned_deque<CovM> C;

    // State vector
    aligned_deque<SO3d> R;
    aligned_deque<Vec3> O;
    aligned_deque<Vec3> S;
    aligned_deque<Vec3> P;
    aligned_deque<Vec3> V;
    aligned_deque<Vec3> A;

public:

    // Destructor
    ~GaussianProcess(){};

    // Constructor
    GaussianProcess(double dt_, Mat3 SigGa_, Mat3 SigNu_, bool keepCov_=false)
        : dt(dt_), gpm(GPMixerPtr(new GPMixer(dt_, SigGa_, SigNu_))), keepCov(keepCov_) {};

    Mat3 getSigGa() const { return gpm->getSigGa(); }
    Mat3 getSigNu() const { return gpm->getSigNu(); }
    bool getKeepCov() const {return keepCov;}

    GPMixerPtr getGPMixerPtr()
    {
        return gpm;
    }

    double getMinTime() const
    {
        return t0;
    }

    double getMaxTime() const
    {
        return t0 + max(0, int(R.size()) - 1)*dt;
    }

    int getNumKnots() const
    {
        return int(R.size());
    }

    double getKnotTime(int kidx) const
    {
        return t0 + kidx*dt;
    }

    double getDt() const
    {
        return dt;
    }

    bool TimeInInterval(double t, double eps=0.0) const
    {
        return (t >= getMinTime() + eps && t < getMaxTime() - eps);
    }

    pair<int, double> computeTimeIndex(double t) const
    {
        int u = int((t - t0)/dt);
        double s = double(t - t0)/dt - u;
        return make_pair(u, s);
    }

    GPState<double> getStateAt(double t) const
    {
        // Find the index of the interval to find interpolation
        auto   us = computeTimeIndex(t);
        int    u  = us.first;
        double s  = us.second;

        int ua = u;  
        int ub = u+1;

        if (ub >= R.size() && fabs(1.0 - s) < 1e-5)
        {
            // printf(KYEL "Boundary issue: ub: %d, Rsz: %d, s: %f, 1-s: %f\n" RESET, ub, R.size(), s, fabs(1.0 - s));
            return GPState(t0 + ua*dt, R[ua], O[ua], S[ua], P[ua], V[ua], A[ua]);
        }

        // Extract the states of the two adjacent knots
        GPState Xa = GPState(t0 + ua*dt, R[ua], O[ua], S[ua], P[ua], V[ua], A[ua]);
        if (fabs(s) < 1e-5)
        {
            // printf(KYEL "Boundary issue: ub: %d, Rsz: %d, s: %f, 1-s: %f\n" RESET, ub, R.size(), s, fabs(1.0 - s));
            return Xa;
        }

        GPState Xb = GPState(t0 + ub*dt, R[ub], O[ub], S[ua], P[ub], V[ub], A[ub]);
        if (fabs(1.0 - s) < 1e-5)
        {
            // printf(KYEL "Boundary issue: ub: %d, Rsz: %d, s: %f, 1-s: %f\n" RESET, ub, R.size(), s, fabs(1.0 - s));
            return Xb;
        }

        SO3d Rab = Xa.R.inverse()*Xb.R;

        // Relative angle between two knots
        Vec3 Thea     = Vec3::Zero();
        Vec3 Thedota  = Xa.O;
        Vec3 Theddota = Xa.S;

        Vec3 Theb     = Rab.log();
        Vec3 Thedotb  = gpm->JrInv(Theb)*Xb.O;
        Vec3 Theddotb = gpm->JrInv(Theb)*Xb.S + gpm->DJrInvXV_DX(Theb, Xb.O)*Thedotb;

        Eigen::Matrix<double, 9, 1> gammaa; gammaa << Thea, Thedota, Theddota;
        Eigen::Matrix<double, 9, 1> gammab; gammab << Theb, Thedotb, Theddotb;

        Eigen::Matrix<double, 9, 1> pvaa; pvaa << Xa.P, Xa.V, Xa.A;
        Eigen::Matrix<double, 9, 1> pvab; pvab << Xb.P, Xb.V, Xb.A;

        Eigen::Matrix<double, 9, 1> gammat; // Containing on-manifold states (rotation and angular velocity)
        Eigen::Matrix<double, 9, 1> pvat;   // Position, velocity, acceleration

        gammat = gpm->LAMDA_ROS(s*dt) * gammaa + gpm->PSI_ROS(s*dt) * gammab;
        pvat   = gpm->LAMDA_PVA(s*dt) * pvaa   + gpm->PSI_PVA(s*dt) * pvab;

        // Retrive the interpolated SO3 in relative form
        Vec3 Thet     = gammat.block(0, 0, 3, 1);
        Vec3 Thedott  = gammat.block(3, 0, 3, 1);
        Vec3 Theddott = gammat.block(6, 0, 3, 1);

        // Assign the interpolated state
        SO3d Rt = Xa.R*SO3d::exp(Thet);
        Vec3 Ot = gpm->Jr(Thet)*Thedott;
        Vec3 St = gpm->Jr(Thet)*Theddott + gpm->DJrXV_DX(Thet, Thedott)*Thedott;
        Vec3 Pt = pvat.block<3, 1>(0, 0);
        Vec3 Vt = pvat.block<3, 1>(3, 0);
        Vec3 At = pvat.block<3, 1>(6, 0);

        return GPState<double>(t, Rt, Ot, St, Pt, Vt, At);
    }

    GPState<double> getKnot(int kidx) const
    {
        return GPState(getKnotTime(kidx), R[kidx], O[kidx], S[kidx], P[kidx], V[kidx], A[kidx]);
    }

    SE3d getKnotPose(int kidx) const
    {
        return SE3d(R[kidx], P[kidx]);
    }

    SE3d pose(double t) const
    {
        GPState X = getStateAt(t);
        return SE3d(X.R, X.P);
    }

    GPState<double> predictState(int steps)
    {
        SO3d Rc = R.back();
        Vec3 Oc = O.back();
        Vec3 Sc = S.back();
        Vec3 Pc = P.back();
        Vec3 Vc = V.back();
        Vec3 Ac = A.back();
        
        for(int k = 0; k < steps; k++)
        {
            SO3d Rpred = Rc*SO3d::exp(dt*Oc + 0.5*dt*dt*Sc);
            Vec3 Opred = Oc + dt*Sc;
            Vec3 Spred = Sc;
            Vec3 Ppred = Pc + dt*Vc + 0.5*dt*dt*Ac;
            Vec3 Vpred = Vc + dt*Ac;
            Vec3 Apred = Ac;

            Rc = Rpred;
            Oc = Opred;
            Sc = Spred;
            Pc = Ppred;
            Vc = Vpred;
            Ac = Apred;
        }

        return GPState<double>(getMaxTime() + steps*dt, Rc, Oc, Sc, Pc, Vc, Ac);
    }

    inline SO3d &getKnotSO3(size_t kidx) { return R[kidx]; }
    inline Vec3 &getKnotOmg(size_t kidx) { return O[kidx]; }
    inline Vec3 &getKnotAlp(size_t kidx) { return S[kidx]; }
    inline Vec3 &getKnotPos(size_t kidx) { return P[kidx]; }
    inline Vec3 &getKnotVel(size_t kidx) { return V[kidx]; }
    inline Vec3 &getKnotAcc(size_t kidx) { return A[kidx]; }
    inline CovM &getKnotCov(size_t kidx) { return C[kidx]; }

    void setStartTime(double t)
    {
        t0 = t;
        if (R.size() == 0)
        {
            R = {SO3d()};
            O = {Vec3(0, 0, 0)};
            S = {Vec3(0, 0, 0)};
            P = {Vec3(0, 0, 0)};
            V = {Vec3(0, 0, 0)};
            A = {Vec3(0, 0, 0)};
            
            if (keepCov)
                C = {CovMZero};
        }
    }
    
    void propagateCovariance()
    {
        CovM Cn = CovMZero;

        // If previous
        if (C.back().cwiseAbs().maxCoeff() != 0.0)
            Cn = gpm->PropagateFullCov(C.back());

        // Add the covariance to buffer
        C.push_back(Cn);
        assert(C.size() == R.size());
    }

    void extendKnotsTo(double t, const GPState<double> &Xn=GPState())
    {
        if(t0 == 0)
        {
            printf("MIN TIME HAS NOT BEEN INITIALIZED. "
                   "PLEASE CHECK, OR ELSE THE KNOT NUMBERS CAN BE VERY LARGE!");
            exit(-1);
        }
        
        // double tmax = getMaxTime();
        // if (tmax > t)
        //     return;

        // // Find the total number of knots at the new max time
        // int newknots = (t - t0 + dt - 1)/dt + 1;

        // Push the new state to the queue
        while(getMaxTime() < t)
        {
            R.push_back(Xn.R);
            O.push_back(Xn.O);
            S.push_back(Xn.S);
            P.push_back(Xn.P);
            V.push_back(Xn.V);
            A.push_back(Xn.A);

            if (keepCov)
                propagateCovariance();
        }
    }

    void extendOneKnot()
    {
        SO3d Rc = R.back();
        Vec3 Oc = O.back();
        Vec3 Sc = S.back();
        Vec3 Pc = P.back();
        Vec3 Vc = V.back();
        Vec3 Ac = A.back();

        SO3d Rn = Rc*SO3d::exp(dt*Oc + 0.5*dt*dt*Sc);
        Vec3 On = Oc + dt*Sc;
        Vec3 Sn = Sc;
        Vec3 Pn = Pc + dt*Vc + 0.5*dt*dt*Ac;
        Vec3 Vn = Vc + dt*Ac;
        Vec3 An = Ac;

        R.push_back(Rn);
        O.push_back(On);
        S.push_back(Sn);
        P.push_back(Pn);
        V.push_back(Vn);
        A.push_back(An);

        if (keepCov)
            propagateCovariance();
    }

    void extendOneKnot(const GPState<double> &Xn)
    {
        R.push_back(Xn.R);
        O.push_back(Xn.O);
        S.push_back(Xn.S);
        P.push_back(Xn.P);
        V.push_back(Xn.V);
        A.push_back(Xn.A);

        if (keepCov)
            propagateCovariance();
    }

    void setSigNu(const Matrix3d &m)
    {
        gpm->setSigNu(m);
    }

    void setSigGa(const Matrix3d &m)
    {
        gpm->setSigGa(m);
    }

    void setKnot(int kidx, const GPState<double> &Xn)
    {
        R[kidx] = Xn.R;
        O[kidx] = Xn.O;
        S[kidx] = Xn.S;
        P[kidx] = Xn.P;
        V[kidx] = Xn.V;
        A[kidx] = Xn.A;
    }

    void setKnotCovariance(int kidx, const CovM &Cov)
    {
        C[kidx] = Cov;
        assert(C.size() == R.size());
    }

    void updateKnot(int kidx, Matrix<double, STATE_DIM, 1> dX)
    {
        R[kidx] = R[kidx]*SO3d::exp(dX.block<3, 1>(0, 0));
        O[kidx] = O[kidx] + dX.block<3, 1>(3, 0);
        S[kidx] = S[kidx] + dX.block<3, 1>(6, 0);
        P[kidx] = P[kidx] + dX.block<3, 1>(9, 0);
        V[kidx] = V[kidx] + dX.block<3, 1>(12, 0);
        A[kidx] = A[kidx] + dX.block<3, 1>(15, 0);
    }

    void genRandomTrajectory(int n, double scale = 5.0)
    {
        R.clear(); O.clear(); S.clear(); P.clear(); V.clear(); A.clear();

        for(int kidx = 0; kidx < n; kidx++)
        {
            R.push_back(SO3d::exp(Vec3::Random()* M_PI));
            O.push_back(Vec3::Random() * scale);
            S.push_back(Vec3::Random() * scale);
            P.push_back(Vec3::Random() * scale);
            V.push_back(Vec3::Random() * scale);
            A.push_back(Vec3::Random() * scale);
        }
    }

    // Copy constructor
    GaussianProcess &operator=(GaussianProcess &other)
    {
        this->t0 = other.getMinTime();
        this->dt = other.getDt();
        
        *(this->gpm) = (*other.getGPMixerPtr());

        this->keepCov = other.keepCov;
        this->C = other.C;

        this->R = other.R;
        this->O = other.O;
        this->S = other.S;
        this->P = other.P;
        this->V = other.V;
        this->A = other.A;

        return *this;
    }

    bool saveTrajectory(string log_dir, int lidx, vector<double> ts)
    {
        string log_ = log_dir + "/gptraj_" + std::to_string(lidx) + ".csv";
        std::ofstream logfile;
        logfile.open(log_); // Open the file for writing
        logfile.precision(std::numeric_limits<double>::digits10 + 1);

        logfile << "Dt:" << dt << ";Order:" << 3 << ";Knots:" << getNumKnots() << ";MinTime:" << t0 << ";MaxTime:" << getMaxTime()
                << ";SigGa:" << getSigGa()(0, 0) << "," << getSigGa()(0, 1) << "," << getSigGa()(0, 2) << ","
                             << getSigGa()(1, 0) << "," << getSigGa()(1, 1) << "," << getSigGa()(1, 2) << ","
                             << getSigGa()(2, 0) << "," << getSigGa()(2, 1) << "," << getSigGa()(2, 2)
                << ";SigNu:" << getSigNu()(0, 0) << "," << getSigNu()(0, 1) << "," << getSigNu()(0, 2) << ","
                             << getSigNu()(1, 0) << "," << getSigNu()(1, 1) << "," << getSigNu()(1, 2) << ","
                             << getSigNu()(2, 0) << "," << getSigNu()(2, 1) << "," << getSigNu()(2, 2)
                << ";keepCov:" << getKeepCov()
                << endl;

        for(int kidx = 0; kidx < getNumKnots(); kidx++)
        {
            logfile << kidx << ", "
                    << getKnotTime(kidx) << ", "
                    << getKnotSO3(kidx).unit_quaternion().x() << ", "
                    << getKnotSO3(kidx).unit_quaternion().y() << ", "
                    << getKnotSO3(kidx).unit_quaternion().z() << ", "
                    << getKnotSO3(kidx).unit_quaternion().w() << ", "
                    << getKnotOmg(kidx).x() << ", "
                    << getKnotOmg(kidx).y() << ", "
                    << getKnotOmg(kidx).z() << ", "
                    << getKnotAlp(kidx).x() << ", "
                    << getKnotAlp(kidx).y() << ", "
                    << getKnotAlp(kidx).z() << ", "
                    << getKnotPos(kidx).x() << ", "
                    << getKnotPos(kidx).y() << ", "
                    << getKnotPos(kidx).z() << ", "
                    << getKnotVel(kidx).x() << ", "
                    << getKnotVel(kidx).y() << ", "
                    << getKnotVel(kidx).z() << ", "
                    << getKnotAcc(kidx).x() << ", "
                    << getKnotAcc(kidx).y() << ", "
                    << getKnotAcc(kidx).z() << endl;
        }

        logfile.close();
        return true;
    }

    bool loadTrajectory(string log_file)
    {
        std::ifstream file(log_file);

        auto splitstr = [](const string s_, const char d) -> vector<string>
        {
            std::istringstream s(s_);
            vector<string> o; string p;
            while(std::getline(s, p, d))
                o.push_back(p);
            return o;    
        };

        double dt_inLog;
        double t0_inLog;
        GPMixerPtr gpm_inLog;

        // Get the first line for specification
        if (file.is_open())
        {
            // Read the first line from the file
            std::string header;
            std::getline(file, header);

            printf("Get header: %s\n", header.c_str());
            vector<string> fields = splitstr(header, ';');
            map<string, int> fieldidx;
            for(auto &field : fields)
            {
                vector<string> fv = splitstr(field, ':');
                fieldidx[fv[0]] = fieldidx.size();
                printf("Field: %s. Value: %s\n", fv[0].c_str(), splitstr(fields[fieldidx[fv[0]]], ':').back().c_str());
            }

            auto strToMat3 = [&splitstr](const string &s, char d) -> Matrix3d
            {
                vector<string> Mstr = splitstr(s, d);
                for(int idx = 0; idx < Mstr.size(); idx++)
                    printf("Mstr[%d] = %s. S: %s\n", idx, Mstr[idx].c_str(), s.c_str());

                vector<double> Mdbl = {stod(Mstr[0]), stod(Mstr[1]), stod(Mstr[2]),
                                       stod(Mstr[3]), stod(Mstr[4]), stod(Mstr[5]), 
                                       stod(Mstr[6]), stod(Mstr[7]), stod(Mstr[8])};

                Eigen::Map<Matrix3d, Eigen::RowMajor> M(&Mdbl[0]);
                return M;
            };
            Matrix3d logSigNu = strToMat3(splitstr(fields[fieldidx["SigNu"]], ':').back(), ',');
            Matrix3d logSigGa = strToMat3(splitstr(fields[fieldidx["SigGa"]], ':').back(), ',');
            double logDt = stod(splitstr(fields[fieldidx["Dt"]], ':').back());
            double logMinTime = stod(splitstr(fields[fieldidx["MinTime"]], ':').back());
            bool logkeepCov = (stoi(splitstr(fields[fieldidx["keepCov"]], ':').back()) == 1);

            printf("Log configs:\n");
            printf("Dt: %f\n", logDt);
            printf("MinTime: %f\n", logMinTime);
            printf("SigNu: \n");
            cout << logSigNu << endl;
            printf("SigGa: \n");
            cout << logSigGa << endl;

            dt_inLog = logDt;
            t0_inLog = logMinTime;
            gpm_inLog = GPMixerPtr(new GPMixer(logDt, logSigGa, logSigNu));

            if (logkeepCov == keepCov)
                printf(KYEL "Covariance tracking is disabled\n" RESET);

            keepCov = false;
        }

        // Read txt to matrix
        auto read_csv =  [](const std::string &path, string dlm, int r_start = 0, int col_start = 0) -> MatrixXd
        {
            std::ifstream indata;
            indata.open(path);
            std::string line;
            std::vector<double> values;
            int row_idx = -1;
            int rows = 0;
            while (std::getline(indata, line))
            {
                row_idx++;
                if (row_idx < r_start)
                    continue;

                // printf("line: %s\n", line.c_str());

                std::stringstream lineStream(line);
                std::string cell;
                int col_idx = -1;
                while (std::getline(lineStream, cell, dlm[0]))
                {
                    if (cell == dlm || cell.size() == 0)
                        continue;

                    col_idx++;
                    if (col_idx < col_start)
                        continue;

                    values.push_back(std::stod(cell));

                    // printf("cell: %s\n", cell.c_str());
                }

                rows++;
            }

            return Eigen::Map<Matrix<double, -1, -1, Eigen::RowMajor>>(values.data(), rows, values.size() / rows);
        };

        // Load the control point values
        MatrixXd traj = read_csv(log_file, ",", 1, 0);
        printf("Found %d control points.\n", traj.rows());
        // for(int ridx = 0; ridx < traj.rows(); ridx++)
        // {
        //     cout << "Row: " << traj.row(ridx) << endl;
        //     if (ridx == 10)
        //         exit(-1);
        // }
        
        if(dt == 0 || dt_inLog == dt)
        {
            printf("dt has not been set. Use log's dt %f.\n", dt_inLog);
            
            // Clear the knots
            R.clear(); O.clear(); S.clear(); P.clear(); V.clear(); A.clear(); C.clear();

            // Set the knot values
            for(int ridx = 0; ridx < traj.rows(); ridx++)
            {
                VectorXd X = traj.row(ridx);
                R.push_back(SO3d(Quaternd(X(5), X(2), X(3), X(4))));
                O.push_back(Vec3(X(6),  X(7),  X(8)));
                S.push_back(Vec3(X(9),  X(10), X(11)));
                P.push_back(Vec3(X(12), X(13), X(14)));
                V.push_back(Vec3(X(15), X(16), X(17)));
                A.push_back(Vec3(X(18), X(19), X(20)));

                // C.push_back(CovMZero);
            }
        }
        else
        {
            printf(KYEL "Logged GPCT is has different knot length. Chosen: %f. Log: %f.\n" RESET, dt, dt_inLog);
            
            // Create a trajectory
            GaussianProcess trajLog(dt_inLog, gpm_inLog->getSigGa(), gpm_inLog->getSigNu());
            trajLog.setStartTime(t0_inLog);

            // Create the trajectory
            for(int ridx = 0; ridx < traj.rows(); ridx++)
            {
                VectorXd X = traj.row(ridx);
                trajLog.extendOneKnot(GPState<double>(ridx*dt_inLog+t0_inLog, SO3d(Quaternd(X(5), X(2), X(3), X(4))),
                                                                              Vec3(X(6),  X(7),  X(8)),
                                                                              Vec3(X(9),  X(10), X(11)),
                                                                              Vec3(X(12), X(13), X(14)),
                                                                              Vec3(X(15), X(16), X(17)),
                                                                              Vec3(X(18), X(19), X(20))));
            }

            // Sample the log trajectory to initialize current trajectory
            t0 = t0_inLog;
            R.clear(); O.clear(); S.clear(); P.clear(); V.clear(); A.clear(); C.clear();
            for(double ts = t0; ts < trajLog.getMaxTime() - trajLog.getDt(); ts += dt)
                extendOneKnot(trajLog.getStateAt(ts));
        }

        return true;
    }
};
// Define the shared pointer
typedef std::shared_ptr<GaussianProcess> GaussianProcessPtr;

/* #endregion Managing control points: cration, extension, queries, ... ---------------------------------------------*/


/* #region Local parameterization when using ceres ------------------------------------------------------------------*/

template <class Groupd>
class GPSO3LocalParameterization : public ceres::LocalParameterization
{
public:
    virtual ~GPSO3LocalParameterization() {}

    using Tangentd = typename Groupd::Tangent;

    /// @brief plus operation for Ceres
    ///
    ///  T * exp(x)
    ///
    virtual bool Plus(double const *T_raw, double const *delta_raw,
                      double *T_plus_delta_raw) const
    {
        Eigen::Map<Groupd const> const T(T_raw);
        Eigen::Map<Tangentd const> const delta(delta_raw);
        Eigen::Map<Groupd> T_plus_delta(T_plus_delta_raw);
        T_plus_delta = T * Groupd::exp(delta);
        return true;
    }

    virtual bool ComputeJacobian(double const *T_raw,
                                 double *jacobian_raw) const
    {
        Eigen::Map<Groupd const> T(T_raw);
        Eigen::Map<Eigen::Matrix<double, Groupd::num_parameters, Groupd::DoF, Eigen::RowMajor>>
        
        jacobian(jacobian_raw);
        jacobian.setZero();

        jacobian(0, 0) = 1;
        jacobian(1, 1) = 1;
        jacobian(2, 2) = 1;
        return true;
    }

    ///@brief Global size
    virtual int GlobalSize() const { return Groupd::num_parameters; }

    ///@brief Local size
    virtual int LocalSize() const { return Groupd::DoF; }
};
typedef GPSO3LocalParameterization<SO3d> GPSO3dLocalParameterization;

/* #endregion Local parameterization when using ceres ----------------------------------------------------------------*/
