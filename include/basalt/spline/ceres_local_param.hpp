/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt-headers.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

@file
@brief Generic local parametrization for Sophus Lie group types to be used with
ceres.
*/

/**
File adapted from Sophus

Copyright 2011-2017 Hauke Strasdat
          2012-2017 Steven Lovegrove

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights  to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
*/

#pragma once

#include <ceres/ceres.h>
#include <sophus/se3.hpp>


namespace basalt
{
    /// @brief Local parametrization for ceres that can be used with Sophus Lie
    /// group implementations.
    template <class Groupd>
    class LieLocalParameterization : public ceres::Manifold
    {
    public:
        virtual ~LieLocalParameterization() {}

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

        ///@brief Jacobian of plus operation for Ceres
        ///
        /// Dx T * exp(x)  with  x=0
        ///
        /// For SO3 group, function 'Dx_this_mul_exp_x_at_0()' is supposed to return a
        /// 4*3 matrix, which is
        ///       /        \
        ///      |  w -z  y |
        /// J =  |  z  w -x |
        ///      | -y  x  w |
        ///      | -x -y -z |
        ///       \        /
        bool PlusJacobian(const double *x, double *jacobian) const
        {
            Eigen::Map<const Groupd> T(x);
            Eigen::Map<Eigen::Matrix<double, Groupd::num_parameters, Groupd::DoF, Eigen::RowMajor>> J(jacobian);
            J = T.Dx_this_mul_exp_x_at_0();

            return true;
        }

        bool Minus(const double *y, const double *x, double *y_minus_x) const
        {
            Eigen::Map<const Groupd> _qx(x + 3);
            Eigen::Map<const Groupd> _qy(y + 3);
            Eigen::Map<Eigen::Matrix<double, Groupd::DoF, 1>> y_minus_x_(y_minus_x);
            y_minus_x_ = (_qx.inverse()*_qy).log();

            return true;
        }

        bool MinusJacobian(const double* x, double* jacobian) const
        {
            Eigen::Map<const Groupd> T(x);
            Eigen::Map<Eigen::Matrix<double, Groupd::DoF, Groupd::num_parameters, Eigen::RowMajor>>  J(jacobian);
            J = T.Dx_log_this_inv_by_x_at_this();

            return true;
        }

        ///@brief Global size
        int AmbientSize() const { return Groupd::num_parameters; }

        ///@brief Local size
        int TangentSize() const { return Groupd::DoF; }
    };


    template <class Groupd>
    class LieAnalyticLocalParameterization : public ceres::Manifold
    {
    public:
        virtual ~LieAnalyticLocalParameterization() {}

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

        virtual bool PlusJacobian(double const *x, double *jacobian) const
        {
            Eigen::Map<Groupd const> T(x);
            Eigen::Map<Eigen::Matrix<double, Groupd::num_parameters, Groupd::DoF,
                                     Eigen::RowMajor>> J(jacobian);
            J.setZero();
            J.template block<Groupd::DoF, Groupd::DoF>(0, 0).setIdentity();

            return true;
        }

        bool Minus(const double *y_, const double *x_, double *y_minus_x_) const
        {
            Eigen::Map<Groupd const> const x(x_);
            Eigen::Map<Groupd const> const y(y_);
            Eigen::Map<Tangentd> y_minus_x(y_minus_x_);

            y_minus_x = (x.inverse()*y).log();

            return true;
        }

        virtual bool MinusJacobian(const double* x, double* jacobian) const
        {
            Eigen::Map<Groupd const> const T(x);
            Eigen::Map<Eigen::Matrix<double, Groupd::DoF, Groupd::num_parameters, Eigen::RowMajor>> J(jacobian);
            J.setZero();
            J.template block<Groupd::DoF, Groupd::DoF>(0, 0).setIdentity();

            return true;
        }

        ///@brief Global size
        int AmbientSize() const { return Groupd::num_parameters; }

        ///@brief Local size
        int TangentSize() const { return Groupd::DoF; }
    };

} // namespace basalt