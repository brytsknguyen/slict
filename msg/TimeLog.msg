# This file is part of slict.
# 
# Copyright (C) 2020 Thien-Minh Nguyen <thienminh.nguyen at ntu dot edu dot sg>,
# School of EEE
# Nanyang Technological Univertsity, Singapore
# 
# For more information please see <https://britsknguyen.github.io>.
# or <https://github.com/brytsknguyen/slict>.
# If you use this code, please cite the respective publications as
# listed on the above websites.
# 
# slict is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# slict is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with slict.  If not, see <http://www.gnu.org/licenses/>.

#
# Created by Thien-Minh Nguyen on 15/12/20.
#

# Cloud Info
Header  header   # Header time stamp is the same with the latest feature batch stamp

int32   OptNum                  # Optimization counter

int32   ceres_iter              # For ceres solver, indicating the number of iterations

float64   t_insert
float64[] t_prop
float64[] t_desk
float64[] t_assoc
float64[] t_feasel
float64[] t_prep
float64[] t_hprior
float64[] t_himu
float64[] t_hlidar
float64[] t_compute
float64[] t_update
float64[] t_marg
float64[] t_solve
float64[] t_aftslv
float64   t_loop
