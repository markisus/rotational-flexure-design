#pragma once

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <limits>

double get_psi(double lambd, double mu, const Eigen::Matrix<double, 2, 2>& grad_u);
Eigen::Matrix<double, 1, 4> get_psi_J(double lambd, double mu, const Eigen::Matrix<double, 2, 2>& grad_u);
Eigen::Matrix<double, 4, 4> get_psi_H(double lambd, double mu, const Eigen::Matrix<double, 2, 2>& grad_u);
double get_grad_phi_det(const Eigen::Matrix<double, 8, 1>& params, const Eigen::Matrix<double, 2, 1>& isocoords);
Eigen::Matrix<double, 2, 2> get_grad_phi_inv(const Eigen::Matrix<double, 8, 1>& params, const Eigen::Matrix<double, 2, 1>& isocoords);
Eigen::Matrix<double, 2, 2> get_grad_phi(const Eigen::Matrix<double, 8, 1>& params, const Eigen::Matrix<double, 2, 1>& isocoords);
Eigen::Matrix<double, 2, 1> get_phi(const Eigen::Matrix<double, 8, 1>& params, const Eigen::Matrix<double, 2, 1>& isocoords);
Eigen::Matrix<double, 2, 2> get_gradU(const Eigen::Matrix<double, 8, 1>& uparams, const Eigen::Matrix<double, 8, 1>& eparams, const Eigen::Matrix<double, 2, 1>& isocoords);
Eigen::Matrix<double, 2, 1> local_to_global_deformed(const Eigen::Matrix<double, 8, 1>& uparams, const Eigen::Matrix<double, 8, 1>& eparams, const Eigen::Matrix<double, 2, 1>& isocoords);
Eigen::Matrix<double, 2, 1> local_to_global_undeformed(const Eigen::Matrix<double, 8, 1>& eparams, const Eigen::Matrix<double, 2, 1>& isocoords);
Eigen::Matrix<double, 4, 8> get_B_square(const Eigen::Matrix<double, 8, 1>& eparams, const Eigen::Matrix<double, 2, 1>& isocoords);

struct FemProblem {
		double lambda = 70;
    double mu = 30;

		bool is_polar = false;
		// in polar mode
		//  x = radial direction r
		//  y = angular direction theta
		//  
		//  update_problem will assume
		//  a connection between nodes
		//  y_idx = num_ynodes-1 and yidx = 0

		double length_x;
		double length_y;
		int num_xnodes;
		int num_ynodes;
		int num_dofs;

		std::vector<int> ux_dof_idxs;
		std::vector<int> uy_dof_idxs;

		std::vector<double> node_xs;
		std::vector<double> node_ys;

		void assert_dofs() {
				// count the number of -1 entires in problem.ux_dof_idxs
				int prescribed_count = 0;
				for (auto idx : this->ux_dof_idxs) {
						if (idx < 0) prescribed_count++;
				}
				for (auto idx : this->uy_dof_idxs) {
						if (idx < 0) prescribed_count++;
				}
				int num_dofs = this->ux_dof_idxs.size() + this->uy_dof_idxs.size() - prescribed_count;

				assert(this->num_dofs == num_dofs && "num_dofs is not set correctly");
				assert(this->num_dofs != 0 && "created a problem with 0 dofs");
		}

		void* user_data = nullptr;
};

struct PolarPrescribedDisplacement {
		double dr = 0;
		double dtheta = 0;
};

Eigen::Matrix<double, 2, 1> get_polar_prescribed_displacement(const FemProblem& polar_problem, int nx, int ny);
Eigen::Matrix<double, 2, 1> get_toy_prescribed_displacement(const FemProblem& toy_problem, int nx, int ny);
FemProblem create_polar_problem(double r0, double r1, int num_rnodes, int num_thetanodes);
FemProblem create_toy_problem(double length_x, double length_y, int num_xnodes, int num_ynodes);

struct FemIteration {
		double Pi = std::numeric_limits<float>::infinity();
		Eigen::MatrixXd d2Pi_du2; // deprecate?
		Eigen::VectorXd dPi_du;
		Eigen::VectorXd us;

		double last_change = 0;

		// coefficients for d2Pi_du2
		std::vector<Eigen::Triplet<double>> sparse_coefficients;

		void reset() {
				Pi = std::numeric_limits<float>::infinity(); 
				d2Pi_du2.setZero();
				dPi_du.setZero();
				us.setZero(); // not strictly necessary?
				sparse_coefficients.clear();
				last_change = 0;
		}
		
};

FemIteration create_fem_it(const FemProblem& problem);


Eigen::Vector2d get_deformed_coordinates(const FemProblem& problem, Eigen::Matrix<double, 2, 1>(*get_prescribed_displacement)(const FemProblem&, int nx, int ny), FemIteration& it, int nx, int ny);
Eigen::Vector2d get_undeformed_coordinates(const FemProblem& problem, int nx, int ny);

void update_problem(const FemProblem& problem, Eigen::Matrix<double, 2, 1>(*get_prescribed_displacement)(const FemProblem&, int nx, int ny), FemIteration& it);

