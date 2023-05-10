#include <cmath>
#include <iostream>
#include <array>
#include <utility>
#include <vector>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "fem.h"
#include "ticktock.h"

#define M_PI 3.14159265358979323846

double quad_area(const Eigen::Matrix<double, 8, 1>& corners) {
		double x1 = corners(0,0);
		double y1 = corners(1,0);
		double x2 = corners(2,0);
		double y2 = corners(3,0);
		double x3 = corners(4,0);
		double y3 = corners(5,0);
		double x4 = corners(6,0);
		double y4 = corners(7,0);
    return 0.5 * ((x1*y2 + x2*y3 + x3*y4 + x4*y1)-(x2*y1 + x3*y2 + x4*y3 + x1*y4));
}

void _fill_element_data(FemProblem& problem) {
		int num_xels = problem.num_xnodes-1;
		int num_yels = problem.is_polar ? problem.num_ynodes : problem.num_ynodes-1;
				
		problem.element_xys.reserve(8*num_xels*num_yels);
		problem.element_nxys.reserve(8*num_xels*num_yels);
		problem.element_dof_idxs.reserve(8*num_xels*num_yels);
		problem.gauss_pt_data.reserve(4*num_xels*num_yels);
		
		for (int nx = 0; nx < num_xels; ++nx) {
				for (int ny=0; ny < num_yels; ++ny) {
						// the corners of the element
						// the mod % below is no-op in non-polar mode
						// in polar mode, it just implements the wrap-around on the last element
						std::array<std::pair<int, int>, 4> nxys = {
								std::make_pair( nx, ny ),
								std::make_pair( nx+1, ny ),
								std::make_pair( nx+1, (ny+1)%problem.num_ynodes),
								std::make_pair( nx, (ny+1)%problem.num_ynodes)
						};

						for (const auto& nxy : nxys) {
								auto [nx, ny] = nxy;
								// ui for idx into the us (dof) buffer
								const int uix = problem.ux_dof_idxs[problem.num_xnodes*ny + nx];
								const int uiy = problem.uy_dof_idxs[problem.num_xnodes*ny + nx];
								problem.element_dof_idxs.push_back(uix);
								problem.element_dof_idxs.push_back(uiy);
								problem.element_nxys.push_back(nx);
								problem.element_nxys.push_back(ny);
						}


						Eigen::Matrix<double, 8, 1> eparams;
						int corner_idx = 0;
						for (const auto& nxy : nxys) {
								auto [nx, ny] = nxy;
								const double ex = problem.node_xs[problem.num_xnodes*ny + nx];
								const double ey = problem.node_ys[problem.num_xnodes*ny + nx];
								problem.element_xys.push_back(ex);
								problem.element_xys.push_back(ey);

								eparams(2*corner_idx) = ex;
								eparams(2*corner_idx+1) = ey;
								corner_idx += 1;
						}

						const double c = 1.0/std::sqrt(3);
						std::array<Eigen::Matrix<double, 2, 1>, 4> qs = {
								{{-c, -c}, {c, -c}, {c, c}, {-c, c}}
						};

						for (const auto& q : qs) {
								QuadData qd;
								qd.det = std::abs(get_grad_phi_det(eparams, q));
								qd.B = get_B_square(eparams, q);
								problem.gauss_pt_data.push_back(qd);
						}
				}
		}
}

FemProblem create_toy_problem(double length_x, double length_y, int num_xnodes, int num_ynodes) {
		FemProblem problem;
		problem.length_x = length_x;
		problem.length_y = length_y;
		problem.num_xnodes = num_xnodes;
		problem.num_ynodes = num_ynodes;

		problem.ux_dof_idxs.resize(num_xnodes*num_ynodes, -1);
		problem.uy_dof_idxs.resize(num_xnodes*num_ynodes, -1);

		problem.node_xs.resize(num_xnodes*num_ynodes);
		problem.node_ys.resize(num_xnodes*num_ynodes);


		// make the dof map
		int next_var_idx = 0;
		for (int x_idx = 0; x_idx < problem.num_xnodes; ++x_idx) {
				for(int y_idx = 0; y_idx < problem.num_ynodes; ++y_idx) {
						problem.node_xs[problem.num_xnodes*y_idx + x_idx] = length_x * x_idx/(num_xnodes-1);
						problem.node_ys[problem.num_xnodes*y_idx + x_idx] = length_y * y_idx/(num_ynodes-1);
						
						// left side and right side are prescribed
						if (x_idx == 0) continue;
						if (x_idx == problem.num_xnodes-1) continue;

						problem.ux_dof_idxs[problem.num_xnodes*y_idx + x_idx] = next_var_idx;
						++next_var_idx;
						problem.num_dofs += 1;

						problem.uy_dof_idxs[problem.num_xnodes*y_idx + x_idx] = next_var_idx;
						++next_var_idx;
						problem.num_dofs += 1;
				}
		}

		_fill_element_data(problem);

		problem.assert_dofs();

		return problem;
}

Eigen::Matrix<double, 2, 1> get_polar_prescribed_displacement(const FemProblem& problem, int nx, int ny) {
		// left side r=r0 is prescribed
		if (nx == 0) {
				// inner edge
				return Eigen::Vector2d::Zero();
		}
		else if (nx == problem.num_xnodes-1 && (ny == 0 || ny == 1)) {
				auto* polar_displacement = static_cast<PolarPrescribedDisplacement*>(problem.user_data);

				// get the x,y coordinates
				auto undeformed = get_undeformed_coordinates(problem, nx, ny);
				return polar_displacement->get_displacement(undeformed);
		}

		assert(false && "queried polar prescribed displacement of nonprescribed point");
}

FemProblem create_polar_problem(double r0, double r1, int num_rnodes, int num_thetanodes) {
		const double num_xnodes = num_rnodes;
		const double num_ynodes = num_thetanodes;

		FemProblem problem;
		problem.is_polar = true;
		problem.length_x = 1.0;
		problem.length_y = 2 * M_PI;
		problem.num_xnodes = num_xnodes;
		problem.num_ynodes = num_ynodes;

		problem.ux_dof_idxs.resize(num_xnodes*num_ynodes, -1);
		problem.uy_dof_idxs.resize(num_xnodes*num_ynodes, -1);

		problem.node_xs.resize(num_xnodes*num_ynodes);
		problem.node_ys.resize(num_xnodes*num_ynodes);


		// make the dof map
		int next_var_idx = 0;
		for (int x_idx = 0; x_idx < problem.num_xnodes; ++x_idx) {
				for(int y_idx = 0; y_idx < problem.num_ynodes; ++y_idx) {
						// translate from polar space to xy space
						const double r = r0 + ((r1 - r0) * x_idx)/(num_xnodes-1);

						// theta_max is not 2pi because we want the gap to be closed by
						// wrap-around
						const double theta_max = (2*M_PI * (num_ynodes-2))/(num_ynodes-1);
						const double theta = (theta_max * y_idx)/(num_ynodes-1);

						const double x = r*std::cos(theta);
						const double y = r*std::sin(theta);

						problem.node_xs[problem.num_xnodes*y_idx + x_idx] = x;
						problem.node_ys[problem.num_xnodes*y_idx + x_idx] = y;
						
						// left side r=r0 is prescribed
						if (x_idx == 0) continue;

						// right side, r=r1 is prescribed around theta=0
						if (x_idx == problem.num_xnodes-1 && (y_idx == 0 || y_idx == 1)) continue;

						problem.ux_dof_idxs[problem.num_xnodes*y_idx + x_idx] = next_var_idx;
						++next_var_idx;
						problem.num_dofs += 1;

						problem.uy_dof_idxs[problem.num_xnodes*y_idx + x_idx] = next_var_idx;
						++next_var_idx;
						problem.num_dofs += 1;
				}
		}

		_fill_element_data(problem);
		problem.assert_dofs();

		return problem;
}


Eigen::Matrix<double, 2, 1> get_toy_prescribed_displacement(const FemProblem& toy_problem, int nx, int ny) {
		assert(toy_problem.user_data != nullptr && "toy problem user_data was null while querying prescribed displacement");

		if (nx == 0) {
				return Eigen::Matrix<double, 2, 1>::Zero();
		}
		
		assert(nx == toy_problem.num_xnodes-1 && "asking for prescribed displacement of wrong node");

		auto* right_edge_displacement = static_cast<Eigen::Matrix<double, 2, 1>*>(toy_problem.user_data);
		return *right_edge_displacement;
}

FemIteration create_fem_it(const FemProblem& problem) {
		assert(problem.num_dofs != 0 && "tried to construct FemIteration with num_dofs == 0");
		
		FemIteration it;
		it.dPi_du = Eigen::VectorXd(problem.num_dofs);
		it.dPi_du.setZero();

		it.d2Pi_du2 = Eigen::MatrixXd(problem.num_dofs, problem.num_dofs);
		it.d2Pi_du2.setZero();

		it.us = Eigen::VectorXd(problem.num_dofs);
		it.us.setZero();

		it.Pi = 0;

		return it;
		
}

Eigen::Vector2d get_undeformed_coordinates(const FemProblem& problem, int nx, int ny) {
		return { problem.node_xs[problem.num_xnodes*ny + nx], problem.node_ys[problem.num_xnodes*ny + nx] };
}

Eigen::Vector2d get_deformed_coordinates(const FemProblem& problem, Eigen::Matrix<double, 2, 1>(*get_prescribed_displacement)(const FemProblem&, int nx, int ny), FemIteration& it, int nx, int ny) {
		assert(it.us.rows() != 0 && "fem iteration not initialized");

		Eigen::Vector2d result = {
				problem.node_xs[problem.num_xnodes*ny + nx],
				problem.node_ys[problem.num_xnodes*ny + nx]
		};

		const int uix = problem.ux_dof_idxs[problem.num_xnodes*ny + nx];
		const int uiy = problem.uy_dof_idxs[problem.num_xnodes*ny + nx];

		if (uix < 0 || uiy < 0) {
				return result + get_prescribed_displacement(problem, nx, ny);
		}

		result(0) += it.us(uix);
		result(1) += it.us(uiy);
		return result;
}

std::array<std::pair<int, int>, 4> get_nxys(int num_ynodes, int nx, int ny) {
		return {
				std::make_pair( nx, ny ),
				std::make_pair( nx+1, ny ),
				std::make_pair( nx+1, (ny+1)%num_ynodes),
				std::make_pair( nx, (ny+1)%num_ynodes)
		};
}

Eigen::Matrix<double, 8, 1> get_uparams(const FemProblem& problem, Eigen::Matrix<double, 2, 1>(*get_prescribed_displacement)(const FemProblem&, int nx, int ny),
																				const FemIteration& it,
																				const std::array<std::pair<int, int>, 4>& nxys, int* uis = nullptr) {
		// assemble the dofs from global to local
		Eigen::Matrix<double, 8, 1> uparams;
		{
				int corner_idx = 0;
				for (const auto& nxy : nxys) {
						auto [nx, ny] = nxy;

						// ui for idx into the us (dof) buffer
						const int uix = problem.ux_dof_idxs[problem.num_xnodes*ny + nx];
						const int uiy = problem.uy_dof_idxs[problem.num_xnodes*ny + nx];

						if (uis) {
								uis[2*corner_idx] = uix;
								uis[2*corner_idx+1] = uiy;
						}

						double ux = 0;
						double uy = 0;
								
						if (uix < 0 || uiy < 0) {
								auto prescribed_displacement = get_prescribed_displacement(problem, nx, ny);
								ux = prescribed_displacement(0,0);
								uy = prescribed_displacement(1,0);
						} else {
								ux = it.us[uix];
								uy = it.us[uiy];
						}
								
						uparams(2*corner_idx, 0) = ux;
						uparams(2*corner_idx+1, 0) = uy;
						corner_idx += 1;
				}
		}

		return uparams;
}

Eigen::Matrix<double, 8, 1> get_eparams(const FemProblem& problem, const std::array<std::pair<int, int>, 4>& nxys) {
		Eigen::Matrix<double, 8, 1> eparams;
		int corner_idx = 0;
		for (const auto& nxy : nxys) {
				auto [nx, ny] = nxy;
				const double ex = problem.node_xs[problem.num_xnodes*ny + nx];
				const double ey = problem.node_ys[problem.num_xnodes*ny + nx];
				eparams(2*corner_idx, 0) = ex;
				eparams(2*corner_idx+1, 0) = ey;
				corner_idx += 1;
		}

		return eparams;
}

void update_problem(const FemProblem& problem, Eigen::Matrix<double, 2, 1>(*get_prescribed_displacement)(const FemProblem&, int nx, int ny), FemIteration& it) {
		// TickTock timer;
		// timer.tick();
		
		if (it.us.rows() == 0) {
				// it was not initialized
				it = create_fem_it(problem);
		}

		assert(it.dPi_du.rows() != 0 && "dPi_du had 0 rows");
		assert(it.d2Pi_du2.rows() != 0 && "dPi_du had 0 rows");

		const double last_Pi = it.Pi;
		
		it.Pi = 0;
		it.d2Pi_du2.setZero();
		it.dPi_du.setZero();
		it.sparse_coefficients.clear();
		it.sparse_coefficients.reserve(problem.num_dofs*28);



		constexpr bool debug = false;

		// timer.tick();
		// minus 1 because we are actually iterating the elements

		const int ny_upper_bound = problem.is_polar ? problem.num_ynodes : problem.num_ynodes-1;

		if (debug) {
				std::cout << "Starting fem calculations" << "\n";
		}
		
		for (int nx = 0; nx < problem.num_xnodes-1; ++nx) {
				for (int ny=0; ny < ny_upper_bound; ++ny) {
						// the corners of the element
						// the mod % below is no-op in non-polar mode
						// in polar mode, it just implements the wrap-around on the last element
						std::array<std::pair<int, int>, 4> nxys = get_nxys(problem.num_ynodes, nx, ny);

						if (debug) {
								std::cout << "Coordinates" << "\n";
								for (auto nxy : nxys) {
										auto [nx, ny] = nxy;
										std::cout << "\tnx,ny " << nx <<", "<< ny << "\n";
								}
						}

						// assemble the dofs from global to local
						std::array<int, 8> uis; // local dof to global dfo
						Eigen::Matrix<double, 8, 1> uparams = get_uparams(problem, get_prescribed_displacement, it, nxys, uis.data());

						if (debug) {
								std::cout << "uparams " << uparams.transpose() << "\n";
						}

						Eigen::Matrix<double, 8, 1> eparams = get_eparams(problem, nxys);

						if (debug) {
								std::cout << "eparams " << eparams.transpose() << "\n";
								std::cout << std::flush;
						}


						bool use_gauss_quad = false;
						if (use_gauss_quad) {
								const double c = 1.0/std::sqrt(3);
								std::array<Eigen::Matrix<double, 2, 1>, 4> qs = {
										{{-c, -c}, {c, -c}, {c, c}, {-c, c}}
								};

								double pi = 0;
								auto dpi_du = Eigen::Matrix<double, 8, 1>::Zero().eval();
                auto d2pi_du2 = Eigen::Matrix<double, 8, 8>::Zero().eval();
								for (const auto& q : qs) {
                    double det = std::abs(get_grad_phi_det(eparams, q));
										auto B_square = get_B_square(eparams, q);

                    auto gradU = get_gradU(uparams, eparams, q); // was this wrong in python?
                    double psi = get_psi(problem.lambda, problem.mu, gradU);
                    pi += psi * det;

										auto psi_J = get_psi_J(problem.lambda, problem.mu, gradU);

                    dpi_du += (psi_J * B_square).transpose() * det;

										auto psi_H = get_psi_H(problem.lambda, problem.mu, gradU);
                    d2pi_du2 += B_square.transpose() * psi_H * B_square * det;
								}

								it.Pi += pi;

								assert(dpi_du.allFinite() && "nan value in dpi_du");
								assert(d2pi_du2.allFinite() && "nan value in dpi2_d2u");

								// scatter the updates back to the global
								for (int i = 0; i < 8; ++i) {
										int ui = uis[i];
										if (ui < 0) continue; // prescribed
										it.dPi_du(ui, 0) += dpi_du(i,0);
								}

								for (int i = 0; i < 8; ++i) {
										int ui = uis[i];
										if (ui < 0) continue;
										for (int j = 0; j < 8; ++j) {
												int uj = uis[j];
												if (uj < 0) continue;

												it.sparse_coefficients.push_back({ui, uj, d2pi_du2(i,j)});
												
												it.d2Pi_du2(ui, uj) += d2pi_du2(i,j);
										}
								}
						}
						else {
								Eigen::Matrix<double, 2, 1> q;
								q.setZero(); // center of the element
								
								double element_area = quad_area(eparams);
								auto B_square = get_B_square(eparams, q);
								auto gradU = get_gradU(uparams, eparams, q);

								assert(B_square.allFinite() && "nan value in B_square");
								assert(gradU.allFinite() && "nan value in gradU");


								double psi = get_psi(problem.lambda, problem.mu, gradU);
								double pi = psi * element_area;
								it.Pi += pi;

								auto psi_J = get_psi_J(problem.lambda, problem.mu, gradU);
								Eigen::Matrix<double, 8, 1> dpi_du = (psi_J * B_square).transpose() * element_area;

								auto psi_H = get_psi_H(problem.lambda, problem.mu, gradU);
								Eigen::Matrix<double, 8, 8> d2pi_du2 = B_square.transpose() * psi_H * B_square * element_area;


								for (int i = 0; i < 8; ++i) {
										int ui = uis[i];
										if (ui < 0) continue; // prescribed
										it.dPi_du(ui, 0) += dpi_du(i,0);
								}

								for (int i = 0; i < 8; ++i) {
										int ui = uis[i];
										if (ui < 0) continue;
										for (int j = 0; j < 8; ++j) {
												int uj = uis[j];
												if (uj < 0) continue;
												it.d2Pi_du2(ui, uj) += d2pi_du2(i,j);

												it.sparse_coefficients.push_back({ui, uj, d2pi_du2(i,j)});
										}
								}

								assert(dpi_du.allFinite() && "nan value in dpi_du");
								assert(d2pi_du2.allFinite() && "nan value in dpi2_d2u");

								if (debug) {
										std::cout << "In element " << nx << ", " << ny << "\n";

										for(int i = 0; i < 8; ++i) {
												std::cout << "uis[" << i << "] = " << uis[i] << "\n";
										}

										std::cout << "dpi_du was\n" << dpi_du.transpose() << "\n";
										std::cout << "dPi_du was\n" << it.dPi_du.transpose() << "\n";

										std::cout << "d2pi_du2 was \n"
															<< d2pi_du2 << "\n";
										std::cout << "d2Pi_du2 was \n"
															<< it.d2Pi_du2 << "\n";

										std::cout << "==============" << "\n";
								}
						}
				}
		}

		// std::cout << "sparse coefficients size" << sparse_coefficients.size() << "\n";
		// std::cout << "sparse coefficients factor" << sparse_coefficients.size()/problem.num_dofs << "\n";

		// timer.tock("gradient computation");

		// now update the dofs
		// if (debug) {
		// 		std::cout << "Pi was " << it.Pi << "\n";
		// 		std::cout << "d2Pi_du2 was "
		// 							<<	it.d2Pi_du2.rows()
		// 							<< " x "
		// 							<< it.d2Pi_du2.cols()
		// 							<< "\n" << it.d2Pi_du2 << "\n";
		// 		std::cout << "dPi_du was\n" << it.dPi_du.transpose() << "\n";
		// }

		// timer.tick();

		Eigen::SparseMatrix<double> d2Pi_du2_sparse(problem.num_dofs, problem.num_dofs);
		d2Pi_du2_sparse.setFromTriplets(it.sparse_coefficients.begin(), it.sparse_coefficients.end());

		Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> chol(d2Pi_du2_sparse);  // performs a Cholesky factorization of A
		Eigen::VectorXd dus = chol.solve(-it.dPi_du);

		// timer.tock("solve sparse");
		// timer.tick();
				
		// Eigen::LLT<Eigen::MatrixXd> lltOfA(it.d2Pi_du2);  // compute the Cholesky decomposition of A
    // Eigen::VectorXd dus = lltOfA.solve(-it.dPi_du);

		// timer.tock("solve");
		if (debug) {
				std::cout << "Pi was " << it.Pi << "\n";
				std::cout << "dus" << dus.transpose();
		}

		// std::cout << "dus-dus_alt" << (dus - dus_alt).array().abs().maxCoeff() << "\n";
		
		it.us += dus;
		it.last_change = it.Pi - last_Pi;

		// timer.tock("update_problem");
}

void update_problem_alt(const FemProblem& problem, Eigen::Matrix<double, 2, 1>(*get_prescribed_displacement)(const FemProblem&, int nx, int ny), FemIteration& it) {
		// this version actualy takes longer!!!
		TickTock timer;
		timer.tick();
		
		
		if (it.us.rows() == 0) {
				// it was not initialized
				it = create_fem_it(problem);
		}

		assert(it.dPi_du.rows() != 0 && "dPi_du had 0 rows");
		assert(it.d2Pi_du2.rows() != 0 && "dPi_du had 0 rows");

		const double last_Pi = it.Pi;
		
		it.Pi = 0;
		it.d2Pi_du2.setZero();
		it.dPi_du.setZero();
		it.sparse_coefficients.clear();
		it.sparse_coefficients.reserve(problem.num_dofs*28);

		// TickTock timer;

		constexpr bool debug = false;

		// timer.tick();

		for (int ei = 0; ei < problem.element_xys.size()/8; ++ei) {
				// assemble the dofs from global to local
				Eigen::Matrix<double, 8, 1> eparams;
				for (int i = 0; i < 8; ++i) {
						eparams(i) = problem.element_xys[8*ei + i];
				}

				
				std::array<int, 8> uis; // local dof to global dof
				Eigen::Matrix<double, 8, 1> uparams;
				for (int i = 0; i < 4; ++i) {
						const int uix = problem.element_dof_idxs[8*ei + 2*i];
						const int uiy = problem.element_dof_idxs[8*ei + 2*i + 1];
						uis[2*i] = uix;
						uis[2*i+1] = uiy;
						
						double ux = 0;
						double uy = 0;
						if (uix < 0 || uiy < 0) {
								const int nx = problem.element_nxys[8*ei + 2*i];
								const int ny = problem.element_nxys[8*ei + 2*i + 1];

								auto prescribed_displacement = get_prescribed_displacement(problem, nx, ny);
								ux = prescribed_displacement(0,0);
								uy = prescribed_displacement(1,0);
						} else {
								ux = it.us[uix];
								uy = it.us[uiy];
						}

						uparams(2*i) = ux;
						uparams(2*i+1) = uy;
				}
								
				// do gauss quadrature
				const double c = 1.0/std::sqrt(3);
				std::array<Eigen::Matrix<double, 2, 1>, 4> qs = {
						{{-c, -c}, {c, -c}, {c, c}, {-c, c}}
				};

				double pi = 0;
				auto dpi_du = Eigen::Matrix<double, 8, 1>::Zero().eval();
				auto d2pi_du2 = Eigen::Matrix<double, 8, 8>::Zero().eval();
				for (int i = 0; i < 4; ++i) {
						const auto q = qs[i];
						
						double det = std::abs(get_grad_phi_det(eparams, q));
						auto B = get_B_square(eparams, q);

						auto gradU = get_gradU(uparams, eparams, q);
						double psi = get_psi(problem.lambda, problem.mu, gradU);
						pi += psi * det;

						auto psi_J = get_psi_J(problem.lambda, problem.mu, gradU);

						dpi_du += (psi_J * B).transpose() * det;

						auto psi_H = get_psi_H(problem.lambda, problem.mu, gradU);
						d2pi_du2 += B.transpose() * psi_H * B * det;
				}

				it.Pi += pi;

				assert(dpi_du.allFinite() && "nan value in dpi_du");
				assert(d2pi_du2.allFinite() && "nan value in dpi2_d2u");

				// scatter the updates back to the global
				for (int i = 0; i < 8; ++i) {
						int ui = uis[i];
						if (ui < 0) continue; // prescribed
						it.dPi_du(ui, 0) += dpi_du(i,0);
				}

				for (int i = 0; i < 8; ++i) {
						int ui = uis[i];
						if (ui < 0) continue;
						for (int j = 0; j < 8; ++j) {
								int uj = uis[j];
								if (uj < 0) continue;

								it.sparse_coefficients.push_back({ui, uj, d2pi_du2(i,j)});
								it.d2Pi_du2(ui, uj) += d2pi_du2(i,j);
						}
				}
		}

		// std::cout << "sparse coefficients size" << sparse_coefficients.size() << "\n";
		// std::cout << "sparse coefficients factor" << sparse_coefficients.size()/problem.num_dofs << "\n";

		// timer.tock("gradient computation");

		// now update the dofs
		// if (debug) {
		// 		std::cout << "Pi was " << it.Pi << "\n";
		// 		std::cout << "d2Pi_du2 was "
		// 							<<	it.d2Pi_du2.rows()
		// 							<< " x "
		// 							<< it.d2Pi_du2.cols()
		// 							<< "\n" << it.d2Pi_du2 << "\n";
		// 		std::cout << "dPi_du was\n" << it.dPi_du.transpose() << "\n";
		// }

		// timer.tick();

		Eigen::SparseMatrix<double> d2Pi_du2_sparse(problem.num_dofs, problem.num_dofs);
		d2Pi_du2_sparse.setFromTriplets(it.sparse_coefficients.begin(), it.sparse_coefficients.end());

		Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> chol(d2Pi_du2_sparse);  // performs a Cholesky factorization of A
		Eigen::VectorXd dus = chol.solve(-it.dPi_du);

		// timer.tock("solve sparse");
		// timer.tick();
				
		// Eigen::LLT<Eigen::MatrixXd> lltOfA(it.d2Pi_du2);  // compute the Cholesky decomposition of A
    // Eigen::VectorXd dus = lltOfA.solve(-it.dPi_du);

		// timer.tock("solve");
		if (debug) {
				std::cout << "Pi was " << it.Pi << "\n";
				std::cout << "dus" << dus.transpose();
		}

		// std::cout << "dus-dus_alt" << (dus - dus_alt).array().abs().maxCoeff() << "\n";
		
		it.us += dus;
		it.last_change = it.Pi - last_Pi;

		timer.tock("update_problem_alt");
}

double get_psi(double lambd, double mu, const Eigen::Matrix<double, 2, 2>& grad_u) {
    double u11 = grad_u(0, 0);
    double u12 = grad_u(0, 1);
    double u21 = grad_u(1, 0);
    double u22 = grad_u(1, 1);

		double psi = (1.0/4.0)*lambd*u11*u11*u11*u11 + lambd*u11*u11*u11 + (1.0/2.0)*lambd*u11*u11*u12*u12
        + (1.0/2.0)*lambd*u11*u11*u21*u21 + (1.0/2.0)*lambd*u11*u11*u22*u22 + lambd*u11*u11*u22 + lambd*u11*u11
        + lambd*u11*u12*u12 + lambd*u11*u21*u21 + lambd*u11*u22*u22 + 2.0*lambd*u11*u22
        + (1.0/4.0)*lambd*u12*u12*u12*u12 + (1.0/2.0)*lambd*u12*u12*u21*u21 + (1.0/2.0)*lambd*u12*u12*u22*u22
        + lambd*u12*u12*u22 + (1.0/4.0)*lambd*u21*u21*u21*u21 + (1.0/2.0)*lambd*u21*u21*u22*u22 + lambd*u21*u21*u22
        + (1.0/4.0)*lambd*u22*u22*u22*u22 + lambd*u22*u22*u22 + lambd*u22*u22 + (1.0/2.0)*mu*u11*u11*u11*u11
        + 2.0*mu*u11*u11*u11 + mu*u11*u11*u12*u12 + mu*u11*u11*u21*u21 + 2.0*mu*u11*u11 + 2.0*mu*u11*u12*u12
        + 2.0*mu*u11*u12*u21*u22 + 2.0*mu*u11*u12*u21 + 2.0*mu*u11*u21*u21 + (1.0/2.0)*mu*u12*u12*u12*u12
        + mu*u12*u12*u22*u22 + 2.0*mu*u12*u12*u22 + mu*u12*u12 + 2.0*mu*u12*u21*u22 + 2.0*mu*u12*u21
        + (1.0/2.0)*mu*u21*u21*u21*u21 + mu*u21*u21*u22*u22 + 2.0*mu*u21*u21*u22 + mu*u21*u21
        + (1.0/2.0)*mu*u22*u22*u22*u22 + 2.0*mu*u22*u22*u22 + 2.0*mu*u22*u22;

    return psi;
}

Eigen::Matrix<double, 1, 4> get_psi_J(double lambd, double mu, const Eigen::Matrix<double, 2, 2>& grad_u) {
		double u11 = grad_u(0, 0);
    double u12 = grad_u(0, 1);
    double u21 = grad_u(1, 0);
    double u22 = grad_u(1, 1);

    Eigen::Matrix<double, 1, 4> psi_J;
		psi_J <<
				lambd*pow(u11, 3) + 3*lambd*pow(u11, 2) + lambd*u11*pow(u12, 2) + lambd*u11*pow(u21, 2) 
        + lambd*u11*pow(u22, 2) + 2*lambd*u11*u22 + 2*lambd*u11 + lambd*pow(u12, 2) 
        + lambd*pow(u21, 2) + lambd*pow(u22, 2) + 2*lambd*u22 + 2*mu*pow(u11, 3) 
        + 6*mu*pow(u11, 2) + 2*mu*u11*pow(u12, 2) + 2*mu*u11*pow(u21, 2) + 4*mu*u11 
        + 2*mu*pow(u12, 2) + 2*mu*u12*u21*u22 + 2*mu*u12*u21 + 2*mu*pow(u21, 2), 
        
        lambd*pow(u11, 2)*u12 + 2*lambd*u11*u12 + lambd*pow(u12, 3) + lambd*u12*pow(u21, 2) 
        + lambd*u12*pow(u22, 2) + 2*lambd*u12*u22 + 2*mu*pow(u11, 2)*u12 + 4*mu*u11*u12 
        + 2*mu*u11*u21*u22 + 2*mu*u11*u21 + 2*mu*pow(u12, 3) + 2*mu*u12*pow(u22, 2) 
        + 4*mu*u12*u22 + 2*mu*u12 + 2*mu*u21*u22 + 2*mu*u21, 
        
        lambd*pow(u11, 2)*u21 
        + 2*lambd*u11*u21 + lambd*pow(u12, 2)*u21 + lambd*pow(u21, 3) + lambd*u21*pow(u22, 2) 
        + 2*lambd*u21*u22 + 2*mu*pow(u11, 2)*u21 + 2*mu*u11*u12*u22 + 2*mu*u11*u12 
        + 4*mu*u11*u21 + 2*mu*u12*u22 + 2*mu*u12 + 2*mu*pow(u21, 3) + 2*mu*u21*pow(u22, 2) 
        + 4*mu*u21*u22 + 2*mu*u21, 
        
        lambd*pow(u11, 2)*u22 + lambd*pow(u11, 2) + 2*lambd*u11*u22 
        + 2*lambd*u11 + lambd*pow(u12, 2)*u22 + lambd*pow(u12, 2) + lambd*pow(u21, 2)*u22 + lambd*pow(u21, 2) 
        + lambd*pow(u22, 3) + 3*lambd*pow(u22, 2) + 2*lambd*u22 + 2*mu*u11*u12*u21 
        + 2*mu*pow(u12, 2)*u22 + 2*mu*pow(u12, 2) + 2*mu*u12*u21 + 2*mu*pow(u21, 2)*u22 
        + 2*mu*pow(u21, 2) + 2*mu*pow(u22, 3) + 6*mu*pow(u22, 2) + 4*mu*u22;

		return psi_J;
}

Eigen::Matrix<double, 4, 4> get_psi_H(double lambd, double mu, const Eigen::Matrix<double, 2, 2>& grad_u) {
		double u11 = grad_u(0, 0);
		double u12 = grad_u(0, 1);
		double u21 = grad_u(1, 0);
		double u22 = grad_u(1, 1);

		Eigen::Matrix<double, 4, 4> psi_H;
		psi_H.row(0) <<
				3*lambd*pow(u11, 2) + 6*lambd*u11 + lambd*pow(u12, 2) + lambd*pow(u21, 2) + lambd*pow(u22, 2) + 2*lambd*u22 + 2*lambd + 6*mu*pow(u11, 2) +
				12*mu*u11 + 2*mu*pow(u12, 2) + 2*mu*pow(u21, 2) + 4*mu, 2*lambd*u11*u12 + 2*lambd*u12 + 4*mu*u11*u12 + 4*mu*u12 + 2*mu*u21*u22 +
				2*mu*u21, 2*lambd*u11*u21 + 2*lambd*u21 + 4*mu*u11*u21 + 2*mu*u12*u22 + 2*mu*u12 + 4*mu*u21, 2*lambd*u11*u22 +
				2*lambd*u11 + 2*lambd*u22 + 2*lambd + 2*mu*u12*u21;

		psi_H.row(1) <<
				2*lambd*u11*u12 + 2*lambd*u12 + 4*mu*u11*u12 + 4*mu*u12 + 2*mu*u21*u22 + 2*mu*u21, lambd*pow(u11, 2) + 2*lambd*u11 +
				3*lambd*pow(u12, 2) + lambd*pow(u21, 2) + lambd*pow(u22, 2) + 2*lambd*u22 + 2*mu*pow(u11, 2) + 4*mu*u11 + 6*mu*pow(u12, 2) + 2*mu*pow(u22, 2) +
				4*mu*u22 + 2*mu, 2*lambd*u12*u21 + 2*mu*u11*u22 + 2*mu*u11 + 2*mu*u22 + 2*mu, 2*lambd*u12*u22 + 2*lambd*u12 +
				2*mu*u11*u21 + 4*mu*u12*u22 + 4*mu*u12 + 2*mu*u21;

		psi_H.row(2) <<
				2*lambd*u11*u21 + 2*lambd*u21 + 4*mu*u11*u21 + 2*mu*u12*u22 + 2*mu*u12 + 4*mu*u21, 2*lambd*u12*u21 + 2*mu*u11*u22 +
				2*mu*u11 + 2*mu*u22 + 2*mu, lambd*pow(u11, 2) + 2*lambd*u11 + lambd*pow(u12, 2) + 3*lambd*pow(u21, 2) + lambd*pow(u22, 2) + 2*lambd*u22 +
				2*mu*pow(u11, 2) + 4*mu*u11 + 6*mu*pow(u21, 2) + 2*mu*pow(u22, 2) + 4*mu*u22 + 2*mu, 2*lambd*u21*u22 + 2*lambd*u21 + 2*mu*u11*u12 +
				2*mu*u12 + 4*mu*u21*u22 + 4*mu*u21;

		psi_H.row(3) <<
				2*lambd*u11*u22 + 2*lambd*u11 + 2*lambd*u22 + 2*lambd + 2*mu*u12*u21, 2*lambd*u12*u22 + 2*lambd*u12 + 2*mu*u11*u21 +
				4*mu*u12*u22 + 4*mu*u12 + 2*mu*u21, 2*lambd*u21*u22 + 2*lambd*u21 + 2*mu*u11*u12 + 2*mu*u12 + 4*mu*u21*u22 +
				4*mu*u21, lambd*pow(u11, 2) + 2*lambd*u11 + lambd*pow(u12, 2) + lambd*pow(u21, 2) + 3*lambd*pow(u22, 2) + 6*lambd*u22 + 2*lambd +
				2*mu*pow(u12, 2) + 2*mu*pow(u21, 2) + 6*mu*pow(u22, 2) + 12*mu*u22 + 4*mu;

    return psi_H;
}


double get_grad_phi_det(const Eigen::Matrix<double, 8, 1>& params, const Eigen::Matrix<double, 2, 1>& isocoords) {
    const double xi = isocoords(0,0);
		const double eta = isocoords(1,0);
		const double x1 = params(0,0);
		const double y1 = params(1,0);
		const double x2 = params(2,0);
		const double y2 = params(3,0);
		const double x3 = params(4,0);
		const double y3 = params(5,0);
		const double x4 = params(6,0);
		const double y4 = params(7,0);

    return -1.0/8*eta*x1*y2 + (1.0/8)*eta*x1*y3 + (1.0/8)*eta*x2*y1 - 1.0/8*eta*x2*y4 - 1.0/8*eta*x3*y1 +
				(1.0/8)*eta*x3*y4 + (1.0/8)*eta*x4*y2 - 1.0/8*eta*x4*y3 - 1.0/8*xi*x1*y3 + (1.0/8)*xi*x1*y4 +
				(1.0/8)*xi*x2*y3 - 1.0/8*xi*x2*y4 + (1.0/8)*xi*x3*y1 - 1.0/8*xi*x3*y2 - 1.0/8*xi*x4*y1 + (1.0/8)*xi*x4*y2 +
				(1.0/8)*x1*y2 - 1.0/8*x1*y4 - 1.0/8*x2*y1 + (1.0/8)*x2*y3 - 1.0/8*x3*y2 + (1.0/8)*x3*y4 + (1.0/8)*x4*y1 - 1.0/8*x4*y3;
}

Eigen::Matrix<double, 2, 2> get_grad_phi_inv(const Eigen::Matrix<double, 8, 1>& params, const Eigen::Matrix<double, 2, 1>& isocoords) {
    const double xi = isocoords(0,0);
		const double eta = isocoords(1,0);
		const double x1 = params(0,0);
		const double y1 = params(1,0);
		const double x2 = params(2,0);
		const double y2 = params(3,0);
		const double x3 = params(4,0);
		const double y3 = params(5,0);
		const double x4 = params(6,0);
		const double y4 = params(7,0);
		
    const double det = get_grad_phi_det(params, isocoords);

		Eigen::Matrix<double, 2, 2> result;
		result.row(0) << -1.0/4*y1*(1 - xi) - 1.0/4*y2*(xi + 1) + (1.0/4)*y3*(xi + 1) + (1.0/4)*y4*(1 - xi), (1.0/4)*x1*(1 - xi) + (1.0/4)*x2*(xi + 1) - 1.0/4*x3*(xi + 1) - 1.0/4*x4*(1 - xi);
		result.row(1) << (1.0/4)*y1*(1 - eta) - 1.0/4*y2*(1 - eta) - 1.0/4*y3*(eta + 1) + (1.0/4)*y4*(eta + 1), -1.0/4*x1*(1 - eta) + (1.0/4)*x2*(1 - eta) + (1.0/4)*x3*(eta + 1) - 1.0/4*x4*(eta + 1);
		result *= (1.0/det);
		
    return result;
}

Eigen::Matrix<double, 2, 2> get_grad_phi(const Eigen::Matrix<double, 8, 1>& params, const Eigen::Matrix<double, 2, 1>& isocoords) {
		const double xi = isocoords(0,0);
		const double eta = isocoords(1,0);
		const double x1 = params(0,0);
		const double y1 = params(1,0);
		const double x2 = params(2,0);
		const double y2 = params(3,0);
		const double x3 = params(4,0);
		const double y3 = params(5,0);
		const double x4 = params(6,0);
		const double y4 = params(7,0);

		Eigen::Matrix<double, 2, 2> result;
		result.row(0) << -1.0/4*x1*(1 - eta) + (1.0/4)*x2*(1 - eta) + (1.0/4)*x3*(eta + 1) - 1.0/4*x4*(eta + 1), -1.0/4*x1*(1 - xi) - 1.0/4*x2*(xi + 1) + (1.0/4)*x3*(xi + 1) + (1.0/4)*x4*(1 - xi);
		result.row(1) << -1.0/4*y1*(1 - eta) + (1.0/4)*y2*(1 - eta) + (1.0/4)*y3*(eta + 1) - 1.0/4*y4*(eta + 1), -1.0/4*y1*(1 - xi) - 1.0/4*y2*(xi + 1) + (1.0/4)*y3*(xi + 1) + (1.0/4)*y4*(1 - xi);
		return result;
}

Eigen::Matrix<double, 2, 1> get_phi(const Eigen::Matrix<double, 8, 1>& params, const Eigen::Matrix<double, 2, 1>& isocoords) {
		const double xi = isocoords(0,0);
		const double eta = isocoords(1,0);
		const double x1 = params(0,0);
		const double y1 = params(1,0);
		const double x2 = params(2,0);
		const double y2 = params(3,0);
		const double x3 = params(4,0);
		const double y3 = params(5,0);
		const double x4 = params(6,0);
		const double y4 = params(7,0);

		Eigen::Matrix<double, 2, 1> phi;
		phi << (1.0/4)*x1*(1 - eta)*(1 - xi) + (1.0/4)*x2*(1 - eta)*(xi + 1) + (1.0/4)*x3*(eta + 1)*(xi + 1) + (1.0/4)*x4*(1 - xi)*(eta + 1),
				(1.0/4)*y1*(1 - eta)*(1 - xi) + (1.0/4)*y2*(1 - eta)*(xi + 1) + (1.0/4)*y3*(eta + 1)*(xi + 1) + (1.0/4)*y4*(1 - xi)*(eta + 1);
		return phi;
}

Eigen::Matrix<double, 2, 2> get_gradU(const Eigen::Matrix<double, 8, 1>& uparams, const Eigen::Matrix<double, 8, 1>& eparams, const Eigen::Matrix<double, 2, 1>& isocoords) {
		const double xi = isocoords(0,0);
		const double eta = isocoords(1,0);
		const auto grad_phiu = get_grad_phi(uparams, isocoords);
    const auto grad_phie_inv = get_grad_phi_inv(eparams, isocoords);
    return grad_phiu * grad_phie_inv;
}

// get the Second Piola Kirchoff Stress Tensor
Eigen::Matrix<double, 2, 2> get_S(const FemProblem& problem, const FemIteration& it,
																	Eigen::Matrix<double, 2, 1>(*get_prescribed_displacement)(const FemProblem&, int nx, int ny),
																	int nx, int ny, const Eigen::Matrix<double, 2, 1>& isocoords, Eigen::Matrix2d* gradUptr) {
		auto nxys = get_nxys(problem.num_ynodes, nx, ny);

 		// assemble the dofs from global to local
		Eigen::Matrix<double, 8, 1> uparams = get_uparams(problem, get_prescribed_displacement, it, nxys);
		Eigen::Matrix<double, 8, 1> eparams = get_eparams(problem, nxys);

		Eigen::Matrix2d& gradU = *gradUptr;
		gradU = get_gradU(uparams, eparams, isocoords);
		const Eigen::Matrix2d E = 0.5*(gradU + gradU.transpose() + gradU.transpose()*gradU);
		return problem.lambda * E.trace() * Eigen::Matrix2d::Identity() + 2*problem.mu * E;
}


Eigen::Matrix<double, 2, 1> local_to_global_deformed(const Eigen::Matrix<double, 8, 1>& uparams, const Eigen::Matrix<double, 8, 1>& eparams, const Eigen::Matrix<double, 2, 1>& isocoords) {
    return get_phi(eparams, isocoords) + get_phi(uparams, isocoords);
}
    
Eigen::Matrix<double, 2, 1> local_to_global_undeformed(const Eigen::Matrix<double, 8, 1>& eparams, const Eigen::Matrix<double, 2, 1>& isocoords) {
    return get_phi(eparams, isocoords);
}

Eigen::Matrix<double, 4, 8> get_B_square(const Eigen::Matrix<double, 8, 1>& eparams, const Eigen::Matrix<double, 2, 1>& isocoords) {
		const double xi = isocoords(0,0); 
		const double eta = isocoords(1,0);

		Eigen::Matrix<double, 2, 2> grad_phie_inv = get_grad_phi_inv(eparams, isocoords);
		const double f11 = grad_phie_inv(0,0);
		const double f12 = grad_phie_inv(0,1);
		const double f21 = grad_phie_inv(1,0);
		const double f22 = grad_phie_inv(1,1);

		Eigen::Matrix<double, 4, 8> result;
		result.row(0) <<
				f11*(0.25*eta - 0.25) + f21*(0.25*xi - 0.25), 0, f11*(0.25 - 0.25*eta) + f21*(-0.25*xi - 0.25), 0, 
				f11*(0.25*eta + 0.25) + f21*(0.25*xi + 0.25), 0, f11*(-0.25*eta - 0.25) + f21*(0.25 - 0.25*xi), 0;
		result.row(1) <<
        f12*(0.25*eta - 0.25) + f22*(0.25*xi - 0.25), 0, f12*(0.25 - 0.25*eta) + f22*(-0.25*xi - 0.25), 0, 
				f12*(0.25*eta + 0.25) + f22*(0.25*xi + 0.25), 0, f12*(-0.25*eta - 0.25) + f22*(0.25 - 0.25*xi), 0;
		result.row(2) <<
				0, f11*(0.25*eta - 0.25) + f21*(0.25*xi - 0.25), 0, f11*(0.25 - 0.25*eta) + f21*(-0.25*xi - 0.25), 0, 
				f11*(0.25*eta + 0.25) + f21*(0.25*xi + 0.25), 0, f11*(-0.25*eta - 0.25) + f21*(0.25 - 0.25*xi);
		result.row(3) <<
        0, f12*(0.25*eta - 0.25) + f22*(0.25*xi - 0.25), 0, f12*(0.25 - 0.25*eta) + f22*(-0.25*xi - 0.25), 0,
				f12*(0.25*eta + 0.25) + f22*(0.25*xi + 0.25), 0, f12*(-0.25*eta - 0.25) + f22*(0.25 - 0.25*xi);
    
    return result;
}
