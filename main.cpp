#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>
#include <random>

#include "Eigen/Dense"

#define SOKOL_VALIDATE_NON_FATAL
#define SOKOL_IMPL
#define SOKOL_GLCORE33
#include "sokol/sokol_app.h"
#include "sokol/sokol_gfx.h"
#include "sokol/sokol_glue.h"
#include "sokol/sokol_time.h"
#include "sokol/util/sokol_shape.h"
#include "imgui/imgui.h"
#include "sokol/util/sokol_imgui.h"

#include "implot/implot.h"

#include "fem.h"

#define M_PI 3.14159265358979323846

const ImVec4 red = ImVec4(0.8f, 0.1f, 0.1f, 1.0f);
const ImVec4 green = ImVec4(0.2f, 0.7f, 0.2f, 1.0f);
const ImVec4 blue = ImVec4(0.1f, 0.4f, 0.8f, 1.0f);
const ImVec4 orange = ImVec4(0.9f, 0.5f, 0.2f, 1.0f);
const ImVec4 purple = ImVec4(0.6f, 0.1f, 0.7f, 1.0f);
const ImVec4 yellow = ImVec4(0.9f, 0.9f, 0.2f, 1.0f);
const ImVec4 cyan = ImVec4(0.1f, 0.7f, 0.7f, 1.0f);
const ImVec4 magenta = ImVec4(0.7f, 0.1f, 0.5f, 1.0f);
const ImVec4 gray = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
const ImVec4 black = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
const ImVec4 white = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

FemProblem problem;
FemIteration problem_it {};

Eigen::Matrix<double, 2, 1> prescribed_displacement { 0, 0 };

sg_pass_action pass_action = {};

std::mt19937 gen;

bool log_file_initted = false;
std::fstream log_file;

void init_file_logging() {
    std::time_t current_time = std::time(nullptr);
    const std::string run_id = std::to_string(current_time);    
    const std::string log_path = run_id + ".log.txt";
    std::cout << "Logging to " << log_path << "\n";
    log_file = std::fstream(log_path, std::fstream::out);
    if (!log_file.is_open()) {
        std::cout << "Could not open log file for writing\n";
        exit(-1);
    }
    log_file_initted = true;
}

void init_cb(void) {
    stm_setup();

    sg_desc desc = {};
    desc.context = sapp_sgcontext();
    sg_setup(&desc);

    simgui_desc_t imgui_desc = {};
    simgui_setup(&imgui_desc);

		ImPlot::CreateContext();
    
    pass_action = {};
    pass_action.colors[0].action = SG_ACTION_CLEAR;
    pass_action.colors[0].value = {0.0f, 0.0f, 0.0f, 1.0f};

    const bool static_seed = false;
    if (static_seed) {
        std::default_random_engine eng{10 /*seed*/};
        gen  = std::mt19937{eng}; //Standard mersenne_twister_engine seeded with rd()
    } else {
        std::random_device rd;
        gen = std::mt19937{rd()}; //Standard mersenne_twister_engine seeded with rd()
    }

		double r0 = 0.2;
		double r1 = 1.0;
		int num_rnodes = 50;
		int num_thetanodes = 50;
		problem = create_polar_problem(r0, r1, num_rnodes, num_thetanodes);
		problem.user_data = (void*)(&prescribed_displacement);
		problem_it = create_fem_it(problem);
}

void debug_gui() {
		ImPlot::ShowDemoWindow();

		ImGui::SetNextWindowSize({ 1000, 700 }, ImGuiCond_FirstUseEver);
    ImGui::Begin("Debug");
		ImGui::Text("Pi: %f", problem_it.Pi);

		static bool optimizing = false;

		static float dx = (float)(prescribed_displacement(0,0));
		static float dy = (float)(prescribed_displacement(1,0));

		static float dr = 0;
		static float dtheta = 0

		const bool dx_changed = ImGui::SliderFloat("dx",&dx, -0.5f, 0.5f);
		const bool dy_changed = ImGui::SliderFloat("dy",&dy, -0.5f, 0.5f);

		const bool dr_changed = ImGui::SliderFloat("dr",&dr, -0.1f, 0.1f);
		const bool dtheta_changed = ImGui::SliderFloat("dtheta",&dtheta, -M_PI/3, M_PI/3);


		// partial prescribed displacement for displacement control
		static Eigen::Vector2d prescribed_displacement_goal = Eigen::Vector2d::Zero();
		static Eigen::Vector2d prescribed_displacement_start = Eigen::Vector2d::Zero();
		static double displacement_progress = 0;
		static float displacement_step = 0.1;

		ImGui::SliderFloat("Displacement step", &displacement_step, 0.01, 0.2);

		bool set_goal = dr_changed || dtheta_changed;

		if (ImGui::Button("Reset")) {
				problem_it.reset();
				prescribed_displacement.setZero();
				set_goal = true;
		}

		if (set_goal) {
				// update the goal
				prescribed_displacement_start = prescribed_displacement;
				prescribed_displacement_goal << dx, dy;
				displacement_progress = 0;
		}

		ImGui::Checkbox("Optimize", &optimizing);
		if (optimizing) {
				ImGui::SameLine();
				ImGui::Text("Currently target displacement %f, %f", prescribed_displacement(0), prescribed_displacement(1));
				ImGui::SameLine();
				ImGui::Text("Total progress %f", displacement_progress);

				update_problem(problem, get_polar_prescribed_displacement, problem_it);
				if (std::abs(problem_it.last_change) < 1e-3) {
						if (displacement_progress >= 1.0) {
								optimizing = false;
						} else {
								// update the displacement progress
								displacement_progress += displacement_step;
								prescribed_displacement = displacement_progress*prescribed_displacement_goal + (1-displacement_progress)*prescribed_displacement_start;
						}
				}
		};

		static std::vector<double> xs;
		static std::vector<double> ys;
		xs.clear();
		ys.clear();

		ImPlot::SetNextAxesLimits(-1.1, 1.1, -1.1, 1.1);
		if(ImPlot::BeginPlot("Visualization", nullptr, nullptr, {700, 700})) {
				ImPlot::SetupAxes("x","y");

				// vertical lines
				for (int xi = 0; xi < problem.num_xnodes; ++xi) {
						xs.clear();
						ys.clear();

						int ymax = problem.is_polar ? problem.num_ynodes+1 : problem.num_ynodes;
						for (int yi = 0; yi < ymax; ++yi) {
								auto xy = get_deformed_coordinates(
										problem, get_polar_prescribed_displacement, problem_it, xi, yi%problem.num_ynodes);
								xs.push_back(xy(0));
								ys.push_back(xy(1));
						}
						ImPlot::PlotLine("Contour (deformed)", xs.data(), ys.data(), xs.size());
				}

				// horizontal lines
				for (int yi = 0; yi < problem.num_ynodes; ++yi) {
						xs.clear();
						ys.clear();
						for (int xi = 0; xi < problem.num_xnodes; ++xi) {				
								auto xy = get_deformed_coordinates(
										problem, get_polar_prescribed_displacement, problem_it, xi, yi);
								xs.push_back(xy(0));
								ys.push_back(xy(1));
						}
						ImPlot::PlotLine("Contour (deformed)", xs.data(), ys.data(), xs.size());
				}


				auto current = get_deformed_coordinates(problem,
																								get_polar_prescribed_displacement,
																								problem_it,
																								problem.num_xnodes-1, 0).eval();
				float current_x = current(0);
				float current_y = current(1);

				auto target = get_undeformed_coordinates(problem, problem.num_xnodes-1, 0).eval();
				target += prescribed_displacement_goal;

				float target_x = target(0);
				float target_y = target(1);
				// ImPlot::PushStyleColor(ImPlotCol_MarkerFill, red);
				// ImPlot::PushStyleColor(ImPlotCol_MarkerOutline, red);
				ImPlot::PlotScatter("Prescribed Target", &target_x, &target_y, 1);
				// ImPlot::PopStyleColor();
				// ImPlot::PopStyleColor();

				// ImPlot::PushStyleColor(ImPlotCol_MarkerFill, white);
				ImPlot::PlotScatter("Attachment Point", &current_x, &current_y, 1);
				// ImPlot::PopStyleColor();


				ImPlot::EndPlot();
		}

    ImGui::End();
}

void frame_cb(void) {
    simgui_frame_desc_t frame_desc = {};
    frame_desc.width = sapp_width();
    frame_desc.height = sapp_height();
    frame_desc.delta_time = sapp_frame_duration();
    frame_desc.dpi_scale = sapp_dpi_scale();
    simgui_new_frame(&frame_desc);

    static bool should_visualize = false;
    static bool should_log = true;
    static int last_logged_round = -1;

    debug_gui(); 
    sg_begin_default_pass(&pass_action, sapp_width(), sapp_height());

    simgui_render();
    sg_end_pass();
    sg_commit();
}

void event_cb(const sapp_event* event) {
    simgui_handle_event(event);
};

void cleanup_cb(void) {
		ImPlot::DestroyContext();
    simgui_shutdown();
    sg_shutdown();
}

sapp_desc sokol_main(int argc, char* argv[]) { 
    sapp_desc desc = {};
    desc.init_cb = init_cb;
    desc.frame_cb = frame_cb;
    desc.event_cb = event_cb;
    desc.cleanup_cb = cleanup_cb;
    desc.width = 1280;
    desc.height = 720;
    desc.window_title = "Finite Element 2 Project: Rotational Flexure";
    desc.win32_console_create = true;
    return desc;
}
