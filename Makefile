# call "c:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" 
# inside developer shell?

imgui_widgets.obj: imgui\imgui_widgets.cpp
	nvcc imgui\imgui_widgets.cpp -O3 -c -o imgui_widgets.obj

imgui_tables.obj: imgui\imgui_tables.cpp
	nvcc imgui\imgui_tables.cpp -O3 -c -o imgui_tables.obj

imgui_demo.obj: imgui\imgui_demo.cpp
	nvcc imgui\imgui_demo.cpp -O3 -c -o imgui_demo.obj

imgui_draw.obj: imgui\imgui_draw.cpp
	nvcc imgui\imgui_draw.cpp -O3 -c -o imgui_draw.obj

imgui.obj: imgui\imgui.cpp
	nvcc imgui\imgui.cpp -O3 -c -o imgui.obj

implot.obj: implot\implot.cpp
	nvcc implot\implot.cpp -O3 -c -o implot.obj

implot_items.obj: implot\implot_items.cpp
	nvcc implot\implot_items.cpp -O3 -c -o implot_items.obj

implot_demo.obj: implot\implot_demo.cpp
	nvcc implot\implot_demo.cpp -O3 -c -o implot_demo.obj

implot: implot_demo.obj implot.obj implot_items.obj

imgui: imgui_widgets.obj imgui_tables.obj imgui_demo.obj imgui_draw.obj imgui.obj

ticktock.obj: ticktock.cc ticktock.h
	nvcc -std=c++17 -w ticktock.cc -O3 -c -o ticktock.obj

fem.obj: fem.cpp fem.h ticktock.obj
	nvcc -std=c++17 -w fem.cpp -O3 -c -o fem.obj 

test_fem: fem.obj fem.h
	nvcc -w fem.obj test_fem.cpp -O3 -o test_fem

main: imgui implot fem.obj fem.h ticktock.obj
	nvcc -std=c++17 -w --default-stream per-thread \
	imgui_tables.obj imgui_demo.obj imgui_draw.obj imgui.obj imgui_widgets.obj \
	implot_demo.obj implot.obj implot_items.obj \
	fem.obj \
	ticktock.obj \
  main.cpp -L COMMODE.OBJ -O3 -o main
