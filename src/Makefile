# Project 3: Real-time 2D Object Recognition
# Shivang Patel (shivang2402)

CC = clang++
CXX = $(CC)

CFLAGS = -std=c++17 -Wall -I/opt/homebrew/opt/opencv/include/opencv4 -I/opt/homebrew/opt/onnxruntime/include -I../include
CXXFLAGS = $(CFLAGS)

LDFLAGS = -L/opt/homebrew/opt/opencv/lib -L/opt/homebrew/opt/onnxruntime/lib -Wl,-rpath,/opt/homebrew/opt/onnxruntime/lib
LDLIBS = -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_dnn -lonnxruntime

BINDIR = ../bin

objrec: main.o vision.o features.o utilities.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f *.o *~