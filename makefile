SOURCE = sobel.cu\
	lodepng.cpp

OBJS = $(SOURCE: .cpp=.o)\
	$(SOURCE: .cu=.o)

GCC = nvcc
LINK = nvcc

CFLAGS = -std=c++11 -Xcompiler -fopenmp

TARGET = edge

all: $(TARGET)

$(TARGET): $(OBJS)
		$(LINK) -o $@ $^ -std=c++11 -Xcompiler -fopenmp

clean:
	rm -rf $(TARGET) *.o *.d
