
all: add

add: add.cu
	/usr/local/cuda-8.0/bin/nvcc -I"/usr/local/cuda-8.0/samples/0_Simple" -I"/usr/local/cuda-8.0/samples/common/inc" -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_21  -x cu -o  "add.o" "add.cu"
	/usr/local/cuda-8.0/bin/nvcc --cudart static --relocatable-device-code=false -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_21 -link -o  "add"  add.o   


clean:
	rm -fr *.o add

