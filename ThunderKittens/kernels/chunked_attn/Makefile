# GPU Selection: 4090, A100, H100
GPU_TARGET=H100

# Compiler
NVCC?=nvcc

# Hacky enabling H100 TK features so it compiles :(
NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE -Xcompiler -fopenmp --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3  -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -MD -MT -MF -x cu -lrt -lpthread -ldl -DKITTENS_HOPPER -gencode arch=compute_90a,code=sm_90a -lcuda -lcudadevrt -lcudart_static -lcublas -lgomp -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype # H100

# Targets
TARGET_CF=attn_chunk_first
TARGET_SF=attn_seq_first
SRC_CF=attn_chunk_first.cu
SRC_SF=attn_seq_first.cu

# Default target: build both
all: $(TARGET_CF) $(TARGET_SF)

# Chunk-first kernel
$(TARGET_CF): $(SRC_CF)
	$(NVCC) $(SRC_CF) $(NVCCFLAGS) -o $(TARGET_CF)

# Seq-first kernel
$(TARGET_SF): $(SRC_SF)
	$(NVCC) $(SRC_SF) $(NVCCFLAGS) -o $(TARGET_SF)

# Clean target
clean:
	rm -f $(TARGET_CF) $(TARGET_SF) *.d
