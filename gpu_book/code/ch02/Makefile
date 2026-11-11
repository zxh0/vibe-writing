# matmul_O0：不开优化；matmul：-O2；matmul_simd：-O3，并允许使用本机的AVX2和FMA指令
# 注意：clang在-O2下默认也会做自动向量化，所以matmul用-fno-vectorize关掉它，作为“不用SIMD”的对照
CC = clang
all: matmul_O0 matmul matmul_simd
matmul_O0: matmul.c
	$(CC) -O0 -o $@ $<
matmul: matmul.c
	$(CC) -O2 -fno-vectorize -fno-slp-vectorize -o $@ $<
matmul_simd: matmul.c
	$(CC) -O3 -march=native -o $@ $<
clean:
	rm -f matmul_O0 matmul matmul_simd
