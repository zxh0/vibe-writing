CC = clang
CFLAGS = -O2 -mavx2 -mfma
all: peak bandwidth
peak: peak.c
	$(CC) $(CFLAGS) -o $@ $<
bandwidth: bandwidth.c
	$(CC) $(CFLAGS) -o $@ $<
clean:
	rm -f peak bandwidth
