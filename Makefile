CFLAGS = -Wall -Wextra -Wpedantic -ggdb
LFLAGS = -lm

all: src/main.c
	mkdir -p build
	gcc $(CFLAGS) -o build/main src/main.c $(LFLAGS)

clean:
	rm -rf build
