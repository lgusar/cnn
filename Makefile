CFLAGS = -Wall -Wextra -Wpedantic -ggdb

all: src/main.c
	mkdir -p build
	gcc $(CFLAGS) -o build/main src/main.c

clean:
	rm -rf build
