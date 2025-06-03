#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define TRAINING_REPS 1000000 

static float training_data[2][2] = {
    {0, 1},
    {1, 0},
};

#define DATA_LEN sizeof(training_data) / sizeof(training_data[0])

// y = wx + b

float rand_float() { return (float)rand() / RAND_MAX; }

float loss(float w, float b) {
    float loss = 0;
    for (size_t i = 0; i < DATA_LEN; i++) {
        float x = training_data[i][0];
        float expected = training_data[i][1];
        float y = w * x + b;
        float diff = y - expected;
        loss += diff * diff;
    }
    loss /= (float)DATA_LEN;
    return loss;
}

int main() {
    // srand(time(0));
    srand(0);

    float w = rand_float();
    float b = rand_float();

    float eps = 1e-2;
    float rate = 1e-2;

    for (size_t i = 0; i < TRAINING_REPS; i++) {
        float l = loss(w, b);
        printf("Loss: %f\n", l);
        float dw = (loss(w + eps, b) - l) / eps;
        float db = (loss(w, b + eps) - l) / eps;

        w -= rate * dw;
        b -= rate * db;
    }

    printf("Final model:\n");
    printf("y = %f * x + %f\n", w, b);
    for(size_t i = 0; i < DATA_LEN; i++) {
        printf("%f -> %f\n", training_data[i][0], w * training_data[i][0] + b);
    }

    return 0;
}
