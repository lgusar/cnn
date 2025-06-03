#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define TRAINING_REPS 1000000

static float training_data[][3] = {
    {0, 0, 0},
    {0, 1, 0},
    {1, 0, 0},
    {1, 1, 1},
};

#define DATA_LEN sizeof(training_data) / sizeof(training_data[0])

// y = wx + b

float rand_float() {
    float r = (float)rand() / RAND_MAX;
    return r;
}

float sigmoid_float(float f) {
    return 1 / (1 + expf(-f));
}

float loss(float w1, float w2, float b) {
    float loss = 0;
    for (size_t i = 0; i < DATA_LEN; i++) {
        float x1 = training_data[i][0];
        float x2 = training_data[i][1];
        float expected = training_data[i][2];

        float y = sigmoid_float(w1 * x1 + w2 * x2 + b);
        float diff = y - expected;
        loss += diff * diff;
    }
    loss /= (float)DATA_LEN;
    return loss;
}

int main() {
    srand(time(0));

    float w1 = rand_float();
    float w2 = rand_float();
    float b = rand_float();

    printf("Start model:\n");
    printf("y = %f * x1 + %f * x2 + %f\n", w1, w2, b);

    float eps = 1e-3;
    float rate = 1e-3;

    for (size_t i = 0; i < TRAINING_REPS; i++) {
        float l = loss(w1, w2, b);
        // printf("Loss: %f\n", l);
        float dw1 = (loss(w1 + eps, w2, b) - l) / eps;
        float dw2 = (loss(w1, w2 + eps, b) - l) / eps;
        float db = (loss(w1, w2, b + eps) - l) / eps;

        w1 -= rate * dw1;
        w2 -= rate * dw2;
        b -= rate * db;
    }

    printf("Final model:\n");
    printf("y = %f * x1 + %f * x2 + %f\n", w1, w2, b);
    for (size_t i = 0; i < DATA_LEN; i++) {
        float x1 = training_data[i][0];
        float x2 = training_data[i][1];
        float y = sigmoid_float(w1 * x1 + w2 * x2 + b);

        printf("%f %f -> %f\n", x1, x2, y);
    }

    return 0;
}
