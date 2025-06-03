#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TRAINING_REPS 1000000

static float training_data[][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
};

struct model {
    float w11;
    float w12;
    float b1;

    float w21;
    float w22;
    float b2;

    float w31;
    float w32;
    float b3;
};

#define DATA_LEN sizeof(training_data) / sizeof(training_data[0])

float rand_float() {
    float r = (float)rand() / RAND_MAX;
    return r;
}

float sigmoid_float(float f) { return 1 / (1 + expf(-f)); }

float forward(struct model m, float x1, float x2) {
    float a1 = sigmoid_float(m.w11 * x1 + m.w12 * x2 + m.b1);
    float a2 = sigmoid_float(m.w21 * x1 + m.w22 * x2 + m.b2);

    return sigmoid_float(m.w31 * a1 + m.w32 * a2 + m.b3);
}

float loss(struct model m) {
    float loss = 0;
    for (size_t i = 0; i < DATA_LEN; i++) {
        float x1 = training_data[i][0];
        float x2 = training_data[i][1];
        float expected = training_data[i][2];

        float y = forward(m, x1, x2);
        float diff = y - expected;
        loss += diff * diff;
    }
    loss /= (float)DATA_LEN;
    return loss;
}

int main() {
    srand(time(0));

    struct model m;

    m.w11 = rand_float();
    m.w12 = rand_float();
    m.b1 = rand_float();

    m.w21 = rand_float();
    m.w22 = rand_float();
    m.b2 = rand_float();

    m.w31 = rand_float();
    m.w32 = rand_float();
    m.b3 = rand_float();

    float eps = 1e-2;
    float rate = 1e-2;

    for (size_t i = 0; i < TRAINING_REPS; i++) {
        float l = loss(m);

        struct model md = m;
        md.w11 += eps;
        m.w11 -= rate * (loss(md) - l) / eps;
        md.w11 -= eps;

        md.w12 += eps;
        m.w12 -= rate * (loss(md) - l) / eps;
        md.w12 -= eps;

        md.w21 += eps;
        m.w21 -= rate * (loss(md) - l) / eps;
        md.w21 -= eps;

        md.w22 += eps;
        m.w22 -= rate * (loss(md) - l) / eps;
        md.w22 -= eps;

        md.w31 += eps;
        m.w31 -= rate * (loss(md) - l) / eps;
        md.w31 -= eps;

        md.w32 += eps;
        m.w32 -= rate * (loss(md) - l) / eps;
        md.w32 -= eps;

        md.b1 += eps;
        m.b1 -= rate * (loss(md) - l) / eps;
        md.b1 -= eps;

        md.b2 += eps;
        m.b2 -= rate * (loss(md) - l) / eps;
        md.b2 -= eps;

        md.b3 += eps;
        m.b3 -= rate * (loss(md) - l) / eps;
        md.b3 -= eps;
    }

    printf("Final model:\n");
    for (size_t i = 0; i < DATA_LEN; i++) {
        float x1 = training_data[i][0];
        float x2 = training_data[i][1];
        float y = forward(m, x1, x2);

        printf("%f %f -> %f\n", x1, x2, y);
    }

    printf("Loss: %f\n", loss(m));

    return 0;
}
