#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

// Маски для треугольников
#define TRIANGLE_MASK_GEOMETRY 1
#define TRIANGLE_MASK_LIGHT    2

// Маски для лучей
#define RAY_MASK_PRIMARY   3
#define RAY_MASK_SHADOW    1
#define RAY_MASK_SECONDARY 1

// Структура, представляющая камеру
struct Camera {
    vector_float3 position;  // Позиция камеры
    vector_float3 right;     // Вектор направления вправо
    vector_float3 up;        // Вектор направления вверх
    vector_float3 forward;   // Вектор направления вперед
};

// Структура, представляющая площадь источника света
struct AreaLight {
    vector_float3 position;  // Позиция источника света
    vector_float3 forward;   // Вектор направления вперед
    vector_float3 right;     // Вектор направления вправо
    vector_float3 up;        // Вектор направления вверх
    vector_float3 color;     // Цвет источника света
};

// Структура, содержащая униформы
struct Uniforms {
    unsigned int width;
    unsigned int height;
    unsigned int blocksWide;
    unsigned int frameIndex;
    struct Camera camera;
    struct AreaLight light;
    float cameraRotationAngle; // Новый параметр
};

#endif
