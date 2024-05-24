#include <metal_stdlib>
#include <simd/simd.h>

#import "ShaderTypes.h"

using namespace metal;

// Представляет трехмерный луч, который будет пересекаться со сценой. Тип луча
// настраивается с использованием свойств MPSRayIntersector.
struct Ray {
    // Начальная точка
    packed_float3 origin;
    
    // Маска, которая будет побитно И использоваться с масками треугольников для фильтрации
    // определенных пересечений. Это используется для того, чтобы источник света был видим
    // для камеры, но не для теневых или вторичных лучей.
    uint mask;
    
    // Направление, в котором движется луч
    packed_float3 direction;
    
    // Максимальное допустимое расстояние пересечения. Это используется для предотвращения
    // превышения источника света теневыми лучами при проверке видимости.
    float maxDistance;
    
    // Накопленный цвет вдоль пути луча
    float3 color;
};

// Представляет пересечение между лучом и сценой, возвращаемое MPSRayIntersector.
// Тип пересечения настраивается с использованием свойств MPSRayIntersector.
struct Intersection {
    // Расстояние от начала луча до точки пересечения. Отрицательное, если луч не пересекает сцену.
    float distance;
    
    // Индекс пересеченного примитива (треугольника), если таковой имеется. Неопределенный, если луч не пересекает сцену.
    int primitiveIndex;
    
    // Барицентрические координаты точки пересечения, если таковые имеются. Неопределенные, если луч не пересекает сцену.
    float2 coordinates;
};

// Генерирует лучи, начинающиеся из положения камеры и направленные к плоскости изображения,
// выровненной с системой координат камеры.
kernel void rayKernel(uint2 tid                    [[thread_position_in_grid]],
                      // Буферы, привязанные к CPU. Обратите внимание, что 'constant' следует использовать для небольших
                      // данных только для чтения, которые будут использоваться повторно в нескольких потоках. 'device' следует
                      // использовать для данных, доступных для записи или данных, которые будут использоваться только одним потоком.
                      constant Uniforms & uniforms [[buffer(0)]],
                      device Ray *rays          [[buffer(1)]],
                      device float2 *random,
                      texture2d<float, access::write> dstTex [[texture(0)]])
{
    // Поскольку мы выровняли количество потоков по размеру группы потоков, индекс потока может быть вне границ
    // размера целевого изображения.
    if (tid.x < uniforms.width && tid.y < uniforms.height) {
        // Вычисляем линейный индекс луча из 2D позиции
        unsigned int rayIdx = tid.y * uniforms.width + tid.x;

        // Луч, который мы создадим
        device Ray & ray = rays[rayIdx];

        // Координаты пикселя для этого потока
        float2 pixel = (float2)tid;

        // Добавляем случайное смещение к координатам пикселя для антиалиасинга
        float2 r = random[(tid.y % 16) * 16 + (tid.x % 16)];
        pixel += r;
        
        // Отображаем координаты пикселя в диапазон -1..1
        float2 uv = (float2)pixel / float2(uniforms.width, uniforms.height);
        uv = uv * 2.0f - 1.0f;
        
        constant Camera & camera = uniforms.camera;
        
        // Лучи начинаются из позиции камеры
        ray.origin = camera.position;
        
        // Отображаем нормализованные координаты пикселей в систему координат камеры
        ray.direction = normalize(uv.x * camera.right +
                                  uv.y * camera.up +
                                  camera.forward);
        // Камера излучает первичные лучи
        ray.mask = RAY_MASK_PRIMARY;
        
        // Не ограничиваем расстояние пересечения
        ray.maxDistance = INFINITY;
        
        // Начинаем с полностью белого цвета. Каждый рикошет будет масштабировать цвет по мере
        // поглощения света поверхностями.
        ray.color = float3(1.0f, 1.0f, 1.0f);
        
        // Очищаем целевое изображение до черного
        dstTex.write(float4(0.0f, 0.0f, 0.0f, 0.0f), tid);
    }
}

// Интерполирует атрибут вершины произвольного типа по поверхности треугольника,
// учитывая барицентрические координаты и индекс треугольника в структуре пересечения
template<typename T>
inline T interpolateVertexAttribute(device T *attributes, Intersection intersection) {
    // Барицентрические координаты суммируются до единицы
    float3 uvw;
    uvw.xy = intersection.coordinates;
    uvw.z = 1.0f - uvw.x - uvw.y;
    
    unsigned int triangleIndex = intersection.primitiveIndex;
    
    // Значения для каждой вершины
    T T0 = attributes[triangleIndex * 3 + 0];
    T T1 = attributes[triangleIndex * 3 + 1];
    T T2 = attributes[triangleIndex * 3 + 2];
    
    // Вычисляем сумму атрибутов вершин, взвешенных по барицентрическим координатам
    return uvw.x * T0 + uvw.y * T1 + uvw.z * T2;
}

// Использует метод инверсии для отображения двух равномерно случайных чисел в трехмерный
// единичный полушар, где вероятность данного образца пропорциональна косинусу
// угла между направлением образца и направлением "вверх" (0, 1, 0)
inline float3 sampleCosineWeightedHemisphere(float2 u) {
    float phi = 2.0f * M_PI_F * u.x;
    
    float cos_phi;
    float sin_phi = sincos(phi, cos_phi);
    
    float cos_theta = sqrt(u.y);
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    
    return float3(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);
}

// Отображает два равномерно случайных числа на поверхность двумерного источника света
// и возвращает направление к этой точке, количество света, которое проходит
// между точкой пересечения и точкой на источнике света, а также
// расстояние между этими двумя точками.
inline void sampleAreaLight(constant AreaLight & light,
                            float2 u,
                            float3 position,
                            thread float3 & lightDirection,
                            thread float3 & lightColor,
                            thread float & lightDistance)
{
    // Отображаем в диапазон -1..1
    u = u * 2.0f - 1.0f;
    
    // Преобразуем в систему координат источника света
    float3 samplePosition = light.position +
                            light.right * u.x +
                            light.up * u.y;
    
    // Вычисляем вектор от точки образца на источнике света до точки пересечения
    lightDirection = samplePosition - position;
    
    lightDistance = length(lightDirection);
    
    float inverseLightDistance = 1.0f / max(lightDistance, 1e-3f);
    
    // Нормализуем направление света
    lightDirection *= inverseLightDistance;
    
    // Начинаем с цвета источника света
    lightColor = light.color;
    
    // Свет затухает обратно пропорционально квадрату расстояния до точки пересечения
    lightColor *= (inverseLightDistance * inverseLightDistance);
    
    // Свет также затухает пропорционально косинусу угла между точкой пересечения и
    // источником света
    lightColor *= saturate(dot(-lightDirection, light.forward));
}

// Совмещает направление на единичном полушаре с тем, чтобы направление "вверх"
// полушара (0, 1, 0) отображалось на заданное направление нормали поверхности
inline float3 alignHemisphereWithNormal(float3 sample, float3 normal) {
    // Устанавливаем вектор "вверх" равным нормали
    float3 up = normal;
    
    // Находим произвольное направление, перпендикулярное нормали. Это станет
    // вектором "вправо".
    float3 right = normalize(cross(normal, float3(0.0072f, 1.0f, 0.0034f)));
    
    // Находим третий вектор, перпендикулярный предыдущим двум. Это будет
    // вектором "вперед".
    float3 forward = cross(right, up);
    
    // Отображаем направление на единичном полушаре на систему координат, выровненную
    // с нормалью.
    return sample.x * right + sample.y * up + sample.z * forward;
}

// Потребляет результаты пересечения луча и треугольника для вычисления затененного изображения
kernel void shadeKernel(uint2 tid [[thread_position_in_grid]],
                        constant Uniforms & uniforms,
                        device Ray *rays,
                        device Ray *shadowRays,
                        device Intersection *intersections,
                        device float3 *vertexColors,
                        device float3 *vertexNormals,
                        device float2 *random,
                        device uint *triangleMasks,
                        texture2d<float, access::write> dstTex)
{
    if (tid.x < uniforms.width && tid.y < uniforms.height) {
        unsigned int rayIdx = tid.y * uniforms.width + tid.x;
        device Ray & ray = rays[rayIdx];
        device Ray & shadowRay = shadowRays[rayIdx];
        device Intersection & intersection = intersections[rayIdx];
        
        float3 color = ray.color;
        
        // Расстояние пересечения будет отрицательным, если луч не пересекся или был отключен на предыдущей
        // итерации.
        if (ray.maxDistance >= 0.0f && intersection.distance >= 0.0f) {
            uint mask = triangleMasks[intersection.primitiveIndex];

            // Источник света включен в структуру ускорения, чтобы мы могли видеть его на
            // финальном изображении. Однако мы будем вычислять и образовать освещение напрямую, поэтому мы маскируем
            // источник света для теневых и вторичных лучей.
            if (mask == TRIANGLE_MASK_GEOMETRY) {
                // Вычисляем точку пересечения
                float3 intersectionPoint = ray.origin + ray.direction * intersection.distance;

                // Интерполируем нормаль вершины в точке пересечения
                float3 surfaceNormal = interpolateVertexAttribute(vertexNormals, intersection);
                surfaceNormal = normalize(surfaceNormal);

                // Считываем два равномерно случайных числа для этого потока
                float2 r = random[(tid.y % 16) * 16 + (tid.x % 16)];

                float3 lightDirection;
                float3 lightColor;
                float lightDistance;
                
                // Вычисляем направление к, цвет и расстояние до случайной точки на источнике света
                sampleAreaLight(uniforms.light, r, intersectionPoint, lightDirection,
                                lightColor, lightDistance);
                
                // Масштабируем цвет света по косинусу угла между направлением света и
                // нормалью поверхности
                lightColor *= saturate(dot(surfaceNormal, lightDirection));

                // Интерполируем цвет вершины в точке пересечения
                color *= interpolateVertexAttribute(vertexColors, intersection);
                
                // Вычисляем теневой луч. Теневой луч проверяет, видима ли точка выборки на
                // источнике света с точки пересечения, которую мы затеняем.
                // Если да, то вклад освещения, который мы только что вычислили, будет добавлен к
                // выходному изображению.
                
                // Добавляем небольшое смещение к точке пересечения, чтобы избежать пересечения с тем же
                // треугольником снова.
                shadowRay.origin = intersectionPoint + surfaceNormal * 1e-3f;
                
                // Направляемся к источнику света
                shadowRay.direction = lightDirection;
                
                // Избегаем пересечения самого источника света
                shadowRay.mask = RAY_MASK_SHADOW;
                
                // Не превышаем источник света
                shadowRay.maxDistance = lightDistance - 1e-3f;
                
                // Умножаем цвет и количество света в точке пересечения, чтобы получить окончательный
                // цвет, и передаем его вместе с теневым лучом, чтобы он мог быть добавлен к
                // выходному изображению, если необходимо.
                shadowRay.color = lightColor * color;
                
                // Далее мы выбираем случайное направление для продолжения пути луча. Это позволит
                // свету отражаться между поверхностями. Обычно мы бы применили множество математических операций
                // для вычисления доли света, отраженного текущей точкой пересечения к предыдущей точке от следующей точки.
                // Однако, выбирая случайное направление с вероятностью, пропорциональной косинусу (скалярное произведение) угла между направлением выборки и нормалью поверхности,
                // математика полностью отменяется, за исключением умножения на интерполированный цвет вершины.
                // Эта стратегия выборки также уменьшает количество шума на выходном изображении.
                float3 sampleDirection = sampleCosineWeightedHemisphere(r);
                sampleDirection = alignHemisphereWithNormal(sampleDirection, surfaceNormal);

                ray.origin = intersectionPoint + surfaceNormal * 1e-3f;
                ray.direction = sampleDirection;
                ray.color = color;
                ray.mask = RAY_MASK_SECONDARY;
            }
            else {
                // В этом случае луч, идущий от камеры, непосредственно попал в источник света,
                // поэтому мы запишем цвет света в выходное изображение.
                dstTex.write(float4(uniforms.light.color, 1.0f), tid);
                
                // Завершаем путь луча
                ray.maxDistance = -1.0f;
                shadowRay.maxDistance = -1.0f;
            }
        }
        else {
            // Луч не пересек сцену, поэтому завершаем путь луча
            ray.maxDistance = -1.0f;
            shadowRay.maxDistance = -1.0f;
        }
    }
}

// Проверяет, пересекся ли теневой луч с чем-либо на пути к источнику света. Если нет,
// то точка, из которой начался теневой луч, не была в тени, и ее цвет должен быть добавлен к выходному изображению.
kernel void shadowKernel(uint2 tid [[thread_position_in_grid]],
                         constant Uniforms & uniforms,
                         device Ray *shadowRays,
                         device float *intersections,
                         texture2d<float, access::read_write> dstTex)
{
    if (tid.x < uniforms.width && tid.y < uniforms.height) {
        unsigned int rayIdx = tid.y * uniforms.width + tid.x;
        device Ray & shadowRay = shadowRays[rayIdx];
        
        // Используем свойство MPSRayIntersection intersectionDataType для возврата
        // расстояния пересечения только для этого ядра. Вам не нужны другие поля,
        // поэтому вы сэкономите пропускную способность памяти.
        float intersectionDistance = intersections[rayIdx];
        
        // Если теневой луч не был отключен (максимальное расстояние >= 0) и он не пересекся с чем-либо
        // на пути к источнику света, добавляем цвет, переданный вместе с теневым лучом,
        // к выходному изображению.
        if (shadowRay.maxDistance >= 0.0f && intersectionDistance < 0.0f) {
            float3 color = shadowRay.color;
            
            color += dstTex.read(tid).xyz;
            
            // Записываем результат в целевое изображение
            dstTex.write(float4(color, 1.0f), tid);
        }
    }
}

// Накопляет изображение текущего кадра со средним значением всех предыдущих кадров для
// уменьшения шума со временем.
kernel void accumulateKernel(uint2 tid [[thread_position_in_grid]],
                             constant Uniforms & uniforms,
                             texture2d<float> renderTex,
                             texture2d<float, access::read_write> accumTex)
{
    if (tid.x < uniforms.width && tid.y < uniforms.height) {
        float3 color = renderTex.read(tid).xyz;

        // Вычисляем среднее значение всех кадров, включая текущий кадр
        if (uniforms.frameIndex > 0) {
            float3 prevColor = accumTex.read(tid).xyz;
            prevColor *= uniforms.frameIndex;
            
            color += prevColor;
            color /= (uniforms.frameIndex + 1);
        }
        
        accumTex.write(float4(color, 1.0f), tid);
    }
}

// Прямоугольник, заполняющий экран, в нормализованных координатах устройства
constant float2 quadVertices[] = {
    float2(-1, -1),
    float2(-1,  1),
    float2( 1,  1),
    float2(-1, -1),
    float2( 1,  1),
    float2( 1, -1)
};

struct CopyVertexOut {
    float4 position [[position]];
    float2 uv;
};

// Простой вершинный шейдер, который пропускает позиции квадрата NDC
vertex CopyVertexOut copyVertex(unsigned short vid [[vertex_id]]) {
    float2 position = quadVertices[vid];
    
    CopyVertexOut out;
    
    out.position = float4(position, 0, 1);
    out.uv = position * 0.5f + 0.5f;
    
    return out;
}

// Простой фрагментный шейдер, который копирует текстуру и применяет простую функцию тональной компрессии
fragment float4 copyFragment(CopyVertexOut in [[stage_in]],
                             texture2d<float> tex)
{
    constexpr sampler sam(min_filter::nearest, mag_filter::nearest, mip_filter::none);
    
    float3 color = tex.sample(sam, in.uv).xyz;
    
    // Применяем очень простую функцию тональной компрессии, чтобы уменьшить динамический диапазон
    // входного изображения в диапазон, который можно отобразить на экране.
    color = color / (1.0f + color);
    
    return float4(color, 1.0f);
}
