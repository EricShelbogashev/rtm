import simd

// Структура Matrix4x4, предоставляющая статические методы для создания матриц преобразований
struct Matrix4x4 {

    // Метод для создания матрицы трансляции
    // x, y, z - координаты сдвига по осям x, y и z соответственно
    static func translation(_ x: Float, _ y: Float, _ z: Float) -> matrix_float4x4 {
        // Создаем и возвращаем матрицу трансляции
        // 1 0 0 x
        // 0 1 0 y
        // 0 0 1 z
        // 0 0 0 1
        return matrix_float4x4(rows: [SIMD4<Float>([1, 0, 0, x]),
                                      SIMD4<Float>([0, 1, 0, y]),
                                      SIMD4<Float>([0, 0, 1, z]),
                                      SIMD4<Float>([0, 0, 0, 1])])
    }

    // Метод для создания матрицы масштабирования
    // x, y, z - коэффициенты масштабирования по осям x, y и z соответственно
    static func scale(_ x: Float, _ y: Float, _ z: Float) -> matrix_float4x4 {
        // Создаем и возвращаем диагональную матрицу масштабирования
        // x 0 0 0
        // 0 y 0 0
        // 0 0 z 0
        // 0 0 0 1
        return matrix_float4x4(diagonal: SIMD4<Float>([x, y, z, 1]))
    }

    // Метод для создания матрицы вращения
    // radians - угол вращения в радианах
    // axis - вектор, вокруг которого происходит вращение (должен быть нормализован)
    static func rotation(radians: Float, axis: SIMD3<Float>) -> matrix_float4x4 {
        // Нормализуем ось вращения
        let unitAxis = normalize(axis)
        // Вычисляем косинус и синус угла вращения
        let ct = cosf(radians)
        let st = sinf(radians)
        // Вычисляем 1 - косинус угла
        let ci = 1 - ct
        // Извлекаем компоненты нормализованного вектора оси
        let x = unitAxis.x, y = unitAxis.y, z = unitAxis.z

        // Создаем и возвращаем матрицу вращения по заданной оси
        // ct + x * x * ci     y * x * ci + z * st     z * x * ci - y * st     0
        // x * y * ci - z * st     ct + y * y * ci     z * y * ci + x * st     0
        // x * z * ci + y * st     y * z * ci - x * st     ct + z * z * ci     0
        // 0                       0                       0                   1
        return matrix_float4x4(columns:(SIMD4<Float>(    ct + x * x * ci, y * x * ci + z * st, z * x * ci - y * st, 0),
                                        SIMD4<Float>(x * y * ci - z * st,     ct + y * y * ci, z * y * ci + x * st, 0),
                                        SIMD4<Float>(x * z * ci + y * st, y * z * ci - x * st,     ct + z * z * ci, 0),
                                        SIMD4<Float>(                  0,                   0,                   0, 1)))
    }
}
