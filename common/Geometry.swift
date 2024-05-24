import MetalKit
import MetalPerformanceShaders
import simd
import os

// Функция для создания сферы
func sphere(center: SIMD3<Float>,
            radius: Float,
            color: SIMD3<Float>,
            segments: Int,
            rings: Int,
            transform: matrix_float4x4,
            triangleMask: uint,
            vertices: inout [SIMD3<Float>],
            normals: inout [SIMD3<Float>],
            colours: inout [SIMD3<Float>],
            masks: inout [uint]) {

    // Временные массивы для уникальных вершин и индексов
    var tempVertices = [SIMD3<Float>]()
    var tempNormals = [SIMD3<Float>]()
    var tempColours = [SIMD3<Float>]()
    var tempMasks = [uint]()
    var indices = [UInt32]()

    // Создание вершин и нормалей сферы
    for i in 0...segments {
        let theta = Float(i) * (Float.pi * 2) / Float(segments)
        for j in 0...rings {
            let phi = Float(j) * Float.pi / Float(rings)

            let x = radius * sin(phi) * cos(theta)
            let y = radius * cos(phi)
            let z = radius * sin(phi) * sin(theta)

            let vertex = SIMD4<Float>(x, y, z, 1) * transform
            tempVertices.append(SIMD3<Float>(vertex.x, vertex.y, vertex.z) + center)
            tempNormals.append(normalize(SIMD3<Float>(x, y, z)))
            tempColours.append(color)
            tempMasks.append(triangleMask)
        }
    }

    // Создание треугольников с использованием индексов
    for i in 0..<segments {
        for j in 0..<rings {
            let first = UInt32(i * (rings + 1) + j)
            let second = first + UInt32(rings + 1)

            indices.append(contentsOf: [first, second, first + 1])
            indices.append(contentsOf: [first + 1, second, second + 1])
        }
    }

    // Добавление уникальных вершин и треугольников в основные массивы
    for index in indices {
        vertices.append(tempVertices[Int(index)])
        normals.append(tempNormals[Int(index)])
        colours.append(tempColours[Int(index)])
        masks.append(tempMasks[Int(index)])
    }
}

// Структура FaceMask, представляющая опции маски лица куба
struct FaceMask: OptionSet {
    let rawValue: UInt32

    static let negativeX = FaceMask(rawValue: 1 << 0)
    static let positiveX = FaceMask(rawValue: 1 << 1)
    static let negativeY = FaceMask(rawValue: 1 << 2)
    static let positiveY = FaceMask(rawValue: 1 << 3)
    static let negativeZ = FaceMask(rawValue: 1 << 4)
    static let positiveZ = FaceMask(rawValue: 1 << 5)
    static let all: FaceMask = [.negativeX, .negativeY, .negativeZ, .positiveX, .positiveY, .positiveZ]
}

// Вспомогательная функция для вычисления нормали треугольника
fileprivate func triangleNormal(v0: SIMD3<Float>, v1: SIMD3<Float>, v2: SIMD3<Float>) -> SIMD3<Float> {
    // Вычисление нормали треугольника через векторное произведение нормализованных векторов
    return cross(normalize(v1 - v0), normalize(v2 - v0))
}

// Вспомогательная функция для создания грани куба
fileprivate func cubeFace(withCubeVertices cubeVertices: [SIMD3<Float>],
                          colour: SIMD3<Float>,
                          index0: Int,
                          index1: Int,
                          index2: Int,
                          index3: Int,
                          inwardNormals: Bool,
                          triangleMask: uint,
                          vertices: inout [SIMD3<Float>],
                          normals: inout [SIMD3<Float>],
                          colours: inout [SIMD3<Float>],
                          masks: inout [uint]) {

    // Получение вершин грани по индексам
    let v0 = cubeVertices[index0]
    let v1 = cubeVertices[index1]
    let v2 = cubeVertices[index2]
    let v3 = cubeVertices[index3]

    // Вычисление нормалей для двух треугольников, составляющих грань
    var n0 = triangleNormal(v0: v0, v1: v1, v2: v2)
    var n1 = triangleNormal(v0: v0, v1: v2, v2: v3)
    if inwardNormals {
        n0 = -n0
        n1 = -n1
    }

    // Добавление вершин, нормалей, цветов и масок в соответствующие массивы
    vertices.append(contentsOf: [v0, v1, v2, v0, v2, v3])
    normals.append(contentsOf: [n0, n0, n0, n1, n1, n1])
    colours.append(contentsOf: [SIMD3<Float>](repeating: colour, count: 6))
    masks.append(contentsOf: [triangleMask, triangleMask])
}

// Функция для создания куба с указанной маской граней
func cube(withFaceMask faceMask: FaceMask,
          colour: SIMD3<Float>,
          transform: matrix_float4x4,
          inwardNormals: Bool,
          triangleMask: uint,
          vertices: inout [SIMD3<Float>],
          normals: inout [SIMD3<Float>],
          colours: inout [SIMD3<Float>],
          masks: inout [uint]) {

    // Определение вершин куба в локальной системе координат
    var cubeVertices = [
        SIMD3<Float>(-0.5, -0.5, -0.5),
        SIMD3<Float>( 0.5, -0.5, -0.5),
        SIMD3<Float>(-0.5,  0.5, -0.5),
        SIMD3<Float>( 0.5,  0.5, -0.5),
        SIMD3<Float>(-0.5, -0.5,  0.5),
        SIMD3<Float>( 0.5, -0.5,  0.5),
        SIMD3<Float>(-0.5,  0.5,  0.5),
        SIMD3<Float>( 0.5,  0.5,  0.5),
    ]

    // Преобразование вершин куба с использованием переданной матрицы трансформации
    cubeVertices = cubeVertices.map { vertex in
        var transformed = SIMD4<Float>(vertex.x, vertex.y, vertex.z, 1)
        transformed = transform * transformed
        return SIMD3<Float>(x: transformed.x, y: transformed.y, z: transformed.z)
    }

    // Создание граней куба в зависимости от маски
    if faceMask.contains(.negativeX) {
        cubeFace(withCubeVertices: cubeVertices, colour: colour,
                 index0: 0, index1: 4, index2: 6, index3: 2,
                 inwardNormals: inwardNormals, triangleMask: triangleMask,
                 vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)
    }

    if faceMask.contains(.positiveX) {
        cubeFace(withCubeVertices: cubeVertices, colour: colour,
                 index0: 1, index1: 3, index2: 7, index3: 5,
                 inwardNormals: inwardNormals, triangleMask: triangleMask,
                 vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)
    }

    if faceMask.contains(.negativeY) {
        cubeFace(withCubeVertices: cubeVertices, colour: colour,
                 index0: 0, index1: 1, index2: 5, index3: 4,
                 inwardNormals: inwardNormals, triangleMask: triangleMask,
                 vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)
    }

    if faceMask.contains(.positiveY) {
        cubeFace(withCubeVertices: cubeVertices, colour: colour,
                 index0: 2, index1: 6, index2: 7, index3: 3,
                 inwardNormals: inwardNormals, triangleMask: triangleMask,
                 vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)
    }

    if faceMask.contains(.negativeZ) {
        cubeFace(withCubeVertices: cubeVertices, colour: colour,
                 index0: 0, index1: 2, index2: 3, index3: 1,
                 inwardNormals: inwardNormals, triangleMask: triangleMask,
                 vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)
    }

    if faceMask.contains(.positiveZ) {
        cubeFace(withCubeVertices: cubeVertices, colour: colour,
                 index0: 4, index1: 5, index2: 7, index3: 6,
                 inwardNormals: inwardNormals, triangleMask: triangleMask,
                 vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)
    }
}

// Функция для создания верхней грани куба и видимых боковых граней
func addVisibleFaces(x: Int, y: Int, z: Float, width: Int, height: Int,
                     colour: SIMD3<Float>, transform: matrix_float4x4,
                     triangleMask: uint,
                     vertices: inout [SIMD3<Float>],
                     normals: inout [SIMD3<Float>],
                     colours: inout [SIMD3<Float>],
                     masks: inout [uint],
                     intensityMap: [[Float]]) {
    // Определение вершин куба в локальной системе координат
    var cubeVertices = [
        SIMD3<Float>(-0.5, -0.5, -0.5),
        SIMD3<Float>( 0.5, -0.5, -0.5),
        SIMD3<Float>(-0.5,  0.5, -0.5),
        SIMD3<Float>( 0.5,  0.5, -0.5),
        SIMD3<Float>(-0.5, -0.5,  0.5),
        SIMD3<Float>( 0.5, -0.5,  0.5),
        SIMD3<Float>(-0.5,  0.5,  0.5),
        SIMD3<Float>( 0.5,  0.5,  0.5),
    ]

    // Преобразование вершин куба с использованием переданной матрицы трансформации
    cubeVertices = cubeVertices.map { vertex in
        var transformed = SIMD4<Float>(vertex.x, vertex.y, vertex.z, 1)
        transformed = transform * transformed
        return SIMD3<Float>(x: transformed.x, y: transformed.y, z: transformed.z)
    }
    
    var negativeX = false // боковая грань
    var negativeY = false // нижняя грань, оставляем false
    var negativeZ = false // боковая грань
    var positiveX = false // боковая грань
    var positiveY = false // верхняя грань
    var positiveZ = false // боковая грань
    let inwardNormals = false
    
    // Добавление верхней грани
    positiveY = true

    // Проверка и добавление боковых граней
    if x == 0 || intensityMap[y][x-1] < intensityMap[y][x] {
        negativeX = true
    }

    if x == width - 1 || intensityMap[y][x+1] < intensityMap[y][x] {
        positiveX = true
    }
    
    if y == 0 || intensityMap[y-1][x] < intensityMap[y][x] {
        negativeZ = true
    }

    if y == height - 1 || intensityMap[y+1][x] < intensityMap[y][x] {
        positiveZ = true
    }
    
    if negativeX {
            cubeFace(withCubeVertices: cubeVertices, colour: colour,
                     index0: 0, index1: 4, index2: 6, index3: 2,
                     inwardNormals: inwardNormals, triangleMask: triangleMask,
                     vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)
        }

    if positiveX {
        cubeFace(withCubeVertices: cubeVertices, colour: colour,
                 index0: 1, index1: 3, index2: 7, index3: 5,
                 inwardNormals: inwardNormals, triangleMask: triangleMask,
                 vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)
    }

    if negativeY {
        cubeFace(withCubeVertices: cubeVertices, colour: colour,
                 index0: 0, index1: 1, index2: 5, index3: 4,
                 inwardNormals: inwardNormals, triangleMask: triangleMask,
                 vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)
    }

    if positiveY {
        cubeFace(withCubeVertices: cubeVertices, colour: colour,
                 index0: 2, index1: 6, index2: 7, index3: 3,
                 inwardNormals: inwardNormals, triangleMask: triangleMask,
                 vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)
    }

    if negativeZ {
        cubeFace(withCubeVertices: cubeVertices, colour: colour,
                 index0: 0, index1: 2, index2: 3, index3: 1,
                 inwardNormals: inwardNormals, triangleMask: triangleMask,
                 vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)
    }

    if positiveZ {
        cubeFace(withCubeVertices: cubeVertices, colour: colour,
                 index0: 4, index1: 5, index2: 7, index3: 6,
                 inwardNormals: inwardNormals, triangleMask: triangleMask,
                 vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)
    }
}

// Создание вершин и нормалей на основе интенсивности пикселей
func createVoxelMap(from grayscaleImage: NSBitmapImageRep, width: Int, height: Int,
                    vertices: inout [SIMD3<Float>],
                    normals: inout [SIMD3<Float>],
                    colours: inout [SIMD3<Float>],
                    masks: inout [uint]) {
    var intensityMap = Array(repeating: Array(repeating: Float(0), count: width), count: height)

    for y in 0..<height {
            for x in 0..<width {
                let intensity = grayscaleImage.colorAt(x: x, y: y)!.whiteComponent
                intensityMap[y][x] = Float(intensity)
            }
        }

        for y in 0..<height {
            for x in 0..<width {
                let intensity = intensityMap[y][x]
                let voxelSize = Float(0.005)
                let scaleFactor = Float(100) // Количество вокселей на интенсивность
                let z = Float(intensity) * scaleFactor
                let transform = Matrix4x4.scale(
                    voxelSize,
                    voxelSize,
                    voxelSize
                ) * Matrix4x4.translation(
                    (Float(x) - Float(width) / 2),
                    z / 2,
                    (Float(y) - Float(height) / 2)
                ) * Matrix4x4.scale(
                    1,
                    z,
                    1
                )
                let colour = SIMD3<Float>(208/255.0, 199/255.0, 108/255.0)

                addVisibleFaces(x: x, y: y, z: z, width: width, height: height, colour: colour, transform: transform, triangleMask: uint(TRIANGLE_MASK_GEOMETRY), vertices: &vertices, normals: &normals, colours: &colours, masks: &masks, intensityMap: intensityMap)
            }
        }
}
