import MetalKit
import MetalPerformanceShaders
import simd
import os

// Максимальное количество кадров в процессе рендеринга
let maxFramesInFlight = 3
// Выравненный размер структуры Uniforms
let alignedUniformsSize = (MemoryLayout<Uniforms>.stride + 255) & ~255

// Размер луча в байтах
let rayStride = 48
// Размер структуры пересечения в байтах
let intersectionStride = MemoryLayout<MPSIntersectionDistancePrimitiveIndexCoordinates>.size

// Перечисление возможных ошибок инициализации Renderer
enum RendererInitError: Error {
    case noDevice
    case noLibrary
    case noQueue
    case errorCreatingBuffer
}

// Класс Renderer, отвечающий за рендеринг сцены
class Renderer: NSObject, MTKViewDelegate {

    let view: MTKView
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary

    let accelerationStructure: MPSTriangleAccelerationStructure
    let intersector: MPSRayIntersector

    var vertexPositionBuffer: MTLBuffer
    var vertexNormalBuffer: MTLBuffer
    var vertexColourBuffer: MTLBuffer
    var rayBuffer: MTLBuffer!
    var shadowRayBuffer: MTLBuffer!
    var intersectionBuffer: MTLBuffer!
    var uniformBuffer: MTLBuffer
    var randomBuffer: MTLBuffer
    var triangleMaskBuffer: MTLBuffer

    let rayPipeline: MTLComputePipelineState
    let shadePipeline: MTLComputePipelineState
    let shadowPipeline: MTLComputePipelineState
    let accumulatePipeline: MTLComputePipelineState
    let copyPipeline: MTLRenderPipelineState

    var renderTarget: MTLTexture!
    var accumulationTarget: MTLTexture!

    let semaphore: DispatchSemaphore
    var size: CGSize!
    var randomBufferOffset: Int!
    var uniformBufferOffset: Int!
    var uniformBufferIndex: Int = 0

    var frameIndex: uint = 0

    var lastCheckPoint = Date()
    var timeIntervals: [CFTimeInterval] = []

    let display: (Double) -> Void
    
    var lastTouchLocation: CGPoint = .zero
    var yaw: Float = 0.0
    var pitch: Float = 0.0
    var cameraPosition: SIMD3<Float> = SIMD3<Float>(2, 1, 2)
    var cameraForward: SIMD3<Float> = SIMD3<Float>(0, 0, 1)
    var cameraRight: SIMD3<Float> = SIMD3<Float>(1, 0, 0)
    var cameraUp: SIMD3<Float> = SIMD3<Float>(0, 1, 0)
    
    // Сцена
    var vertices = [SIMD3<Float>]()
    var normals = [SIMD3<Float>]()
    var colours = [SIMD3<Float>]()
    var masks = [uint]()

    // Инициализатор Renderer
    init(withMetalKitView view: MTKView, displayCounter: @escaping (Double) -> Void) throws {
        display = displayCounter
        self.view = view
        // Проверяем наличие устройства
        guard let device = view.device else { throw RendererInitError.noDevice }
        self.device = device
        os_log("Metal device name is %s", device.name)

        semaphore = DispatchSemaphore(value: maxFramesInFlight)

        // Настройка представления Metal
        view.colorPixelFormat = .rgba16Float
        view.sampleCount = 1
        view.drawableSize = view.frame.size
        // Загрузка библиотеки Metal
        guard let library = device.makeDefaultLibrary() else { throw RendererInitError.noLibrary }
        self.library = library
        // Создание очереди команд
        guard let queue = device.makeCommandQueue() else { throw RendererInitError.noQueue }
        self.queue = queue

        // Создание вычислительных конвейеров
        let computeDescriptor = MTLComputePipelineDescriptor()
        computeDescriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
        
        // Создание конвейера для генерации лучей
        computeDescriptor.computeFunction = library.makeFunction(name: "rayKernel")
        self.rayPipeline = try device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)
        
        // Создание конвейера для затенения
        computeDescriptor.computeFunction = library.makeFunction(name: "shadeKernel")
        self.shadePipeline = try device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)
        
        // Создание конвейера для теневых лучей
        computeDescriptor.computeFunction = library.makeFunction(name: "shadowKernel")
        self.shadowPipeline = try device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)
        
        // Создание конвейера для накопления изображения
        computeDescriptor.computeFunction = library.makeFunction(name: "accumulateKernel")
        self.accumulatePipeline = try device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)
        
        // Создание графического конвейера для копирования изображения
        let renderDescriptor = MTLRenderPipelineDescriptor()
        renderDescriptor.sampleCount = view.sampleCount
        renderDescriptor.vertexFunction = library.makeFunction(name: "copyVertex")
        renderDescriptor.fragmentFunction = library.makeFunction(name: "copyFragment")
        renderDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
        self.copyPipeline = try device.makeRenderPipelineState(descriptor: renderDescriptor)

        let transform = Matrix4x4.translation(0, 0.5, 0) * Matrix4x4.scale(1.5, 1, 1.5)
        let colour = SIMD3<Float>(208/255.0, 199/255.0, 108/255.0)
        cube(withFaceMask: [.negativeY], colour: colour, transform: transform, inwardNormals: true, triangleMask: uint(TRIANGLE_MASK_GEOMETRY), vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)
        
        // Создание буферов
        // Буфер униформ содержит несколько небольших значений, которые изменяются от кадра к кадру. У нас может быть до 3
        // кадров в процессе одновременно, поэтому выделяем область буфера для каждого кадра. GPU будет читать из
        // одного куска, пока CPU записывает в следующий кусок. Каждый кусок должен быть выровнен по 256 байтам на macOS
        // и 16 байтам на iOS.
        let uniformBufferSize = alignedUniformsSize * maxFramesInFlight

        // Данные вершин должны храниться в приватных или управляемых буферах на системах с дискретными GPU (AMD, NVIDIA).
        // Приватные буферы хранятся полностью в памяти GPU и недоступны для CPU. Управляемые
        // буферы поддерживают копию в памяти CPU и копию в памяти GPU.
        let storageOptions: MTLResourceOptions

        #if arch(x86_64)
        storageOptions = .storageModeManaged
        #else // iOS, tvOS
        storageOptions = .storageModeShared
        #endif

        // Выделяем буферы для позиций вершин, цветов и нормалей. Обратите внимание, что каждая позиция вершины - это
        // float3, который выровнен по 16 байтам.
        guard let uniformBuffer = device.makeBuffer(length: uniformBufferSize, options: storageOptions) else {
            throw RendererInitError.errorCreatingBuffer
        }
        self.uniformBuffer = uniformBuffer

        let float2Size = MemoryLayout<SIMD2<Float>>.stride
        guard let randomBuffer = device.makeBuffer(length: 256 * maxFramesInFlight * float2Size, options: storageOptions) else {
            throw RendererInitError.errorCreatingBuffer
        }
        self.randomBuffer = randomBuffer

        let float3Size = MemoryLayout<SIMD3<Float>>.stride
        guard let vertexPositionBuffer = device.makeBuffer(bytes: &vertices, length: vertices.count * float3Size, options: storageOptions) else {
            throw RendererInitError.errorCreatingBuffer
        }
        self.vertexPositionBuffer = vertexPositionBuffer

        guard let vertexColourBuffer = device.makeBuffer(bytes: &colours, length: colours.count * float3Size, options: storageOptions) else {
            throw RendererInitError.errorCreatingBuffer
        }
        self.vertexColourBuffer = vertexColourBuffer

        guard let vertexNormalBuffer = device.makeBuffer(bytes: &normals, length: normals.count * float3Size, options: storageOptions) else {
            throw RendererInitError.errorCreatingBuffer
        }
        self.vertexNormalBuffer = vertexNormalBuffer

        let uintSize = MemoryLayout<uint>.stride
        guard let triangleMaskBuffer = device.makeBuffer(bytes: &masks, length: masks.count * uintSize, options: storageOptions) else {
            throw RendererInitError.errorCreatingBuffer
        }
        self.triangleMaskBuffer = triangleMaskBuffer

        // Для управляемых буферов необходимо указать, что мы изменили буфер, чтобы копия на GPU могла быть обновлена
        #if arch(x86_64)
        if storageOptions.contains(.storageModeManaged) {
            vertexPositionBuffer.didModifyRange(0..<vertexPositionBuffer.length)
            vertexColourBuffer.didModifyRange(0..<vertexColourBuffer.length)
            vertexNormalBuffer.didModifyRange(0..<vertexNormalBuffer.length)
            triangleMaskBuffer.didModifyRange(0..<triangleMaskBuffer.length)
        }
        #endif

        // Создание трассировщика лучей для устройства Metal
        intersector = MPSRayIntersector(device: device)
        intersector.rayDataType = .originMaskDirectionMaxDistance
        intersector.rayStride = rayStride
        intersector.rayMaskOptions = .primitive

        // Создание структуры ускорения из данных позиций вершин
        accelerationStructure = MPSTriangleAccelerationStructure(device: device)
        accelerationStructure.vertexBuffer = vertexPositionBuffer
        accelerationStructure.maskBuffer = triangleMaskBuffer
        accelerationStructure.triangleCount = vertices.count / 3

        accelerationStructure.rebuild()
    }
    
    func loadImage(from imagePath: String) {
        var vertices = [SIMD3<Float>]()
        var normals = [SIMD3<Float>]()
        var colours = [SIMD3<Float>]()
        var masks = [uint]()
        
        // Источник света
        var transform = matrix_identity_float4x4
        
        // Загрузка изображения и извлечение интенсивности пикселей
        guard let image = NSImage(contentsOfFile: imagePath),
              let tiffData = image.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiffData) else {
            print("Failed to load image")
            return
        }
        
        // Преобразование изображения в градации серого
        guard let grayscaleImage = convertToGrayscale(image: bitmap) else {
            print("Failed to convert image to grayscale")
            return
        }
        
        let width = grayscaleImage.pixelsWide
        let height = grayscaleImage.pixelsHigh
        
        // Создание вершин и нормалей на основе интенсивности пикселей
        createVoxelMap(from: grayscaleImage, width: width, height: height, vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)
        
        transform = Matrix4x4.translation(0, 0.5, 0) * Matrix4x4.scale(Float(width) * 1.5, 1, Float(height) * 1.5)
        let colour = SIMD3<Float>(208/255.0, 199/255.0, 108/255.0)
        cube(withFaceMask: [.negativeY], colour: colour, transform: transform, inwardNormals: true, triangleMask: uint(TRIANGLE_MASK_GEOMETRY), vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)
        
        // Создание новых буферов для данных вершин
        let float3Size = MemoryLayout<SIMD3<Float>>.stride
        let uintSize = MemoryLayout<uint>.stride
        
        let storageOptions: MTLResourceOptions
        #if arch(x86_64)
        storageOptions = .storageModeManaged
        #else
        storageOptions = .storageModeShared
        #endif
         
        vertexPositionBuffer = device.makeBuffer(bytes: vertices, length: vertices.count * float3Size, options: storageOptions)!
        vertexNormalBuffer = device.makeBuffer(bytes: normals, length: normals.count * float3Size, options: storageOptions)!
        vertexColourBuffer = device.makeBuffer(bytes: colours, length: colours.count * float3Size, options: storageOptions)!
        triangleMaskBuffer = device.makeBuffer(bytes: masks, length: masks.count * uintSize, options: storageOptions)!
        
        #if arch(x86_64)
        if storageOptions.contains(.storageModeManaged) {
            vertexPositionBuffer.didModifyRange(0..<vertexPositionBuffer.length)
            vertexNormalBuffer.didModifyRange(0..<vertexNormalBuffer.length)
            vertexColourBuffer.didModifyRange(0..<vertexColourBuffer.length)
            triangleMaskBuffer.didModifyRange(0..<triangleMaskBuffer.length)
        }
        #endif
        
        // Перезагрузка структуры ускорения
        accelerationStructure.vertexBuffer = vertexPositionBuffer
        accelerationStructure.maskBuffer = triangleMaskBuffer
        accelerationStructure.triangleCount = vertices.count / 3
        accelerationStructure.rebuild()
    }

    // Функция для преобразования изображения в градации серого
    func convertToGrayscale(image: NSBitmapImageRep) -> NSBitmapImageRep? {
        let width = image.pixelsWide
        let height = image.pixelsHigh
        let bitsPerSample = 8
        let samplesPerPixel = 1
        let bytesPerRow = width * samplesPerPixel
        let colorSpace = CGColorSpaceCreateDeviceGray()
        
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: bitsPerSample, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: CGImageAlphaInfo.none.rawValue),
              let cgImage = image.cgImage else {
            return nil
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let grayscaleImage = context.makeImage() else {
            return nil
        }
        
        return NSBitmapImageRep(cgImage: grayscaleImage)
    }

    // Метод, вызываемый при изменении размера представления
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        self.size = size

        lastCheckPoint = Date()
        timeIntervals.removeAll()

        // Обработка изменения размера окна путем выделения буфера, достаточного для одного стандартного луча,
        // одного теневого луча и одного результата пересечения луч/треугольник на каждый пиксель
        let rayCount = Int(size.width * size.height)
        // Мы используем приватные буферы здесь, потому что лучи и результаты пересечения будут полностью производиться
        // и потребляться на GPU
        rayBuffer = device.makeBuffer(length: rayStride * rayCount, options: .storageModePrivate)
        shadowRayBuffer = device.makeBuffer(length: rayStride * rayCount, options: .storageModePrivate)
        intersectionBuffer = device.makeBuffer(length: intersectionStride * rayCount, options: .storageModePrivate)

        // Создание цели рендеринга, в которую ядро затенения будет записывать
        let renderTargetDescriptor = MTLTextureDescriptor()
        renderTargetDescriptor.pixelFormat = .rgba32Float
        renderTargetDescriptor.textureType = .type2D
        renderTargetDescriptor.width = Int(size.width)
        renderTargetDescriptor.height = Int(size.height)
        // Хранение в приватной памяти, так как текстура будет только читаться и записываться с GPU
        renderTargetDescriptor.storageMode = .private
        // Указываем, что мы будем читать и записывать текстуру с GPU
        renderTargetDescriptor.usage = [.shaderRead, .shaderWrite]

        renderTarget = device.makeTexture(descriptor: renderTargetDescriptor)
        accumulationTarget = device.makeTexture(descriptor: renderTargetDescriptor)
        frameIndex = 0
    }

    // Метод рендеринга кадра
    func draw(in view: MTKView) {
        // Мы используем буфер униформ для потоковой передачи данных униформ на GPU, поэтому нам нужно подождать, пока
        // самый старый кадр GPU не завершится, прежде чем мы сможем повторно использовать это пространство в буфере.
        semaphore.wait()

        // Отчет о производительности рендеринга
        let now = Date()
        let timePassed = now.timeIntervalSince(lastCheckPoint)
        if timePassed > 1 {
            let totalPixels = Int(size.width * size.height) * timeIntervals.count
            let totalTime = timeIntervals.reduce(0, +)
            DispatchQueue.main.async { [unowned self] in
                self.display(Double(totalPixels) / totalTime)
            }
            timeIntervals.removeAll()
            lastCheckPoint = now
        }

        // Создаем командный буфер, который будет содержать наши команды для GPU
        guard let commandBuffer = queue.makeCommandBuffer() else { return }
        // Когда кадр завершен, сигнализируем, что мы можем повторно использовать пространство буфера униформ из этого кадра.
        // Обратите внимание, что содержимое обработчиков завершения должно быть как можно быстрее, так как драйвер GPU может
        // иметь другую работу, запланированную в очереди.
        commandBuffer.addCompletedHandler { [unowned self] cb in
            let executionDuration = cb.gpuEndTime - cb.gpuStartTime
            self.timeIntervals.append(executionDuration)
            self.semaphore.signal()
        }

        updateUniforms()

        let width = Int(size.width)
        let height = Int(size.height)
        // Мы запускаем прямоугольную сетку потоков на GPU для генерации лучей. Потоки запускаются в группах потоков,
        // называемых "threadgroups". Нам нужно выровнять количество потоков, чтобы оно было кратно размеру группы потоков.
        // Мы указали при компиляции конвейера, что размер группы потоков будет кратен ширине выполнения потока (размеру группы SIMD),
        // который обычно составляет 32 или 64, поэтому размер группы потоков 8x8 безопасен и должен поддерживаться на большинстве устройств.
        let w = rayPipeline.threadExecutionWidth
        let h = rayPipeline.maxTotalThreadsPerThreadgroup / w
        let threadsPerThreadgroup = MTLSizeMake(w, h, 1)

        // Сначала мы будем генерировать лучи на GPU. Создаем кодировщик вычислений, который будет использоваться
        // для добавления команд в командный буфер.
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        // Привязываем буферы, необходимые для вычислительного конвейера
        computeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: 0)
        computeEncoder.setBuffer(rayBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(randomBuffer, offset: randomBufferOffset, index: 2)

        computeEncoder.setTexture(renderTarget, index: 0)
        // Привязываем вычислительный конвейер для генерации лучей
        computeEncoder.setComputePipelineState(rayPipeline)
        // Запускаем потоки
        let threadsPerGrid = MTLSizeMake(width, height, 1)
        computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        // Завершаем кодировщик
        computeEncoder.endEncoding()

        // Мы будем итеративно запускать следующие ядра несколько раз, чтобы позволить свету распространяться по сцене
        for _ in 0..<3 {
            intersector.intersectionDataType = .distancePrimitiveIndexCoordinates
            // Затем мы передаем лучи в MPSRayIntersector для вычисления пересечений с нашей структурой ускорения
            intersector.encodeIntersection(commandBuffer: commandBuffer,
                                           intersectionType: .nearest,
                                           rayBuffer: rayBuffer,
                                           rayBufferOffset: 0,
                                           intersectionBuffer: intersectionBuffer,
                                           intersectionBufferOffset: 0,
                                           rayCount: width * height,
                                           accelerationStructure: accelerationStructure)
            // Запускаем другой конвейер для обработки результатов пересечения и затенения сцены
            guard let shadeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }

            let buffers = [uniformBuffer, rayBuffer, shadowRayBuffer, intersectionBuffer,
                           vertexColourBuffer, vertexNormalBuffer, randomBuffer, triangleMaskBuffer]
            let offsets: [Int] = [uniformBufferOffset, 0, 0, 0, 0, 0, randomBufferOffset, 0]
            shadeEncoder.setBuffers(buffers, offsets: offsets, range: 0..<8)

            shadeEncoder.setTexture(renderTarget, index: 0)
            shadeEncoder.setComputePipelineState(shadePipeline)
            shadeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            shadeEncoder.endEncoding()

            // Мы пересекаем лучи со сценой, за исключением того, что в этот раз мы пересекаем теневые лучи. Нам нужно
            // только знать, пересекается ли теневой луч с чем-либо на пути к источнику света, а не какой треугольник был пересечен.
            // Поэтому мы можем использовать тип пересечения "any", чтобы завершить поиск пересечений, как только будет найдено любое пересечение.
            // Это обычно намного быстрее, чем поиск ближайшего пересечения. Мы также можем использовать MPSIntersectionDataTypeDistance,
            // потому что нам не нужен индекс треугольника и барицентрические координаты.
            intersector.intersectionDataType = .distance
            intersector.encodeIntersection(commandBuffer: commandBuffer,
                                           intersectionType: .any,
                                           rayBuffer: shadowRayBuffer,
                                           rayBufferOffset: 0,
                                           intersectionBuffer: intersectionBuffer,
                                           intersectionBufferOffset: 0,
                                           rayCount: width * height,
                                           accelerationStructure: accelerationStructure)
            // Наконец, мы запускаем ядро, которое записывает цвет, вычисленный ядром затенения, в выходное изображение,
            // но только если соответствующий теневой луч не пересекается с чем-либо на пути к источнику света.
            // Если теневой луч пересекает треугольник перед достижением источника света, исходная точка пересечения была в тени.
            guard let colourEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
            colourEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: 0)
            colourEncoder.setBuffer(shadowRayBuffer, offset: 0, index: 1)
            colourEncoder.setBuffer(intersectionBuffer, offset: 0, index: 2)

            colourEncoder.setTexture(renderTarget, index: 0)
            colourEncoder.setComputePipelineState(shadowPipeline)
            colourEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            colourEncoder.endEncoding()
        }
        // Финальное ядро усредняет изображение текущего кадра со всеми предыдущими кадрами для уменьшения шума из-за случайной выборки сцены.
        guard let denoiseEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        denoiseEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: 0)

        denoiseEncoder.setTexture(renderTarget, index: 0)
        denoiseEncoder.setTexture(accumulationTarget, index: 1)

        denoiseEncoder.setComputePipelineState(accumulatePipeline)
        denoiseEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        denoiseEncoder.endEncoding()

        // Копируем полученное изображение в наше представление с помощью графического конвейера, так как мы не можем записывать непосредственно
        // в него с помощью вычислительного ядра. Нам нужно задержать получение текущего дескриптора прохода рендеринга как можно дольше, чтобы избежать
        // остановки до тех пор, пока GPU/композитор не освободит drawable. Дескриптор прохода рендеринга может быть nil, если окно было перемещено с экрана.
        if let renderPassDescriptor = view.currentRenderPassDescriptor {
            guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else { return }
            renderEncoder.setRenderPipelineState(copyPipeline)
            renderEncoder.setFragmentTexture(accumulationTarget, index: 0)
            // Рисуем прямоугольник, заполняющий экран
            renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
            renderEncoder.endEncoding()
            // Отображаем drawable на экране
            guard let drawable = view.currentDrawable else { return }
            commandBuffer.present(drawable)
        }
        // Наконец, коммитим командный буфер, чтобы GPU мог начать выполнение
        commandBuffer.commit()
    }

    func updateUniforms() {
        uniformBufferOffset = alignedUniformsSize * uniformBufferIndex
        let uniformsPointer = uniformBuffer.contents().advanced(by: uniformBufferOffset)
        let uniforms = uniformsPointer.bindMemory(to: Uniforms.self, capacity: 1)

        // Создание матриц вращения по yaw и pitch
        let yawMatrix = Matrix4x4.rotation(radians: yaw, axis: SIMD3<Float>(0, 1, 0))
        let pitchMatrix = Matrix4x4.rotation(radians: pitch, axis: SIMD3<Float>(1, 0, 0))
        let rotationMatrix = yawMatrix * pitchMatrix

        // Применение матрицы вращения к направлению камеры
        let forward = SIMD4<Float>(0, 0, -1, 0)
        let right = SIMD4<Float>(1, 0, 0, 0)
        let up = SIMD4<Float>(0, 1, 0, 0)

        let rotatedForward = rotationMatrix * forward
        let rotatedRight = rotationMatrix * right
        let rotatedUp = rotationMatrix * up

        cameraForward = SIMD3<Float>(rotatedForward.x, rotatedForward.y, rotatedForward.z)
        cameraRight = SIMD3<Float>(rotatedRight.x, rotatedRight.y, rotatedRight.z)
        cameraUp = SIMD3<Float>(rotatedUp.x, rotatedUp.y, rotatedUp.z)

        uniforms.pointee.camera.position = cameraPosition
        uniforms.pointee.camera.forward = cameraForward
        uniforms.pointee.camera.right = cameraRight
        uniforms.pointee.camera.up = cameraUp

        uniforms.pointee.light.position = SIMD3<Float>(10, 10, 10)
        uniforms.pointee.light.forward = SIMD3<Float>(0, -1, 0)
        uniforms.pointee.light.right = SIMD3<Float>(0.25, 0, 0)
        uniforms.pointee.light.up = SIMD3<Float>(0, 0, 0.25)
        uniforms.pointee.light.color = SIMD3<Float>(1000, 1000, 1000)

        let fieldOfView = 45.0 * (Float.pi / 180.0)
        let aspectRatio = Float(size.width) / Float(size.height)
        let imagePlaneHeight = tanf(fieldOfView / 2.0)
        let imagePlaneWidth = aspectRatio * imagePlaneHeight

        uniforms.pointee.camera.right *= imagePlaneWidth
        uniforms.pointee.camera.up *= imagePlaneHeight

        uniforms.pointee.width = UInt32(size.width)
        uniforms.pointee.height = UInt32(size.height)
        uniforms.pointee.blocksWide = (uniforms.pointee.width + 15) / 16
        uniforms.pointee.frameIndex = frameIndex
        frameIndex += 1

        // Для управляемого режима хранения
        #if arch(x86_64)
        uniformBuffer.didModifyRange(uniformBufferOffset..<uniformBufferOffset + alignedUniformsSize)
        #endif

        randomBufferOffset = 256 * MemoryLayout<SIMD2<Float>>.stride * uniformBufferIndex
        let float2Pointer = randomBuffer.contents().advanced(by: randomBufferOffset)
        var randoms = float2Pointer.bindMemory(to: SIMD2<Float>.self, capacity: 1)
        for _ in 0..<256 {
            randoms.pointee = SIMD2<Float>(Float.random(in: 0..<1), Float.random(in: 0..<1))
            randoms = randoms.advanced(by: 1)
        }

        // Для управляемого режима хранения
        #if arch(x86_64)
        randomBuffer.didModifyRange(randomBufferOffset..<randomBufferOffset + 256 * MemoryLayout<SIMD2<Float>>.stride)
        #endif

        uniformBufferIndex = (uniformBufferIndex + 1) % maxFramesInFlight
    }
}
