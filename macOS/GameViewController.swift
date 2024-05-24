import Cocoa
import MetalKit
import UniformTypeIdentifiers

class GameViewController: NSViewController {
    var renderer: Renderer!
    var mtkView: MTKView!
    var counterView: NSTextField!
    var lastMouseLocation: NSPoint = .zero

    override func viewDidLoad() {
        super.viewDidLoad()
        setupMTKView()
        setupCounterView()
        setupRenderer()
        setupOpenImageButton()
    }

    override func viewDidAppear() {
        super.viewDidAppear()
        self.view.window?.makeFirstResponder(self)
    }

    override func becomeFirstResponder() -> Bool {
        return true
    }

    private func setupMTKView() {
        guard let mtkView = self.view as? MTKView else {
            print("View attached to GameViewController is not an MTKView")
            return
        }
        self.mtkView = mtkView

        guard let linearSRGB = CGColorSpace(name: CGColorSpace.linearSRGB) else {
            print("Linear SRGB colour space is not supported on this device")
            return
        }
        mtkView.colorspace = linearSRGB

        guard let defaultDevice = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device")
            return
        }

        mtkView.device = defaultDevice
    }

    private func setupCounterView() {
        counterView = NSTextField(frame: NSRect(x: 10, y: 10, width: 150, height: 50))
        counterView.stringValue = "----"
        counterView.textColor = .white
        counterView.isEditable = false
        counterView.isBezeled = false
        counterView.drawsBackground = false
        mtkView.addSubview(counterView)
    }

    private func setupRenderer() {
        do {
            let newRenderer = try Renderer(withMetalKitView: mtkView) { [unowned self] value in
                self.counterView.stringValue = String(format: "MRays/s: %.3f", value / 1_000_000)
            }
            renderer = newRenderer
            renderer.mtkView(mtkView, drawableSizeWillChange: mtkView.drawableSize)
            mtkView.delegate = renderer
        } catch {
            print("Renderer cannot be initialized: \(error)")
        }
    }

    private func setupOpenImageButton() {
        let openImageButton = NSButton(frame: NSRect(x: 10, y: 70, width: 150, height: 50))
        openImageButton.title = "Open Image"
        openImageButton.target = self
        openImageButton.action = #selector(openImage)
        mtkView.addSubview(openImageButton)
    }

    @objc private func openImage() {
        let openPanel = NSOpenPanel()
        openPanel.canChooseFiles = true
        openPanel.allowedContentTypes = [UTType.png, .jpeg, .tiff]
        openPanel.begin { response in
            if response == .OK, let url = openPanel.url {
                self.renderer.loadImage(from: url.path)
            }
        }
    }

    override func mouseDown(with event: NSEvent) {
        lastMouseLocation = event.locationInWindow
    }

    override func mouseDragged(with event: NSEvent) {
        let currentLocation = event.locationInWindow
        let deltaX = Float(currentLocation.x - lastMouseLocation.x)
        let deltaY = Float(currentLocation.y - lastMouseLocation.y)
        renderer.yaw -= deltaX * 0.001
        renderer.pitch += deltaY * 0.001
        lastMouseLocation = currentLocation
        renderer.mtkView(mtkView, drawableSizeWillChange: mtkView.drawableSize)
    }

    override func mouseUp(with event: NSEvent) {
        let currentLocation = event.locationInWindow
        let deltaX = Float(currentLocation.x - lastMouseLocation.x)
        let deltaY = Float(currentLocation.y - lastMouseLocation.y)
        renderer.yaw -= deltaX * 0.001
        renderer.pitch += deltaY * 0.001
        renderer.mtkView(mtkView, drawableSizeWillChange: mtkView.drawableSize)
    }

    override func keyDown(with event: NSEvent) {
        let moveSpeed: Float = 0.08
        if event.keyCode == 0x0D { // W
            renderer.cameraPosition += renderer.cameraForward * moveSpeed
        }
        if event.keyCode == 0x01 { // S
            renderer.cameraPosition -= renderer.cameraForward * moveSpeed            
        }
        if event.keyCode == 0x00 { // A
            renderer.cameraPosition -= renderer.cameraRight * moveSpeed
        }
        if event.keyCode == 0x02 { // D
            renderer.cameraPosition += renderer.cameraRight * moveSpeed
        }
        if event.keyCode == 0x0F { // D
            renderer.cameraPosition += SIMD3<Float>(0, moveSpeed, 0)
        }
        if event.keyCode == 0x03 { // D
            renderer.cameraPosition -= SIMD3<Float>(0, moveSpeed, 0)
        }

        renderer.mtkView(mtkView, drawableSizeWillChange: mtkView.drawableSize)
    }
}
