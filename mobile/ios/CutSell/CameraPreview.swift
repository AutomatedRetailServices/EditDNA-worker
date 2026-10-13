import AVFoundation
import SwiftUI
import UIKit

final class PreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    /// Camera Mirror: the preview is flipped exactly like the file that will be saved.
    var mirrored = false
    /// Changes when the camera flips, so the preview connection is set again.
    var position: AVCaptureDevice.Position = .back

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.backgroundColor = .black
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.previewLayer.session = session
        Self.applyMirror(mirrored, to: uiView.previewLayer)
        // The connection appears once the session is configured: set it again shortly after.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { Self.applyMirror(mirrored, to: uiView.previewLayer) }
    }

    static func applyMirror(_ mirrored: Bool, to layer: AVCaptureVideoPreviewLayer) {
        guard let connection = layer.connection, connection.isVideoMirroringSupported else { return }
        connection.automaticallyAdjustsVideoMirroring = false
        connection.isVideoMirrored = mirrored
    }
}
