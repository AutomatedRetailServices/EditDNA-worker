import CoreTransferable
import Foundation
import UniformTypeIdentifiers

struct ImportedVideoFile: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { item in
            SentTransferredFile(item.url)
        } importing: { received in
            let suffix = received.file.pathExtension.isEmpty ? "mov" : received.file.pathExtension
            let destination = try persistentImportURL(extension: suffix)
            try FileManager.default.copyItem(at: received.file, to: destination)
            return ImportedVideoFile(url: destination)
        }
    }
}

struct ImportedImageFile: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .image) { item in
            SentTransferredFile(item.url)
        } importing: { received in
            let suffix = received.file.pathExtension.isEmpty ? "jpg" : received.file.pathExtension
            let destination = try persistentImportURL(extension: suffix)
            try FileManager.default.copyItem(at: received.file, to: destination)
            return ImportedImageFile(url: destination)
        }
    }
}

private func persistentImportURL(extension suffix: String) throws -> URL {
    let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("CutSell", isDirectory: true)
        .appendingPathComponent("Imports", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root.appendingPathComponent("media-\(UUID().uuidString).\(suffix.lowercased())")
}
