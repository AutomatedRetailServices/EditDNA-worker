import Foundation
// Uploads a small file with URLSession uploadTask(fromFile:) (what the iPhone does for each part)
// to an echo service, and prints the Content-Type header the server actually received.
let file = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("part.bin")
try! Data(repeating: 7, count: 1000).write(to: file)
func probe(_ label: String, _ session: URLSession) {
    var req = URLRequest(url: URL(string: "https://httpbin.org/put")!)
    req.httpMethod = "PUT"
    let done = DispatchSemaphore(value: 0)
    session.uploadTask(with: req, fromFile: file) { data, resp, err in
        if let data, let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let headers = obj["headers"] as? [String: Any] {
            print("\(label): Content-Type sent = \(headers["Content-Type"] ?? "(none)")")
        } else { print("\(label): error \(String(describing: err)) status \((resp as? HTTPURLResponse)?.statusCode ?? -1)") }
        done.signal()
    }.resume()
    done.wait()
}
probe("default session uploadTask(fromFile:)", URLSession(configuration: .default))
var fixed = URLRequest(url: URL(string: "https://httpbin.org/put")!)
fixed.httpMethod = "PUT"; fixed.setValue("", forHTTPHeaderField: "Content-Type")
let d = DispatchSemaphore(value: 0)
URLSession(configuration: .default).uploadTask(with: fixed, fromFile: file) { data, _, _ in
    let h = ((try? JSONSerialization.jsonObject(with: data ?? Data())) as? [String: Any])?["headers"] as? [String: Any]
    print("with Content-Type set to empty string: \(h?["Content-Type"] ?? "(none)")"); d.signal()
}.resume(); d.wait()
