import Foundation
// PUTs one part to a real presigned S3 part URL exactly like BackgroundPartUploader (request + uploadTask(fromFile:)).
let url = URL(string: CommandLine.arguments[1])!
let file = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("s3part.bin")
try! Data(repeating: 7, count: 1000).write(to: file)
var req = URLRequest(url: url); req.httpMethod = "PUT"
let done = DispatchSemaphore(value: 0)
URLSession(configuration: .default).uploadTask(with: req, fromFile: file) { data, resp, err in
    let code = (resp as? HTTPURLResponse)?.statusCode ?? -1
    let body = String(data: data ?? Data(), encoding: .utf8) ?? ""
    let code2 = body.range(of: "<Code>[^<]*</Code>", options: .regularExpression).map { String(body[$0]) } ?? ""
    let sts = body.range(of: "<StringToSign>PUT[^<]{0,60}", options: .regularExpression).map { String(body[$0]) } ?? ""
    print("S3 part PUT like the app: HTTP \(code) \(code2) \(sts.replacingOccurrences(of: "\n", with: "|"))")
    done.signal()
}.resume()
done.wait()
