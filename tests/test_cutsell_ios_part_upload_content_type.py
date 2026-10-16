"""TestFlight bug (builds 3-4): no video ever uploaded from an iPhone, so no cut and no editor.

Cause (proven on a macOS runner against the live server): URLSession adds
"Content-Type: application/octet-stream" to a file upload; the server's part links are signed
with an EMPTY Content-Type (legacy S3 signature, where Content-Type is part of what is signed),
so S3 answered 403 SignatureDoesNotMatch. The app now sends an empty Content-Type (= none)."""
from pathlib import Path

from botocore.session import get_session

APP = Path(__file__).resolve().parents[1] / "mobile" / "ios" / "CutSell"


def test_part_upload_sends_no_content_type():
    uploader = (APP / "BackgroundPartUploader.swift").read_text()
    assert 'static let partContentType = ""' in uploader
    assert 'request.setValue(Self.partContentType, forHTTPHeaderField: "Content-Type")' in uploader
    # Set before any extra headers, and the multipart manager passes none.
    body = uploader.split("func upload(fileURL: URL, to signedURL: URL", 1)[1]
    assert body.index("Self.partContentType") < body.index("for (key, value) in headers")
    manager = (APP / "MultipartUploadManager.swift").read_text()
    assert "backgroundUploader.upload(fileURL: partFile, to: signedURL, onBytes: onBytes)" in manager


def test_server_part_links_sign_the_content_type():
    # The same presign call the server makes (multipart_uploads.presign_part), offline with fake keys:
    # the default us-east-1 client produces a legacy-signature link, whose signature covers Content-Type.
    client = get_session().create_client(
        "s3", region_name="us-east-1", aws_access_key_id="AKIDEXAMPLE", aws_secret_access_key="secret"
    )
    url = client.generate_presigned_url(
        "upload_part",
        Params={"Bucket": "example-bucket", "Key": "k.mp4", "UploadId": "u", "PartNumber": 1},
        ExpiresIn=600,
    )
    assert "Signature=" in url and "AWSAccessKeyId=" in url and "X-Amz-SignedHeaders" not in url


def test_project_without_finished_upload_offers_resume_not_editor():
    detail = (APP / "ProjectsView.swift").read_text()
    route = detail.split("struct ProjectDetailView", 1)[1]
    assert route.index("Project.uploadNotFinishedState") < route.index("DraftEditorView(project: project)")
    assert "UploadNotFinishedView(project: project)" in route
    assert 'static let uploadNotFinishedState = "created"' in (APP / "Models.swift").read_text()
    assert "NewCutView(pending: record)" in detail
