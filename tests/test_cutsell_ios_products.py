"""Edits · Products (Figma 1135:2686 / 2876 / 3059): filters, product view, assign sheet, album hint."""
from pathlib import Path

IOS = Path(__file__).resolve().parents[1] / "mobile" / "ios" / "CutSell"


def read(name):
    return (IOS / name).read_text()


def test_edits_filters_in_figma_order_and_sliding():
    edits = read("ProjectsView.swift")
    assert 'case processing = "Processing", ready = "Ready", products = "Products", drafts = "Drafts", all = "All"' in edits
    assert "ScrollView(.horizontal, showsIndicators: false)" in edits and "ProductsList(store: catalog" in edits
    assert '"Your cuts, organized by brand and product."' in edits


def test_products_screens_use_the_catalog_api():
    view = read("ProductsView.swift")
    for path in ('"/v1/catalog"', '"/v1/catalog/products/\\(productID)"', '"/v1/catalog/assign"', '"/v1/catalog/rename"'):
        assert path in view
    for text in ("Which product is this?", "✦ SUGGESTED FROM THE VIDEO", "+  New brand or product", "We couldn’t tell from the video",
                 "✦  Grouped automatically from what you said in each video.", "✎ Rename", "in bulk"):
        assert text in view


def test_album_name_is_sent_as_a_free_hint():
    assert "static func albumName(of asset: PHAsset)" in read("GalleryPickerView.swift")
    submission = read("CutSubmission.swift")
    assert 'out["album_hint"] = album' in submission and "Self.metadata(bulkTag, album: video.albumHint)" in submission
