from fastapi.testclient import TestClient

from meeting_scribe.app import app


def test_privacy_and_terms_are_public_and_filled():
    c = TestClient(app)
    for path, needle in (("/privacy", "Limited Use"), ("/terms", "Terms of Service")):
        r = c.get(path)
        assert r.status_code == 200
        assert needle in r.text
        assert "__CONTACT_EMAIL__" not in r.text
