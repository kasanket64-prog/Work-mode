import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from camera_system import vehicles
from camera_system.review import make_server


def test_providers_validation_and_credential_boundaries(monkeypatch):
    calls = []

    def fetch(url, **kwargs):
        calls.append((url, kwargs))
        if "rdw.nl" in url:
            return [{"kenteken": "AB123C", "merk": "TEST", "owner": "must not return"}]
        if "licensing.api" in url:
            return {"registrationNumber": "AB12CDE", "make": "TEST", "owner": "must not return"}
        return {"Results": [{"VIN": "1HGCM82633A004352", "Make": "HONDA", "ErrorCode": "0"}]}

    monkeypatch.setattr(vehicles, "request_json", fetch)
    assert vehicles.lookup("NL", "ab-123-c")["details"] == {"make": "TEST"}
    monkeypatch.delenv("DVLA_API_KEY", raising=False)
    assert vehicles.lookup("GB", "AB12CDE")["status"] == "credentials_required"
    assert len(calls) == 1
    monkeypatch.setenv("DVLA_API_KEY", "test-key")
    result = vehicles.lookup("GB", "AB12 CDE")
    assert result["details"] == {"make": "TEST"}
    assert calls[-1][1]["headers"]["x-api-key"] == "test-key"
    assert "test-key" not in json.dumps(result)
    assert vehicles.lookup("US", "1HGCM82633A004352")["details"]["Make"] == "HONDA"
    for country, value in [("US", "AB1234"), ("NL", "AB'123"), ("XX", "AB123C")]:
        with pytest.raises(ValueError):
            vehicles.lookup(country, value)


def test_missing_partial_mismatched_and_provider_failure(monkeypatch):
    monkeypatch.setattr(vehicles, "request_json", lambda *a, **kw: [])
    assert vehicles.lookup("NL", "AB123C")["status"] == "not_found"
    monkeypatch.setattr(vehicles, "request_json", lambda *a, **kw: [{"kenteken": "ZZ999Z"}])
    assert vehicles.lookup("NL", "AB123C")["status"] == "unavailable"
    monkeypatch.setattr(
        vehicles,
        "request_json",
        lambda *a, **kw: {
            "Results": [{"VIN": "1HGCM82633A004352", "Make": "HONDA", "ErrorCode": "5"}]
        },
    )
    assert vehicles.lookup("US", "1HGCM82633A004352")["status"] == "partial"

    def limited(*a, **kw):
        raise HTTPError("https://example.invalid", 429, "secret text", {}, None)

    monkeypatch.setattr(vehicles, "request_json", limited)
    assert vehicles.lookup("NL", "AB123C")["status"] == "rate_limited"


def test_lookup_route_blocks_foreign_origins_and_bad_input(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "camera_system.review.lookup",
        lambda country, identifier: calls.append((country, identifier)) or {"status": "found"},
    )
    with make_server([], 0) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        origin = f"http://127.0.0.1:{server.server_port}"
        try:

            def request(body, source):
                return Request(
                    origin + "/api/vehicle-lookup",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json", "Origin": source},
                )

            with pytest.raises(HTTPError) as error:
                urlopen(
                    request({"country": "NL", "identifier": "AB123C"}, "https://untrusted.invalid")
                )
            assert error.value.code == 403 and not calls
            with pytest.raises(HTTPError) as error:
                urlopen(
                    request(
                        {
                            "country": "NL",
                            "identifier": "AB123C",
                            "url": "https://untrusted.invalid",
                        },
                        origin,
                    )
                )
            assert error.value.code == 400 and not calls
            with urlopen(request({"country": "NL", "identifier": "AB123C"}, origin)) as response:
                assert json.load(response)["status"] == "found"
            assert len(calls) == 1
        finally:
            server.shutdown()
            thread.join(timeout=5)
