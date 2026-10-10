"""Explicit vehicle lookups against fixed official endpoints; no owner data or scraping."""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Keep identifiers and credentials on the selected official endpoint.


def request_json(url, *, data=None, headers=None):
    request = Request(url, data=data, headers={"Accept": "application/json", **(headers or {})})
    with build_opener(NoRedirect).open(request, timeout=15) as response:
        payload = response.read(1_000_001)
    if len(payload) > 1_000_000:
        raise ValueError("Provider response exceeded the size limit.")
    return json.loads(payload)


def normalize_identifier(country, identifier):
    if not isinstance(identifier, str) or not 1 <= len(identifier) <= 40:
        raise ValueError("Enter a plate or VIN of at most 40 characters.")
    if not re.fullmatch(r"[A-Za-z0-9 -]+", identifier):
        raise ValueError("Use letters, numbers, spaces or hyphens only.")
    value = re.sub(r"[ -]", "", identifier).upper()
    if country == "US":
        if not re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", value):
            raise ValueError("US lookup requires a 17-character VIN, not a plate.")
    elif country == "NL":
        if not re.fullmatch(r"[A-Z0-9]{6}", value):
            raise ValueError("Enter a six-character Dutch plate, excluding separators.")
    elif country == "GB":
        if not re.fullmatch(r"[A-Z0-9]{1,7}", value):
            raise ValueError("Enter a UK registration number of 1–7 letters/digits.")
    else:
        raise ValueError("Supported automated providers are NL, GB and US (VIN only).")
    return value


def lookup(country, identifier):
    identifier = normalize_identifier(country, identifier)
    providers = {"NL": "RDW", "GB": "DVLA VES", "US": "NHTSA vPIC"}
    result = {
        "country": country,
        "provider": providers[country],
        "identifier": identifier,
        "identifier_type": "vin" if country == "US" else "plate",
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "status": "unavailable",
        "details": {},
    }
    try:
        if country == "NL":
            url = "https://opendata.rdw.nl/resource/m9d7-ebf2.json?" + urlencode(
                {"kenteken": identifier, "$limit": 2}
            )
            records = request_json(url)
            if not isinstance(records, list):
                raise ValueError("Unexpected RDW response.")
            if not records:
                return dict(result, status="not_found")
            if len(records) != 1 or records[0].get("kenteken") != identifier:
                raise ValueError("RDW returned an ambiguous or mismatched identifier.")
            fields = {
                "merk": "make",
                "handelsbenaming": "model",
                "voertuigsoort": "vehicle_type",
                "eerste_kleur": "colour",
                "datum_eerste_toelating": "first_registration_yyyymmdd",
                "vervaldatum_apk": "inspection_expiry_yyyymmdd",
                "aantal_zitplaatsen": "seats",
            }
            result["details"] = {
                label: records[0][key] for key, label in fields.items() if key in records[0]
            }
        elif country == "GB":
            key = os.environ.get("DVLA_API_KEY", "").strip()
            if not key:
                return dict(
                    result,
                    status="credentials_required",
                    message="Configure DVLA_API_KEY on the local server. New DVLA VES registrations may be closed.",
                )
            data = request_json(
                "https://driver-vehicle-licensing.api.gov.uk/vehicle-enquiry/v1/vehicles",
                data=json.dumps({"registrationNumber": identifier}).encode(),
                headers={"Content-Type": "application/json", "x-api-key": key},
            )
            if not isinstance(data, dict) or data.get("registrationNumber") != identifier:
                raise ValueError("Unexpected or mismatched DVLA response.")
            result["details"] = {
                k: data[k]
                for k in (
                    "make",
                    "colour",
                    "fuelType",
                    "yearOfManufacture",
                    "engineCapacity",
                    "motStatus",
                    "motExpiryDate",
                    "taxStatus",
                    "taxDueDate",
                )
                if k in data
            }
        else:
            data = request_json(
                f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/{identifier}?format=json"
            )
            records = data.get("Results") if isinstance(data, dict) else None
            if (
                not isinstance(records, list)
                or len(records) != 1
                or not isinstance(records[0], dict)
            ):
                raise ValueError("Unexpected NHTSA response.")
            record = records[0]
            if record.get("VIN", "").upper() != identifier:
                raise ValueError("NHTSA returned a mismatched VIN.")
            result["details"] = {
                k: record[k]
                for k in (
                    "Make",
                    "Model",
                    "ModelYear",
                    "BodyClass",
                    "FuelTypePrimary",
                    "VehicleType",
                    "EngineCylinders",
                )
                if record.get(k)
            }
            codes = str(record.get("ErrorCode", ""))
            if codes != "0":
                return dict(
                    result,
                    status="partial" if result["details"] else "not_found",
                    message="VIN decoding returned validation warnings; details may be incomplete.",
                )
        return dict(result, status="found" if result["details"] else "not_found")
    except HTTPError as exc:
        return dict(
            result,
            status={
                404: "not_found",
                401: "credentials_rejected",
                403: "access_denied",
                429: "rate_limited",
            }.get(exc.code, "unavailable"),
            message=f"Provider returned HTTP {exc.code}.",
        )
    except (URLError, TimeoutError, OSError):
        return dict(result, message="Provider could not be reached. Try again later.")
    except (ValueError, TypeError, KeyError, AttributeError):
        return dict(result, message="Provider returned an unexpected response.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", required=True, choices=["NL", "GB", "US"])
    parser.add_argument("--identifier", required=True)
    args = parser.parse_args(argv)
    try:
        result = lookup(args.country, args.identifier)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "found" else 2


if __name__ == "__main__":
    raise SystemExit(main())
