# Official vehicle lookups

The review includes a Vehicle details form. Users choose the service, enter a plate/VIN, confirm the value, then explicitly submit it. No video, frame, location, or track ID is sent. This feature currently uses manually entered identifiers: automatic plate detection/OCR and multi-frame plate confirmation are not implemented.

| Country | Integration | Access |
| --- | --- | --- |
| Netherlands | RDW registration lookup | Key-free; live verified |
| United Kingdom | DVLA VES registration lookup | Requires `DVLA_API_KEY`; mocked response tests only, no key available |
| United States | NHTSA vPIC VIN decoder | Key-free; live verified; cannot look up a US plate |
| India | Official Parivahan link | Manual service only; no automated connector |
| Norway | Official API information/access link | Manual access setup only; no automated connector |

No owner names or addresses are returned. Identifiers are normalized without guessing ambiguous OCR characters. The selected service must cover the vehicle's registration country. VIN decoding describes manufacturer specifications; it does not verify ownership or registration status.

## Run

```powershell
.\.venv\Scripts\python.exe -m camera_system.review outputs/office-40-scheduled-20261006.jsonl --output outputs/vehicle-review.html --serve --port 8768
```

Open `http://127.0.0.1:8768/`. A saved HTML file opened directly can still review detections, but online lookup requires the local server. Existing generated HTML files must be regenerated to include the new form. Result paths must be new. Lookup results appear separately from the video session and are not included in its JSON export or automatically associated with a tracked car.

CLI:

```powershell
.\.venv\Scripts\python.exe -m camera_system.vehicles --country NL --identifier H323DK
.\.venv\Scripts\python.exe -m camera_system.vehicles --country US --identifier 1HGCM82633A004352
```

After reinstalling the editable package, `camera-vehicle` offers the same CLI. For UK use, configure `DVLA_API_KEY` in the server process environment through your usual secret management; never commit it or paste it into browser fields. DVLA's current guide says new VES registrations are closed. Do not assume a key can currently be obtained.

Only fixed HTTPS endpoints are called. Responses have a 15-second timeout and a 1 MB limit; redirects are rejected so credentials cannot follow a redirect to another host. Provider errors return explicit statuses without raw response bodies or credentials. Field allowlists limit returned data. The local POST endpoint accepts JSON only with the matching localhost Origin and Host. No automatic retries, bulk lookup, history, or disk cache are used. The existing standard-library server handles requests sequentially, so a lookup can delay another request until its timeout.

## Verified 7 October 2026

Six connector/review tests passed for normal responses, normalization, missing credentials, identifier mismatch, partial VIN decoding, rate limiting and local request boundaries. Live RDW returned AUDI / E-TRON 50 for a registration from its public dataset; live NHTSA returned HONDA / Accord / 2003 for a public example VIN. These calls verify provider connectivity, not video OCR accuracy. UK was not live-tested.

## Official references

- [RDW open-data information](https://www.rdw.nl/over-rdw/dienstverlening/open-data/algemene-informatie)
- [RDW vehicle dataset](https://opendata.rdw.nl/resource/m9d7-ebf2.json)
- [DVLA VES guide and access status](https://developer-portal.driver-vehicle-licensing.api.gov.uk/apis/vehicle-enquiry-service/vehicle-enquiry-service-description.html)
- [DVLA response specification](https://developer-portal.driver-vehicle-licensing.api.gov.uk/apis/vehicle-enquiry-service/v1.2.0-vehicle-enquiry-service.html)
- [NHTSA vPIC API](https://vpic.nhtsa.dot.gov/api/)
- [India official lookup instructions](https://odishatransport.gov.in/content/Know-your-vehicle-details/1)
- [Norway technical vehicle API access](https://www.vegvesen.no/fag/teknologi/apne-data/et-utvalg-apne-data/api-for-tekniske-kjoretoyopplysninger/)
