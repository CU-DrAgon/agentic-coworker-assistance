"""Shared Google Maps Platform HTTP helper (Routes, Places, Weather).

Auth is Application Default Credentials (local login or the Cloud Run service account): a bearer
token plus `X-Goog-User-Project`. If an API rejects OAuth (some Maps endpoints only take keys) and
`SALID_MAPS_API_KEY` is set, the call is retried with that key. The key is never logged.
Every failure becomes `MapsError`; `maps_unavailable` is the code for "not enabled / no
permission / network", so callers can still help (a directions link, a polite message).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import httpx

logger = logging.getLogger(__name__)

TIMEOUT = 8.0
SCOPE = "https://www.googleapis.com/auth/cloud-platform"
PLACES_SEARCH = "https://places.googleapis.com/v1/places:searchText"


class MapsError(Exception):
    def __init__(self, code: str, reason: str = ""):
        super().__init__(f"{code}: {reason}")
        self.code, self.reason = code, reason


def adc_token_provider() -> Callable[[], str]:
    """Bearer tokens from ADC, refreshed when they expire. Imported lazily (tests inject a fake)."""
    state: dict = {}
    lock = threading.Lock()

    def token() -> str:
        import google.auth
        from google.auth.exceptions import GoogleAuthError
        from google.auth.transport.requests import Request

        with lock:
            try:
                if "creds" not in state:
                    state["creds"], _ = google.auth.default(scopes=[SCOPE])
                creds = state["creds"]
                if not creds.valid:
                    creds.refresh(Request())
                return creds.token
            except GoogleAuthError as e:
                raise MapsError("maps_unavailable", "no_credentials") from e

    return token


def _is_auth_rejection(status: int, text: str) -> bool:
    if status == 401:
        return True
    low = text.lower()
    return status == 403 and any(
        k in low for k in ("api keys", "api key", "oauth", "unauthenticated", "credential")
    )


class GoogleHttp:
    def __init__(
        self,
        project: str,
        *,
        token: Callable[[], str] | None = None,
        api_key: str = "",
        http: httpx.Client | None = None,
        timeout: float = TIMEOUT,
    ):
        self.project = project
        self._token = token or adc_token_provider()
        self._key = api_key
        self._http = http or httpx.Client()
        self.timeout = timeout
        self._key_first = False

    def _headers(self, mask: str | None, use_key: bool) -> dict:
        h = {"Content-Type": "application/json"}
        if mask:
            h["X-Goog-FieldMask"] = mask
        if use_key:
            h["X-Goog-Api-Key"] = self._key
        else:
            h["Authorization"] = f"Bearer {self._token()}"
            h["X-Goog-User-Project"] = self.project
        return h

    def _send(self, method, url, json, params, mask, use_key) -> httpx.Response:
        return self._http.request(
            method, url, json=json, params=params, headers=self._headers(mask, use_key),
            timeout=self.timeout,
        )  # fmt: skip

    def request(
        self, method: str, url: str, *, json=None, params=None, mask: str | None = None
    ) -> dict:
        use_key = self._key_first and bool(self._key)
        try:
            r = self._send(method, url, json, params, mask, use_key)
            if not use_key and self._key and _is_auth_rejection(r.status_code, r.text):
                r = self._send(method, url, json, params, mask, True)
                if r.is_success:
                    self._key_first = True
        except MapsError:
            raise
        except httpx.TimeoutException as e:
            raise MapsError("maps_unavailable", "timeout") from e
        except httpx.HTTPError as e:
            raise MapsError("maps_unavailable", "network") from e
        if r.is_success:
            try:
                return r.json() or {}
            except ValueError as e:
                raise MapsError("maps_unavailable", "bad_response") from e
        reason = f"http_{r.status_code}"
        logger.warning("google maps call failed: %s %s", url.split("?")[0], reason)
        if r.status_code in (401, 403, 404, 429) or r.status_code >= 500:
            raise MapsError("maps_unavailable", reason)
        raise MapsError("bad_request", reason)

    def post(self, url: str, body: dict, mask: str | None = None) -> dict:
        return self.request("POST", url, json=body, mask=mask)

    def get(self, url: str, params: dict) -> dict:
        return self.request("GET", url, params=params)


class Geocoder:
    """Place name / address -> (lat, lng) with the Places API (New) text search. Cached."""

    def __init__(self, http: GoogleHttp, clock: Callable[[], float] = time.monotonic):
        self.http, self.clock = http, clock
        self._cache: dict[str, tuple[float, tuple[float, float] | None]] = {}

    def locate(self, text: str, lang: str = "en") -> tuple[float, float] | None:
        key = " ".join(text.casefold().split())
        hit = self._cache.get(key)
        if hit and hit[0] > self.clock():
            return hit[1]
        data = self.http.post(
            PLACES_SEARCH,
            {"textQuery": text, "languageCode": lang, "regionCode": "TH", "maxResultCount": 1},
            "places.location,places.displayName,places.formattedAddress",
        )
        found = None
        for p in data.get("places") or []:
            loc = p.get("location") or {}
            if "latitude" in loc and "longitude" in loc:
                found = (float(loc["latitude"]), float(loc["longitude"]))
                break
        self._cache[key] = (self.clock() + 24 * 3600, found)
        return found


MAPS_PROMPT = (
    "\nเวลาเดินทางและสภาพอากาศ (Google Maps / Google Weather): ใช้ travel_time เมื่อมีคนถาม “ไปยังไง / ใช้เวลากี่นาที / รถติดไหม / ควรออกกี่โมง” "
    "(origin/destination เป็นชื่อสถานที่ที่สลิดจำไว้ เช่น บ้านผม ออฟฟิศ สาขาบางนา หรือที่อยู่ก็ได้; "
    "depart_at เช่น “พรุ่งนี้ 8 โมง”; ถ้าต้องไปให้ทันเวลาให้ใช้ arrive_by แล้วบอกเวลาที่ควรออก leave_by). "
    "ตอบแบบประมาณการสั้นๆ: เวลา (มี traffic_note ถ้ามี) ระยะทาง เส้นทางคร่าวๆ และแนบ maps_link, และบอกว่าเป็นการประมาณ. "
    "ถ้าผลลัพธ์ไม่มี maps_link หรือไม่มี address แปลว่าสถานที่นั้นเป็นส่วนตัวสำหรับบางคนในห้อง: บอกแค่ชื่อสถานที่กับเวลา ห้ามบอกที่อยู่. "
    "ถ้า error เป็น maps_unavailable ให้บอกตรงๆ ว่าตอนนี้ประเมินเวลาไม่ได้ ห้ามเดาตัวเลข (แนบ maps_link ถ้ามี). "
    "ใช้ weather เมื่อถามอากาศ/ฝน/ร่ม (“พรุ่งนี้ฝนตกไหม”, “บ่ายนี้ที่สาขาบางนาฝนจะตกไหม”, “ควรพกร่มไหม”): "
    "place ว่างได้ (ใช้ออฟฟิศ ไม่มีก็กรุงเทพฯ), when เช่น now/today/tomorrow/17:00; ตอบสั้นๆ พร้อมโอกาสฝนและอุณหภูมิ ระบุว่าเป็นพยากรณ์จาก Google Weather."
)
