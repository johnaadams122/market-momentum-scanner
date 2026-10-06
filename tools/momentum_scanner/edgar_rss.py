import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import requests

from momentum_scanner import config
from momentum_scanner.errors import ScannerNetworkError

ATOM_NS = "http://www.w3.org/2005/Atom"
# count=100: EDGAR peak hours can exceed 40 filings/hour; use 100 to reduce silent misses
EDGAR_ATOM_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type=8-K&dateb=&owner=include&count=100&output=atom"
)
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
def _headers():
    user_agent = os.environ.get("SEC_EDGAR_USER_AGENT", "").strip()
    if not user_agent:
        raise ScannerNetworkError("SEC_EDGAR_USER_AGENT must be explicitly configured")
    return {"User-Agent": user_agent}

_CIK_RE = re.compile(r"\((\d+)\) \(Filer\)")
_NAME_RE = re.compile(r"8-K - (.+?) \(\d+\) \(Filer\)")
_ACCESSION_RE = re.compile(r"\d{10}-\d{2}-\d{6}")


@dataclass
class FilingResult:
    ticker: str | None
    company_name: str
    cik: str
    headline: str
    url: str
    filed_at: datetime


@dataclass
class FilingDetail:
    cik: str
    triggering_items: set
    triggering_doc_url: str | None
    recent_filings: list = field(default_factory=list)


def _get(url, fetcher=None, *, timeout=15):
    """Single HTTP entry point. Resolves `fetcher` at call time (so tests that
    patch `requests.get` work) and converts any transport error or non-200 into
    ScannerNetworkError so callers never see a raw requests exception."""
    fetcher = fetcher or requests.get
    try:
        resp = fetcher(url, headers=_headers(), timeout=timeout)
        resp.raise_for_status()
        return resp
    except ScannerNetworkError:
        raise
    except Exception as exc:
        # Exception TYPE (plus the HTTP status when there is one) only, and no chained cause: the raw
        # text of a header error quotes the contact User-Agent verbatim.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        detail = type(exc).__name__ + (f" (HTTP {status})" if isinstance(status, int) else "")
        raise ScannerNetworkError(f"EDGAR fetch failed for {url}: {detail}") from None


_SAFE_DOC_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*\Z")
_SAFE_ACCESSION_RE = re.compile(r"[0-9][0-9\-]*\Z")


def archive_url(cik, accession, primary_doc) -> str:
    """Filing-document URL. `accession` and `primary_doc` come from remote JSON, so they must be ONE plain
    folder name and ONE plain file name (no slash, backslash, '..', scheme, query or whitespace);
    anything else raises ScannerNetworkError so the caller fails closed instead of fetching it."""
    if not isinstance(accession, str) or not _SAFE_ACCESSION_RE.match(accession):
        raise ScannerNetworkError("EDGAR accession number malformed")
    if not isinstance(primary_doc, str) or ".." in primary_doc or not _SAFE_DOC_RE.match(primary_doc):
        raise ScannerNetworkError("EDGAR primary document name malformed")
    return (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
        f"{accession.replace('-', '')}/{primary_doc}"
    )


def _accession_from_url(url):
    m = _ACCESSION_RE.search(url or "")
    return m.group(0) if m else None


def _load_ticker_map(fetcher=None) -> dict[str, str]:
    resp = _get(TICKER_MAP_URL, fetcher, timeout=10)
    try:
        data = resp.json()
        return {
            str(int(v["cik_str"])).zfill(10): v["ticker"].upper()
            for v in data.values()
        }
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ScannerNetworkError(f"EDGAR ticker map malformed: {exc}") from exc


def normalize_ticker(s) -> str:
    """One canonical ticker form used before grouping, CIK lookup, and scan_candidates.
    Upper + strip; map the dot class-separator to SEC's hyphen ('BRK.B'->'BRK-B')."""
    return (s or "").strip().upper().replace(".", "-")


def ticker_to_ciks(fetcher=None) -> dict[str, set[str]]:
    """Strict SEC ticker -> set of zero-padded CIK10. Set-valued so the catalyst
    gate can fail closed when a ticker maps to ZERO or MULTIPLE CIKs (no auto-pass around the
    authoritative EDGAR dilution veto). A top-level fetch/parse
    failure raises ScannerNetworkError; a single malformed row is skipped."""
    resp = _get(TICKER_MAP_URL, fetcher, timeout=10)
    try:
        data = resp.json()
    except (ValueError, AttributeError) as exc:
        raise ScannerNetworkError(f"EDGAR ticker map malformed: {exc}") from exc
    out: dict[str, set[str]] = {}
    for v in data.values():
        try:
            cik = str(int(v["cik_str"])).zfill(10)
            t = normalize_ticker(v["ticker"])
        except (KeyError, TypeError, ValueError):
            continue
        if t:
            out.setdefault(t, set()).add(cik)
    return out


def fetch_recent_8k_filings(
    lookback_minutes: int = 60,
    ticker_map: dict[str, str] | None = None,
    fetcher=None,
) -> list[FilingResult]:
    if ticker_map is None:
        ticker_map = _load_ticker_map(fetcher=fetcher)

    resp = _get(EDGAR_ATOM_URL, fetcher)
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as exc:
        raise ScannerNetworkError(f"EDGAR atom feed malformed: {exc}") from exc

    ns = {"atom": ATOM_NS}
    now_utc = datetime.now(timezone.utc)
    cutoff_ts = now_utc.timestamp() - lookback_minutes * 60

    results = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        updated_el = entry.find("atom:updated", ns)
        link_el = entry.find("atom:link", ns)

        if title_el is None or updated_el is None:
            continue

        title = title_el.text or ""
        updated_str = (updated_el.text or "").strip()
        url = (link_el.attrib.get("href", "") if link_el is not None else "")

        try:
            filed_at = datetime.fromisoformat(updated_str)
        except ValueError:
            continue

        if filed_at.timestamp() < cutoff_ts:
            continue

        cik_m = _CIK_RE.search(title)
        if not cik_m:
            continue
        cik = cik_m.group(1).zfill(10)

        name_m = _NAME_RE.match(title)
        company_name = name_m.group(1) if name_m else title

        results.append(FilingResult(
            ticker=ticker_map.get(cik),
            company_name=company_name,
            cik=cik,
            headline=title,
            url=url,
            filed_at=filed_at,
        ))

    return results


def fetch_filing_detail(cik, *, trigger_url=None, now=None, fetcher=None) -> FilingDetail:
    """Pull the issuer's recent filings (submissions JSON), retaining everything
    within the WIDER 365-day reverse-split window with parsed dates. Select the
    triggering 8-K by accession parsed from `trigger_url` (the firehose filing's
    url); fall back to the most-recent 8-K."""
    now = now or datetime.now(timezone.utc)
    cik10 = str(int(cik)).zfill(10)
    resp = _get(SUBMISSIONS_URL.format(cik10=cik10), fetcher)
    try:
        recent = resp.json()["filings"]["recent"]
        forms = recent["form"]
        items = recent.get("items", [])
        dates = recent.get("filingDate", [])
        accs = recent.get("accessionNumber", [])
        docs = recent.get("primaryDocument", [])
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ScannerNetworkError(f"EDGAR submissions malformed for {cik}: {exc}") from exc

    cutoff = now.date() - timedelta(days=config.REVERSE_SPLIT_LOOKBACK_DAYS)
    rows = []
    for i in range(len(forms)):
        fd_str = dates[i] if i < len(dates) else ""
        try:
            fd = datetime.strptime(fd_str, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            # Fail-closed: an unparseable/missing date must NOT make an offering row vanish
            # from the veto. Treat it as today so it lands inside every window and is evaluated.
            fd = now.date()
        if fd < cutoff:
            continue
        rows.append({
            "form": forms[i],
            "items": {s.strip() for s in (items[i] or "").split(",") if s.strip()} if i < len(items) else set(),
            "filing_date": fd,
            "accession": accs[i] if i < len(accs) else "",
            "primary_doc": docs[i] if i < len(docs) else "",
        })

    trig = None
    trig_acc = _accession_from_url(trigger_url) if trigger_url else None
    if trig_acc:
        norm = trig_acc.replace("-", "")
        trig = next((r for r in rows if r["accession"].replace("-", "") == norm), None)
    if trig is None:
        trig = next((r for r in rows if r["form"] == "8-K"), None)

    trig_doc_url = None
    if trig and trig["primary_doc"]:
        try:
            trig_doc_url = archive_url(cik, trig["accession"], trig["primary_doc"])
        except ScannerNetworkError:
            trig_doc_url = None        # unsafe name from the remote JSON -> no body url -> caller rejects
    return FilingDetail(
        cik=cik10,
        triggering_items=(trig["items"] if trig else set()),
        triggering_doc_url=trig_doc_url,
        recent_filings=rows,
    )


def fetch_filing_body(url, *, fetcher=None, max_bytes=2_000_000) -> str:
    resp = _get(url, fetcher)
    return (resp.text or "")[:max_bytes]
