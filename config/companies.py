"""Reference metadata for the 15 modeled Philippine Stock Exchange (PSE) companies.

Maintained independently in this repository purely for display grouping,
sector roll-ups, and validation checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class CompanyMetadata:
    """Reference metadata for one PSE company."""

    symbol: str
    name: str
    sector: str


COMPANIES: Final[tuple[CompanyMetadata, ...]] = (
    CompanyMetadata("ALI", "Ayala Land, Inc.", "Property"),
    CompanyMetadata("APX", "Apex Mining Co., Inc.", "Mining and Oil"),
    CompanyMetadata("BPI", "Bank of the Philippine Islands", "Financials"),
    CompanyMetadata("GLO", "Globe Telecom, Inc.", "Services"),
    CompanyMetadata(
        "ICT", "International Container Terminal Services, Inc.", "Services"
    ),
    CompanyMetadata("JFC", "Jollibee Foods Corporation", "Industrial"),
    CompanyMetadata("MBT", "Metropolitan Bank & Trust Company", "Financials"),
    CompanyMetadata("MEG", "Megaworld Corporation", "Property"),
    CompanyMetadata("MER", "Manila Electric Company", "Industrial"),
    CompanyMetadata("NIKL", "Nickel Asia Corporation", "Mining and Oil"),
    CompanyMetadata("PGOLD", "Puregold Price Club, Inc.", "Services"),
    CompanyMetadata("SCC", "Semirara Mining and Power Corporation", "Mining and Oil"),
    CompanyMetadata("SECB", "Security Bank Corporation", "Financials"),
    CompanyMetadata("SHLPH", "Shell Pilipinas Corporation", "Industrial"),
    CompanyMetadata("SMPH", "SM Prime Holdings, Inc.", "Property"),
)

COMPANY_BY_SYMBOL: Final[dict[str, CompanyMetadata]] = {
    c.symbol: c for c in COMPANIES
}

SECTOR_BY_SYMBOL: Final[dict[str, str]] = {
    c.symbol: c.sector for c in COMPANIES
}

ALL_SYMBOLS: Final[tuple[str, ...]] = tuple(c.symbol for c in COMPANIES)


def get_company(symbol: str) -> CompanyMetadata:
    """Lookup company metadata by ticker symbol."""
    norm = symbol.strip().upper()
    if norm not in COMPANY_BY_SYMBOL:
        raise KeyError(f"Unknown company symbol: {symbol}")
    return COMPANY_BY_SYMBOL[norm]

