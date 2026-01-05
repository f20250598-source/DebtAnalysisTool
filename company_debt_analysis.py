"""
Company Debt Analysis Tool

This tool provides comprehensive debt analysis using the full financial data API endpoint.
It analyzes:
- Total debt levels and composition (short-term vs long-term)
- Cost of debt (Interest Expense / Total Debt)
- Leverage ratios (Debt-to-Equity, Debt-to-Assets)
- Net debt position (Total Debt - Cash)
- Debt trends over multiple years
- Interest coverage ratios
- High-cost borrowing reduction opportunities
"""

import sys
import os
import re
import requests
from typing import Dict, Any, List, Optional
import pandas as pd
from langchain_core.tools import tool

# ============================================================================
# Configuration
# ============================================================================
API_KEY = ""  # Paste your API key here

BASE_URL = "https://ac-api-server.vercel.app"
NSE_CSV_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
NSE_CSV_CACHE = os.path.join(os.path.expanduser("~"), ".nse_master.csv")


# ============================================================================
# Helper Functions - NSE Symbol Lookup
# ============================================================================
def _normalize(s: Optional[str]) -> str:
    """Normalize company name for fuzzy matching."""
    s = s or ""
    s = s.upper()
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _download_nse_csv(path: str) -> None:
    """Download NSE master CSV if not already cached."""
    try:
        resp = requests.get(NSE_CSV_URL, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to download NSE master CSV: {e}") from e

    with open(path, "wb") as f:
        f.write(resp.content)


def find_symbol_for_name(company_name: str) -> Optional[str]:
    """Return best-matching NSE symbol for the given company name, or None."""
    if not company_name or not isinstance(company_name, str):
        return None

    company_name = company_name.strip()
    if not company_name:
        return None

    # Check if it's already a symbol
    if company_name.endswith(".NS") or company_name.endswith(".BO"):
        return company_name

    # Download CSV if needed
    if not os.path.exists(NSE_CSV_CACHE):
        _download_nse_csv(NSE_CSV_CACHE)

    try:
        df = pd.read_csv(NSE_CSV_CACHE)
    except Exception as e:
        print(f"Error reading NSE CSV: {e}")
        return None

    if "SYMBOL" not in df.columns or "NAME OF COMPANY" not in df.columns:
        return None

    target = _normalize(company_name)
    exact_matches = []
    partial_matches = []
    best = None
    best_score = 0

    for _, row in df.iterrows():
        sym = str(row["SYMBOL"]).strip().upper()
        cname = str(row["NAME OF COMPANY"]).strip()
        norm = _normalize(cname)

        if not sym or not norm:
            continue

        sym_ns = sym + ".NS"

        # Exact match
        if norm == target:
            exact_matches.append(sym_ns)
            continue

        # Partial match
        if len(target) >= 3:
            if target in norm or norm in target:
                partial_matches.append(sym_ns)
                continue

            # Token overlap scoring
            tset = set(target.split())
            nset = set(norm.split())
            score = len(tset & nset)
            if score > best_score:
                best_score = score
                best = sym_ns

    if exact_matches:
        return exact_matches[0]
    if partial_matches:
        return partial_matches[0]
    return best


# ============================================================================
# Helper Functions - Data Extraction
# ============================================================================
def _get_numeric(value: Any) -> Optional[float]:
    """Convert any value to float, handling strings and None."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _get_year_from_record(record: Dict[str, Any]) -> Optional[int]:
    """Extract the year from a financial record."""
    for key in ("calendarYear", "calendar_year", "year", "fiscalYear", "fiscal_year"):
        if key in record:
            try:
                return int(record[key])
            except Exception:
                pass
    
    # Try to parse from date field
    for key in ("date", "period", "end_date", "reportDate"):
        if key in record and record[key]:
            try:
                date_str = str(record[key])
                if "-" in date_str:
                    year = date_str.split("-")[0]
                    return int(year)
            except Exception:
                pass
    
    return None


def _format_inr(amount: float) -> str:
    """Format large INR amounts in billions/millions."""
    if amount >= 1e9:
        return f"₹{amount/1e9:.2f}B"
    elif amount >= 1e6:
        return f"₹{amount/1e6:.2f}M"
    elif amount >= 1e3:
        return f"₹{amount/1e3:.2f}K"
    else:
        return f"₹{amount:.2f}"


def _get_sector_benchmarks(sector: str) -> Dict[str, float]:
    """Return sector-specific cost of debt benchmarks."""
    # Industry-specific benchmarks based on typical debt costs
    benchmarks = {
        "Technology": {"low": 5.0, "moderate": 8.0, "high": 12.0},
        "IT": {"low": 5.0, "moderate": 8.0, "high": 12.0},
        "Financial Services": {"low": 4.0, "moderate": 6.0, "high": 10.0},
        "Banking": {"low": 3.0, "moderate": 5.0, "high": 8.0},
        "Healthcare": {"low": 5.5, "moderate": 8.5, "high": 12.0},
        "Pharmaceuticals": {"low": 5.5, "moderate": 8.5, "high": 12.0},
        "Consumer Goods": {"low": 6.0, "moderate": 9.0, "high": 13.0},
        "Manufacturing": {"low": 7.0, "moderate": 10.0, "high": 14.0},
        "Real Estate": {"low": 8.0, "moderate": 11.0, "high": 15.0},
        "Construction": {"low": 8.5, "moderate": 12.0, "high": 16.0},
        "Energy": {"low": 6.5, "moderate": 9.5, "high": 13.0},
        "Utilities": {"low": 5.0, "moderate": 7.5, "high": 11.0},
        "Retail": {"low": 7.0, "moderate": 10.0, "high": 14.0},
        "Telecommunications": {"low": 6.0, "moderate": 9.0, "high": 13.0},
        "Transportation": {"low": 7.5, "moderate": 10.5, "high": 14.0},
    }
    
    # Default benchmarks if sector not found
    default = {"low": 6.0, "moderate": 9.0, "high": 13.0}
    
    return benchmarks.get(sector, default)


def _get_sector_peers_avg_cost(sector: str) -> Optional[float]:
    """Fetch average cost of debt for sector peers from API."""
    if not sector:
        return None
    
    try:
        # Get top companies in the sector
        url = f"{BASE_URL}/server/comparison/sector/{sector}"
        params = {"metric": "totalDebt", "limit": 10}
        headers = {"x-api-key": API_KEY}
        
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code != 200:
            return None
        
        data = resp.json()
        if "data" not in data or not data["data"]:
            return None
        
        # Calculate average cost of debt for peer companies
        peer_costs = []
        for company in data["data"]:
            symbol = company.get("symbol")
            if not symbol:
                continue
            
            # Fetch full financial data for each peer
            peer_url = f"{BASE_URL}/server/company/{symbol}"
            peer_resp = requests.get(peer_url, headers=headers, timeout=10)
            if peer_resp.status_code != 200:
                continue
            
            peer_data = peer_resp.json()
            if "data" not in peer_data or not peer_data["data"]:
                continue
            
            # Get latest record
            latest = peer_data["data"][0]
            int_exp = _get_numeric(latest.get("interestExpense"))
            total_debt = _get_numeric(latest.get("totalDebt"))
            
            if int_exp and total_debt and total_debt > 0:
                cost = (int_exp / total_debt) * 100
                peer_costs.append(cost)
        
        if peer_costs:
            return sum(peer_costs) / len(peer_costs)
    except Exception:
        pass
    
    return None


# ============================================================================
# Core Analysis Functions
# ============================================================================
@tool
def get_company_debt_analysis(symbol: str) -> str:
    """Analyze a company's debt position, leverage, and financing costs.
    
    This tool provides comprehensive debt analysis including:
    - Total debt levels and composition (short-term vs long-term)
    - Cost of debt calculation (Interest Expense / Total Debt)
    - Leverage ratios (Debt-to-Equity, Debt-to-Assets)
    - Net debt position (Total Debt - Cash & Equivalents)
    - Debt trends over recent fiscal years
    - Interest coverage ratio (EBITDA / Interest Expense)
    - Debt repayment analysis and high-cost borrowing recommendations
    
    Args:
        symbol: Stock symbol (e.g., 'TCS.NS')
        
    Returns:
        Formatted comprehensive debt analysis report as a string
    """
    if not symbol or not isinstance(symbol, str):
        return "Invalid input: `symbol` must be a non-empty string."

    # Use the comprehensive endpoint that includes all financial data
    url = f"{BASE_URL}/server/company/{symbol}"
    headers = {"x-api-key": API_KEY, "Accept": "application/json"}

    try:
        resp = requests.get(url, headers=headers, timeout=15)
    except requests.RequestException as e:
        return f"Network error while requesting financial data for {symbol}: {e}"

    if resp.status_code != 200:
        if resp.status_code == 403:
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            return (
                f"API request for {symbol} returned 403 Forbidden. Server message: {body}.\n"
                "Check that your API key is set correctly."
            )
        text = resp.text.strip()
        snippet = (text[:400] + "...") if len(text) > 400 else text
        return f"API request failed for {symbol} with status {resp.status_code}. Response: {snippet}"

    try:
        data = resp.json()
    except Exception:
        return f"API returned non-JSON response for {symbol}. Raw response: {resp.text[:400]}"

    if "data" not in data or not isinstance(data["data"], list) or not data["data"]:
        return f"API response for {symbol} does not contain financial records."

    records = data["data"]
    
    # Extract company sector from first record
    company_sector = None
    if records and len(records) > 0:
        company_sector = records[0].get("sector", None)
    
    # Get sector benchmarks and peer average
    sector_benchmarks = _get_sector_benchmarks(company_sector) if company_sector else _get_sector_benchmarks("General")
    sector_avg_cost = None
    
    # Try to get actual sector peer average (may take time, so we'll continue if it fails)
    if company_sector:
        try:
            sector_avg_cost = _get_sector_peers_avg_cost(company_sector)
        except Exception:
            pass
    
    # Extract and organize data from all available years
    financial_data = []
    for rec in records:
        year = _get_year_from_record(rec)
        if not year:
            continue
        
        financial_data.append({
            "year": year,
            "short_term_debt": _get_numeric(rec.get("shortTermDebt")),
            "long_term_debt": _get_numeric(rec.get("longTermDebt")),
            "total_debt": _get_numeric(rec.get("totalDebt")),
            "cash": _get_numeric(rec.get("cashAndCashEquivalents")),
            "net_debt": _get_numeric(rec.get("netDebt")),
            "total_assets": _get_numeric(rec.get("totalAssets")),
            "total_equity": _get_numeric(rec.get("totalEquity")),
            "interest_expense": _get_numeric(rec.get("interestExpense")),
            "interest_income": _get_numeric(rec.get("interestIncome")),
            "ebitda": _get_numeric(rec.get("ebitda")),
            "revenue": _get_numeric(rec.get("revenue")),
            "free_cash_flow": _get_numeric(rec.get("freeCashFlow")),
            "capital_lease_obligations": _get_numeric(rec.get("capitalLeaseObligations")),
        })
    
    if not financial_data:
        return f"No usable financial data found for {symbol}."
    
    # Sort by year descending (most recent first)
    financial_data.sort(key=lambda x: x["year"], reverse=True)
    
    # Build focused report - only 5 sections requested
    lines = []
    lines.append(f"DEBT ANALYSIS FOR {symbol}")
    lines.append("=" * 90)
    lines.append("")
    
    # ========================================================================
    # SECTION 1: COST OF DEBT
    # ========================================================================
    lines.append("1. COST OF DEBT")
    if company_sector:
        lines.append(f"   Sector: {company_sector}")
    lines.append("-" * 90)
    
    # Show sector benchmarks
    if sector_benchmarks:
        lines.append(f"\nSector Benchmarks ({company_sector or 'General'}):")
        lines.append(f"  Excellent: <{sector_benchmarks['low']:.1f}%")
        lines.append(f"  Good: {sector_benchmarks['low']:.1f}%-{sector_benchmarks['moderate']:.1f}%")
        lines.append(f"  Moderate: {sector_benchmarks['moderate']:.1f}%-{sector_benchmarks['high']:.1f}%")
        lines.append(f"  High: >{sector_benchmarks['high']:.1f}%")
    
    # Show sector peer average if available
    if sector_avg_cost:
        lines.append(f"\nSector Peer Average (Top 10 companies): {sector_avg_cost:.2f}%")
    
    for rec in financial_data:
        year = rec["year"]
        int_exp = rec["interest_expense"] or 0
        total_debt = rec["total_debt"] or ((rec["short_term_debt"] or 0) + (rec["long_term_debt"] or 0))
        
        lines.append(f"\nFY{year}:")
        lines.append(f"  Interest Expense: {_format_inr(int_exp):>15s}")
        lines.append(f"  Total Debt:       {_format_inr(total_debt):>15s}")
        
        if total_debt > 0:
            cost_of_debt = (int_exp / total_debt) * 100
            lines.append(f"  Cost of Debt:     {cost_of_debt:>14.2f}%")
            
            # Sector-specific interpretation
            if cost_of_debt < sector_benchmarks['low']:
                lines.append(f"  → Excellent - Well below sector average")
            elif cost_of_debt < sector_benchmarks['moderate']:
                lines.append(f"  → Good - Competitive within sector")
            elif cost_of_debt < sector_benchmarks['high']:
                lines.append(f"  → Moderate - Within sector range")
            else:
                lines.append(f"  → High - Above sector benchmark, refinancing recommended")
            
            # Compare with peer average if available
            if sector_avg_cost:
                diff = cost_of_debt - sector_avg_cost
                if abs(diff) < 0.5:
                    lines.append(f"  → On par with sector peers")
                elif diff < 0:
                    lines.append(f"  → {abs(diff):.2f}% lower than sector average ✓")
                else:
                    lines.append(f"  → {diff:.2f}% higher than sector average")
        else:
            lines.append(f"  Cost of Debt:     N/A (minimal/no debt)")
    
    lines.append("")
    
    # ========================================================================
    # SECTION 2: DEBT COMPOSITION
    # ========================================================================
    lines.append("2. DEBT COMPOSITION")
    lines.append("-" * 90)
    
    for i, rec in enumerate(financial_data):
        year = rec["year"]
        st_debt = rec["short_term_debt"] or 0
        lt_debt = rec["long_term_debt"] or 0
        total_debt = rec["total_debt"] or (st_debt + lt_debt)
        
        lines.append(f"\nFY{year}:")
        lines.append(f"  Short-term Debt:  {_format_inr(st_debt):>15s}")
        lines.append(f"  Long-term Debt:   {_format_inr(lt_debt):>15s}")
        lines.append(f"  TOTAL DEBT:       {_format_inr(total_debt):>15s}")
        
        if total_debt > 0:
            st_pct = (st_debt / total_debt) * 100
            lt_pct = (lt_debt / total_debt) * 100
            lines.append(f"  Composition:      {st_pct:.1f}% ST / {lt_pct:.1f}% LT")
        
        # Show YoY change
        if i < len(financial_data) - 1:
            prev_rec = financial_data[i + 1]
            prev_total_debt = prev_rec["total_debt"] or (prev_rec["short_term_debt"] or 0) + (prev_rec["long_term_debt"] or 0)
            if prev_total_debt > 0:
                change = total_debt - prev_total_debt
                change_pct = (change / prev_total_debt) * 100
                direction = "↑" if change > 0 else "↓"
                lines.append(f"  YoY Change:       {direction} {abs(change_pct):.2f}% ({_format_inr(abs(change))})")
    
    lines.append("")
    
    # ========================================================================
    # SECTION 3: LEVERAGE LEVELS
    # ========================================================================
    lines.append("3. LEVERAGE LEVELS")
    lines.append("-" * 90)
    
    for rec in financial_data:
        year = rec["year"]
        total_debt = rec["total_debt"] or ((rec["short_term_debt"] or 0) + (rec["long_term_debt"] or 0))
        total_equity = rec["total_equity"] or 0
        total_assets = rec["total_assets"] or 0
        cash = rec["cash"] or 0
        net_debt = rec["net_debt"] or (total_debt - cash)
        
        lines.append(f"\nFY{year}:")
        
        # Debt-to-Equity
        if total_equity > 0:
            de_ratio = total_debt / total_equity
            lines.append(f"  Debt-to-Equity:   {de_ratio:>15.4f}")
            if de_ratio < 0.5:
                lines.append(f"  → Conservative leverage")
            elif de_ratio < 1.0:
                lines.append(f"  → Moderate leverage")
            elif de_ratio < 2.0:
                lines.append(f"  → Elevated leverage")
            else:
                lines.append(f"  → High leverage - monitor closely")
        else:
            lines.append(f"  Debt-to-Equity:   N/A")
        
        # Debt-to-Assets
        if total_assets > 0:
            da_ratio = total_debt / total_assets
            lines.append(f"  Debt-to-Assets:   {da_ratio:>15.4f}")
            if da_ratio < 0.3:
                lines.append(f"  → Low financial risk")
            elif da_ratio < 0.5:
                lines.append(f"  → Moderate financial risk")
            else:
                lines.append(f"  → Higher financial risk")
        else:
            lines.append(f"  Debt-to-Assets:   N/A")
        
        # Net Debt
        lines.append(f"  Net Debt:         {_format_inr(net_debt):>15s}")
        if net_debt < 0:
            lines.append(f"  → Net cash position")
    
    lines.append("")
    
    # ========================================================================
    # SECTION 4: DEBT REPAYMENT SCHEDULES
    # ========================================================================
    lines.append("4. DEBT REPAYMENT SCHEDULES")
    lines.append("-" * 90)
    
    latest = financial_data[0]
    year = latest["year"]
    fcf = latest["free_cash_flow"] or 0
    total_debt = latest["total_debt"] or ((latest["short_term_debt"] or 0) + (latest["long_term_debt"] or 0))
    st_debt = latest["short_term_debt"] or 0
    lt_debt = latest["long_term_debt"] or 0
    
    lines.append(f"\nFY{year} Repayment Analysis:")
    lines.append(f"  Free Cash Flow:       {_format_inr(fcf):>15s}")
    lines.append(f"  Short-term Debt:      {_format_inr(st_debt):>15s} (due within 1 year)")
    lines.append(f"  Long-term Debt:       {_format_inr(lt_debt):>15s} (due after 1 year)")
    lines.append(f"  Total Debt:           {_format_inr(total_debt):>15s}")
    
    if fcf > 0 and total_debt > 0:
        lines.append(f"\n  Repayment Capacity:")
        years_to_payoff = total_debt / fcf
        lines.append(f"  • Full debt payoff:   {years_to_payoff:.2f} years (at current FCF)")
        
        if fcf >= st_debt:
            lines.append(f"  • ✓ FCF covers all short-term obligations")
        else:
            shortfall = st_debt - fcf
            lines.append(f"  • ⚠ FCF shortfall for ST debt: {_format_inr(shortfall)}")
            lines.append(f"  • Action: May need to refinance or use cash reserves")
    
    # Show trend over years
    if len(financial_data) >= 2:
        lines.append(f"\n  Debt Trend (Past {len(financial_data)} Years):")
        for rec in financial_data:
            y = rec["year"]
            td = rec["total_debt"] or ((rec["short_term_debt"] or 0) + (rec["long_term_debt"] or 0))
            lines.append(f"  • FY{y}: {_format_inr(td)}")
    
    lines.append("")
    
    # ========================================================================
    # SECTION 5: REDUCTION OF HIGH COST BORROWINGS
    # ========================================================================
    lines.append("5. REDUCTION OF HIGH COST BORROWINGS")
    lines.append("-" * 90)
    lines.append("")
    
    # Analyze latest year for high-cost borrowing recommendations
    latest = financial_data[0]
    total_debt = latest["total_debt"] or ((latest["short_term_debt"] or 0) + (latest["long_term_debt"] or 0))
    cost_of_debt = ((latest["interest_expense"] or 0) / total_debt * 100) if total_debt > 0 else 0
    st_debt = latest["short_term_debt"] or 0
    lt_debt = latest["long_term_debt"] or 0
    cash = latest["cash"] or 0
    fcf = latest["free_cash_flow"] or 0
    
    lines.append(f"Current Cost of Debt: {cost_of_debt:.2f}%")
    lines.append(f"Annual Interest Burden: {_format_inr(latest['interest_expense'] or 0)}")
    lines.append("")
    
    # Recommendation logic for cost reduction
    if cost_of_debt > 12:
        lines.append("⚠ HIGH COST OF DEBT - IMMEDIATE ACTION REQUIRED")
        lines.append("")
        
    elif cost_of_debt > 8:
        lines.append("⚡ MODERATE COST - OPTIMIZATION OPPORTUNITIES")
        lines.append("")
        
    else:
        lines.append("✓ LOW COST OF DEBT - WELL OPTIMIZED")
        lines.append("")
        lines.append("Current Position:")
        lines.append(f"  • Cost of debt ({cost_of_debt:.2f}%) is competitive")
        lines.append("  • Continue monitoring for further optimization opportunities")
    
    lines.append("")
    
    # Prepayment opportunities
    if cash > st_debt and cost_of_debt > 8:
        lines.append("Cash Position Analysis:")
        lines.append(f"  • Available cash: {_format_inr(cash)}")
        lines.append(f"  • Short-term debt: {_format_inr(st_debt)}")
        excess_cash = cash - st_debt
        lines.append(f"  • Excess cash: {_format_inr(excess_cash)}")
        lines.append("")
        lines.append("  💡 Opportunity: Use excess cash to prepay high-cost debt")
        if cost_of_debt > 8:
            annual_savings = (cost_of_debt / 100) * min(excess_cash, lt_debt)
            lines.append(f"     Potential annual interest savings: {_format_inr(annual_savings)}")
    
    # Short-term heavy warning
    st_pct = (st_debt / total_debt * 100) if total_debt > 0 else 0
    if st_pct > 40:
        lines.append("")
        lines.append("⚠ High Short-term Debt Concentration:")
        lines.append(f"  • {st_pct:.1f}% of total debt is short-term")
        lines.append("  • Risk: Higher refinancing risk and typically higher rates")
        lines.append("  • Action: Convert portion to long-term debt for:")
        lines.append("    - Lower rates (LT debt typically 1-2% cheaper)")
        lines.append("    - Improved stability and reduced rollover risk")
    
    # Debt growth trend
    if len(financial_data) >= 2:
        current_debt = total_debt
        prev_debt = financial_data[1]["total_debt"] or ((financial_data[1]["short_term_debt"] or 0) + (financial_data[1]["long_term_debt"] or 0))
        if current_debt > prev_debt * 1.2:
            lines.append("")
            lines.append("⚠ Rapid Debt Growth Detected:")
            growth_pct = ((current_debt - prev_debt) / prev_debt * 100) if prev_debt > 0 else 0
            lines.append(f"  • Debt increased by {growth_pct:.1f}% YoY")
            lines.append("  • Action: Ensure debt is funding profitable growth")
            lines.append("  • Monitor: Return on invested capital > cost of debt")
    
    lines.append("")
    lines.append("=" * 90)
    lines.append("Analysis complete. All figures in INR (Indian Rupees).")
    
    return "\n".join(lines)


@tool
def find_company_symbol(company_name: str) -> str:
    """Look up the NSE stock symbol for a given company name.
    
    Args:
        company_name: Name of the company (e.g., 'Tata Consultancy Services')
        
    Returns:
        NSE symbol with .NS suffix (e.g., 'TCS.NS') or error message
    """
    if not company_name or not isinstance(company_name, str):
        return "Invalid input: `company_name` must be a non-empty string."
    
    symbol = find_symbol_for_name(company_name)
    if symbol:
        return f"Symbol for '{company_name}': {symbol}"
    else:
        return f"Could not find symbol for company: {company_name}"


# ============================================================================
# CLI Interface
# ============================================================================
def main():
    """Command-line interface for the debt analysis tool."""
    # Check if company name/symbol provided as argument
    if len(sys.argv) > 1:
        input_name = sys.argv[1].strip()
    else:
        # Interactive mode - prompt user
        try:
            input_name = input("Enter company name or symbol: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nNo company name provided. Exiting.")
            sys.exit(1)
    
    if not input_name:
        print("No company name provided. Exiting.")
        sys.exit(1)
    
    # Try to determine if it's a symbol or company name
    if input_name.endswith(".NS") or input_name.endswith(".BO"):
        symbol = input_name
        print(f"Using symbol: {symbol}\n")
    else:
        print(f"Looking up: {input_name}")
        symbol = find_symbol_for_name(input_name)
        if not symbol:
            print(f"Could not find symbol for: {input_name}")
            sys.exit(1)
        print(f"Found symbol: {symbol}\n")
    
    # Run the analysis
    result = get_company_debt_analysis.invoke({"symbol": symbol})
    print(result)


if __name__ == "__main__":
    main()
