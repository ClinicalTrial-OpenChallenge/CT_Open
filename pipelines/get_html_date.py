import string
import copy
import pickle
import random
from typing import List, Tuple
from bs4 import BeautifulSoup, Tag, NavigableString
from tqdm import tqdm
import tiktoken
import numpy as np
encoding = tiktoken.encoding_for_model("gpt-4o")
# from generation.generate import generate
import random
# from parse_html_page import parse_html_page
import json
import re
import js2py
import re
from tqdm import tqdm
import json
from dateutil.parser import parse
from dateutil import parser
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from typing import Optional
import datetime as _dt
from concurrent.futures import ProcessPoolExecutor, as_completed
from bs4 import BeautifulSoup
from tqdm import tqdm

def to_ymd(s: str, *, dayfirst: bool = True, yearfirst: bool = False) -> Optional[str]:
    """
    Strictly convert supported date formats to ISO-8601 without guessing.

    Returns a reduced-precision ISO date string:
      - 'YYYY-MM-DD' for full dates
      - 'YYYY-MM'     for year+month inputs
      - 'YYYY'        for year-only inputs

    Behavior on problems:
      - If month is out of 1..12 or day is out of range for the month/year, returns None.
      - If the input is unrecognized/ambiguous or anything unexpected happens, returns None.
      - Never raises.

    Supported patterns (found anywhere in the string; ignores extra words/symbols):
      - Year-first numeric full:         YYYY[-/.]MM[-/.]DD
      - Day-first numeric full:          DD[-/.]MM[-/.]YYYY
      - Day-first with 2-digit year:     DD[-/.]MM[-/.]YY  (pivot: yy<=25 -> 20yy else 19yy)
      - Compact full:                    YYYYMMDD
      - Month-name full:                 25 Aug 2025 / Aug 25 2025 / 2025 Aug 25
      - ISO/RFC-ish datetime:            2025-06-12T12:00:00Z (date part extracted)
      - Year+month (numeric):            YYYY-MM / YYYY/MM / YYYY.MM / YYYYMM
      - Year+month (with names):         Aug 2025 / 2025 Aug
      - Year-only:                       YYYY

    Notes:
      - `dayfirst` and `yearfirst` are ignored (kept for backward compatibility).
    """
    try:
        if not isinstance(s, str) or not s.strip():
            return None

        raw = s.strip()

        # Normalize: remove ordinal suffixes (1st -> 1), commas, squeeze spaces
        cleaned = re.sub(r'(\d{1,2})(st|nd|rd|th)\b', r'\1', raw, flags=re.IGNORECASE)
        cleaned = cleaned.replace(",", " ")
        cleaned = re.sub(r'\s+', " ", cleaned).strip()

        # ---------- helpers ----------
        def _finalize_y(y: int) -> Optional[str]:
            try:
                _validate_y(y)
                return f"{y:04d}"
            except Exception:
                return None

        def _finalize_ym(y: int, m: int) -> Optional[str]:
            try:
                _validate_ym(y, m)
                return f"{y:04d}-{m:02d}"
            except Exception:
                return None

        def _finalize_ymd(y: int, m: int, d: int) -> Optional[str]:
            try:
                _validate_ymd(y, m, d)  # may raise month/day range errors
                return _dt.date(y, m, d).isoformat()
            except Exception:
                return None

        # Month map
        mon_map = {
            'jan': 1, 'january': 1,
            'feb': 2, 'february': 2,
            'mar': 3, 'march': 3,
            'apr': 4, 'april': 4,
            'may': 5,
            'jun': 6, 'june': 6,
            'jul': 7, 'july': 7,
            'aug': 8, 'august': 8,
            'sep': 9, 'sept': 9, 'september': 9,
            'oct': 10, 'october': 10,
            'nov': 11, 'november': 11,
            'dec': 12, 'december': 12,
        }

        # ---------- 1) ISO/RFC-ish datetime inside text: YYYY-MM-DD[ time... ] ----------
        m = re.search(
            r'\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})'
            r'(?:[ T]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)?\b',
            cleaned
        )
        if m:
            return _finalize_ymd(int(m['y']), int(m['m']), int(m['d']))

        # ---------- 2) Full dates (year-first numeric) anywhere ----------
        m = re.search(r'\b(?P<y>\d{4})[-/.](?P<m>\d{1,2})[-/.](?P<d>\d{1,2})\b', cleaned)
        if m:
            return _finalize_ymd(int(m['y']), int(m['m']), int(m['d']))

        # ---------- 3) Full dates (day-first numeric, year last) anywhere ----------
        m = re.search(r'\b(?P<d>\d{1,2})[-/.](?P<m>\d{1,2})[-/.](?P<y>\d{4})\b', cleaned)
        if m:
            return _finalize_ymd(int(m['y']), int(m['m']), int(m['d']))

        # Day-first with 2-digit year + pivot
        m = re.search(r'\b(?P<d>\d{1,2})[-/.](?P<m>\d{1,2})[-/.](?P<y>\d{2})\b', cleaned)
        if m:
            yy = int(m['y'])
            y = 2000 + yy if yy <= 25 else 1900 + yy
            return _finalize_ymd(y, int(m['m']), int(m['d']))

        # ---------- 4) Compact full date YYYYMMDD anywhere ----------
        m = re.search(r'\b(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})\b', cleaned)
        if m:
            return _finalize_ymd(int(m['y']), int(m['m']), int(m['d']))

        # ---------- 5) Month-name full dates anywhere ----------
        # D Mon Y
        m = re.search(r'\b(?P<d>\d{1,2})\s+(?P<mon>[A-Za-z]+)\s+(?P<y>\d{4})\b', cleaned)
        if m:
            mon = m.group('mon').lower()
            if mon in mon_map:
                return _finalize_ymd(int(m.group('y')), mon_map[mon], int(m.group('d')))
        # Mon D Y
        m = re.search(r'\b(?P<mon>[A-Za-z]+)\s+(?P<d>\d{1,2})\s+(?P<y>\d{4})\b', cleaned)
        if m:
            mon = m.group('mon').lower()
            if mon in mon_map:
                return _finalize_ymd(int(m.group('y')), mon_map[mon], int(m.group('d')))
        # Y Mon D
        m = re.search(r'\b(?P<y>\d{4})\s+(?P<mon>[A-Za-z]+)\s+(?P<d>\d{1,2})\b', cleaned)
        if m:
            mon = m.group('mon').lower()
            if mon in mon_map:
                return _finalize_ymd(int(m.group('y')), mon_map[mon], int(m.group('d')))

        # ---------- 6) Year + month (numeric) anywhere ----------
        # Match YYYY-MM / YYYY/MM / YYYY.MM that are NOT followed by another sep+DD
        m = re.search(r'\b(?P<y>\d{4})[-/.](?P<m>\d{1,2})(?![-/.]\d)\b', cleaned)
        if m:
            return _finalize_ym(int(m['y']), int(m['m']))
        # Compact YYYYMM (but not YYYYMMDD handled above)
        m = re.search(r'\b(?P<y>\d{4})(?P<m>\d{2})\b', cleaned)
        if m:
            end = m.end()
            if end == len(cleaned) or not cleaned[end].isdigit():
                return _finalize_ym(int(m['y']), int(m['m']))

        # ---------- 7) Year + month (with names) anywhere ----------
        # Mon YYYY
        m = re.search(r'\b(?P<mon>[A-Za-z]+)\s+(?P<y>\d{4})\b', cleaned)
        if m:
            mon = m.group('mon').lower()
            if mon in mon_map:
                return _finalize_ym(int(m.group('y')), mon_map[mon])
        # YYYY Mon
        m = re.search(r'\b(?P<y>\d{4})\s+(?P<mon>[A-Za-z]+)\b', cleaned)
        if m:
            mon = m.group('mon').lower()
            if mon in mon_map:
                return _finalize_ym(int(m.group('y')), mon_map[mon])

        # ---------- 8) Year-only anywhere ----------
        m = re.search(r'\b(?P<y>\d{4})\b', cleaned)
        if m:
            return _finalize_y(int(m['y']))

        # Nothing matched
        return None

    except Exception:
        # Any truly unexpected error -> None, never raise
        return None


def _validate_y(y: int) -> None:
    if not (1 <= y <= 9999):
        raise ValueError("Year out of range.")

def _validate_ym(y: int, m: int) -> None:
    _validate_y(y)
    if not (1 <= m <= 12):
        raise ValueError("Month out of range.")

def _validate_ymd(y: int, m: int, d: int) -> None:
    _validate_ym(y, m)
    # Let datetime validate the day (handles leap years)
    _dt.datetime(year=y, month=m, day=d)

def clean_date(raw: str | None) -> str | None:
    """
    Cleans a raw date string into 'YYYY-MM-DD' format.
    """
    return to_ymd(raw)


def find_date_in_raw_text(text: str) -> str | None:
    """
    Finds 'datePublished' in the raw text.
    - Returns None if there is not exactly one occurrence.
    - Returns None if the date string cannot be parsed as a valid date.
    """
    if not text:
        return None

    matches = re.findall(r'"datePublished"\s*:\s*"(.*?)"', text)

    # Ensure exactly one occurrence
    if len(matches) != 1:
        return None

    date_str = matches[0]

    # Try parsing the date with multiple common formats
    date_str = to_ymd(date_str)
    return date_str  # Valid date format found

def find_date_in_meta_tags(soup: BeautifulSoup) -> str | None:
    """
    Finds the most reliable publication date by searching through HTML in a
    prioritized order.

    The search order is:
    1. JSON-LD structured data ('datePublished').
    2. High-priority meta tags (og:published_time, article:published_time).
    3. The <time> tag's 'datetime' attribute.
    4. Schema.org 'itemprop' attributes.
    5. Lower-priority meta tags.

    Args:
        soup: A BeautifulSoup object representing the parsed HTML of a page.

    Returns:
        The first reliable date found as a 'YYYY-MM-DD' formatted string,
        or None if no date is found.
    """
    # Note: Strategy 1 (JSON-LD) from the docstring is not included in the
    # original code snippet and is therefore omitted here.

    # --- Strategy 2: High-Priority Meta Tags ---
    # Find all high-priority dates and ensure they are consistent.
    high_priority_keys = ['article:published_time', 'og:published_time']
    high_priority_dates = set()
    for key in high_priority_keys:
        tag = soup.find('meta', attrs={'property': key})
        if tag and (date_str := tag.get('content')):
            try:
                # Parse and normalize the date, then add to the set.
                parsed_date = to_ymd(date_str)
                high_priority_dates.add(parsed_date)
            except parser.ParserError:
                # Ignore tags with unparseable date strings.
                continue

    # If any dates were found, assert they are all the same before returning.
    if high_priority_dates:
        if len(high_priority_dates) != 1:
            pass
        else:
            return high_priority_dates.pop()

    # --- Strategy 5: Lower-Priority Meta Tags ---
    # Collect all dates from less-standard tags and ensure consistency.
    low_priority_keys = [
        'publication_date', 'PublishDate', 'publish_date', 'sailthru.date',
        'dc.date.issued', 'article:published_time', 'og:published_time', 'DC.Date.created','dc.Date',
        'dc.Date.created',"pub_date",'dc.Date.issued', "publicationDate","publishDate", "datePublished","DC.Date","dc.date"
    ]
    low_priority_dates = set()
    for key in low_priority_keys:
        # Note: This loop only checks for the 'name' attribute as in the original code.
        # Some keys (e.g., 'og:published_time') are typically found with the 'property' attribute.
        tag = soup.find('meta', attrs={'name': key})
        if tag and (date_str := tag.get('content')):
            try:
                # Parse and normalize the date, then add to the set.
                parsed_date = to_ymd(date_str)
                low_priority_dates.add(parsed_date)
            except parser.ParserError:
                # Ignore tags with unparseable date strings.
                continue

    # If any dates were found, assert they are all the same before returning.
    if low_priority_dates:
        if len(low_priority_dates) != 1:
            return "more than one publish date found"
        return low_priority_dates.pop()

    return None # Return None if no reliable date was found

def find_date_in_publish_tags(soup: BeautifulSoup) -> str | None:
    # # --- Strategy 3: The <time> Tag ---
    time_tag = soup.find('time', attrs={'itemprop': 'datePublished'})
    if time_tag and (date_str := time_tag.get('datetime')):
        try:
            return to_ymd(date_str)
        except parser.ParserError:
            pass # Continue if parsing fails

    # --- Strategy 4: Schema.org 'itemprop' Attribute ---
    # This specifically looks for the 'content' attribute which is machine-readable
    itemprop_tag = soup.find(attrs={'itemprop': 'datePublished'})
    if itemprop_tag and (date_str := itemprop_tag.get('content')):
        try:
            return to_ymd(date_str)
        except parser.ParserError:
            pass
    return None

def find_date_citation_date(soup: BeautifulSoup) -> str | None:
    low_priority_keys = [
        "citation_publication_date"
    ]
    low_priority_dates = set()
    for key in low_priority_keys:
        # Note: This loop only checks for the 'name' attribute as in the original code.
        # Some keys (e.g., 'og:published_time') are typically found with the 'property' attribute.
        tag = soup.find('meta', attrs={'name': key})
        if tag and (date_str := tag.get('content')):
            try:
                # Parse and normalize the date, then add to the set.
                parsed_date = to_ymd(date_str)
                low_priority_dates.add(parsed_date)
            except parser.ParserError:
                # Ignore tags with unparseable date strings.
                continue

    # If any dates were found, assert they are all the same before returning.
    if low_priority_dates:
        return min(low_priority_dates)

    return None # Return None if no reliable date was found

def find_earliest_date(soup: BeautifulSoup) -> str | None:
    """
    Finds the earliest date from all <time> tags on a page.

    It checks the 'datetime' attribute first, then the tag's content.
    It robustly parses various date formats and returns the earliest
    date found in YYYY-MM-DD format.
    """
    earliest_date = None
    time_tags = soup.find_all('time')

    for tag in time_tags:
        date_string = tag.get('datetime') or tag.string
        if not date_string:
            continue

        try:
            current_date = to_ymd(date_string.strip())

            if earliest_date is None or current_date < earliest_date:
                earliest_date = current_date
        except Exception as e:
            continue

    # If an earliest date was found, return it in the specified format
    if earliest_date:
        # --- THIS LINE IS CHANGED ---
        return to_ymd(earliest_date)

    return None


def find_date_in_time_tags(soup: BeautifulSoup) -> str | None:
    """
    Finds the first <time> tag with a valid datetime attribute.
    """
    # Find all <time> tags
    time_tags = soup.find_all('time')
    for tag in time_tags:
        # The 'datetime' attribute is the most reliable
        if tag.get('datetime'):
            return tag['datetime']
        # Fallback to the tag's text content if 'datetime' is missing
        if tag.string:
            return tag.string
    return None

def is_valid_date(date_str: str) -> bool:
    if not date_str:
        return False
    try:
        year_match = re.search(r'(\d{4})', date_str)
        if year_match:
            year = int(year_match.group(1))
            current_year = datetime.now().year
            if year > current_year:
                return False
            if year < 1990:
                return False
        return True
    except (ValueError, TypeError):
        return False

def find_date_in_visible_text(soup: BeautifulSoup) -> str | None:
    if not soup:
        return None

    main_content = soup.find('article') or soup.find('main')
    if not main_content:
        return None

    hinted_elements = main_content.find_all(
        attrs={'class': re.compile(r'date|time|publish|post-meta|byline|entry-date', re.I)}
    )
    hinted_elements.extend(
        main_content.find_all(
            attrs={'id': re.compile(r'date|time|publish|meta|byline', re.I)}
        )
    )

    generic_date_pattern = re.compile(r'([A-Za-z]{3,9}\s\d{1,2},?\s\d{4}|\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})')

    for element in hinted_elements:
        element_text = element.get_text(strip=True)
        match = generic_date_pattern.search(element_text)
        if match:
            date_str = match.group(1)
            if is_valid_date(date_str):
                return date_str

    top_content_text = main_content.get_text(separator=' ', strip=True)[:400]

    keyword_date_pattern = re.compile(
        r'\b(?:Published|Posted|Updated|On)\b\s*:?\s*(?:on\s*)?'
        r'([A-Za-z]{3,9}\s\d{1,2},?\s\d{4}|\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})',
        re.I
    )

    match = keyword_date_pattern.search(top_content_text)
    if match:
        date_str = match.group(1)
        if is_valid_date(date_str):
            return date_str

    return None

def get_dd_date(soup: BeautifulSoup) -> str | None:
    """
    Finds the publication date from a BeautifulSoup object.

    This function looks for a <dt> tag containing the exact text "Date:"
    and returns the text of the immediately following <dd> tag.

    Args:
        soup: A BeautifulSoup object of the parsed HTML page.

    Returns:
        A string containing the date if found, otherwise None.
    """
    try:
        # Find all <dt> (definition term) tags in the document
        dt_elements = soup.find_all('dt')

        # Loop through each <dt> tag found
        for dt in dt_elements:
            # Check if the stripped text of the tag is exactly "Date:"
            if dt.get_text(strip=True) == 'Date:':
                # If it is, find the immediate next sibling tag
                next_dd = dt.find_next_sibling('dd')
                # If a <dd> tag exists, return its text
                if next_dd:
                    return next_dd.get_text(strip=True)

        # If the loop completes without finding the date, return None
        return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def find_div_publish_date(soup: BeautifulSoup) -> str | None:
    """
    Finds publication dates from a BeautifulSoup object and ensures they are consistent.

    This function finds all <div> tags with the class 'publish-date', extracts
    the date from each, and formats it to 'YYYY-MM-DD'. It returns the date
    only if one or more dates are found and they are all identical.

    Args:
        soup: A BeautifulSoup object containing the parsed HTML.

    Returns:
        The single, consistent formatted date string (e.g., '2021-08-31'),
        or None if no dates are found, they cannot be parsed, or they
        are not all the same.
    """
    # Find all <div> elements with the class "publish-date"
    date_divs = soup.find_all('div', class_='publish-date')

    if not date_divs:
        return None

    found_dates = []
    for div in date_divs:
        try:
            # Get the text from the <p> tag inside the div
            date_string = div.p.get_text(strip=True)

            # Remove the "Published " prefix to isolate the date
            clean_date_string = date_string.replace('Published ', '')

            # Parse the date string and format it to 'YYYY-MM-DD'
            formatted_date = to_ymd(clean_date_string)
            found_dates.append(formatted_date)
        except Exception as e:
            continue

    # If no valid dates were parsed, return None
    if not found_dates:
        return None

    # Use a set to find the number of unique dates.
    # If there is exactly 1 unique date, all found dates were the same.
    unique_dates = set(found_dates)
    if len(unique_dates) == 1:
        return unique_dates.pop()  # Return the single, consistent date
    else:
        print(f"Found multiple different dates: {found_dates}")

    # If there are multiple different dates, return None
    return None

def get_first_published_date_from_script(soup):
    """
    Extracts the 'dateFirstPublished' value by parsing the JSON object
    within a script tag from a BeautifulSoup object.

    This function searches a pre-parsed BeautifulSoup object for a specific
    script tag containing the 'digitalData' variable. It then uses a regular
    expression to extract the JSON object and parses it to retrieve the date.

    Requires the following package to be installed:
    pip install beautifulsoup4

    Args:
        soup (BeautifulSoup): A BeautifulSoup object of the parsed HTML page.

    Returns:
        str: The first publication date in "YYYY-MM-DD" format, or None if not found.
    """
    try:
        # 1. Find the script tag that contains the 'digitalData' variable definition
        script_tag = soup.find('script', string=re.compile(r'var\s+digitalData\s*='))

        if not script_tag:
            # print("Script tag with 'digitalData' not found.")
            return None

        javascript_code = script_tag.string

        # 2. Use a regular expression to extract the JSON object string.
        # The re.DOTALL flag allows '.' to match newlines.
        match = re.search(r'var\s+digitalData\s*=\s*({.*?});', javascript_code, re.DOTALL)

        if not match:
            # print("Could not extract digitalData JSON object from script.")
            return None

        json_string = match.group(1)

        # 3. Parse the extracted string as JSON
        data = json.loads(json_string)

        # 4. Access the nested date value from the resulting dictionary
        date_published = data.get('page', {}).get('dateFirstPublished')

        if not date_published:
            # print("'dateFirstPublished' key not found in the page object.")
            return None

        return date_published

    except json.JSONDecodeError:
        print("Failed to parse the extracted JavaScript object as JSON.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def find_date_ld_json(soup):
    """
    Finds all ld+json script tags, extracts all "datePublished" values,
    and returns the chronologically earliest date.

    Args:
        soup (BeautifulSoup): A BeautifulSoup object of the parsed page.

    Returns:
        str: The earliest publication date string, or None if no dates are found.
    """
    found_dates = [] # 1. Initialize a list to store all dates

    # 2. Use find_all() to get every ld+json script tag
    script_tags = soup.find_all('script', {'type': 'application/ld+json'})

    for script_tag in script_tags:
        if script_tag and script_tag.string:
            try:
                json_data = json.loads(script_tag.string)
                items_to_check = []

                if isinstance(json_data, dict):
                    items_to_check = json_data.get('@graph', [json_data])
                elif isinstance(json_data, list):
                    items_to_check = json_data

                # 3. Collect all "datePublished" values
                for item in items_to_check:
                    if isinstance(item, dict) and 'datePublished' in item:
                        # Add every found date to our list
                        found_dates.append(item.get('datePublished'))
                    elif isinstance(item, dict) and 'dateCreated' in item:
                        found_dates.append(item.get('dateCreated'))

            except json.JSONDecodeError:
                # Ignore any tags with malformed JSON
                continue

    # 4. Return the earliest date from the list, or None if the list is empty
    if found_dates:
        return min(found_dates)
    else:
        return None

def find_biorxiv_date(html_text):
    """
    Extracts the posting date from a raw HTML string.

    Args:
        html_text (str): The raw HTML content of the bioRxiv page.

    Returns:
        str: The extracted posting date as a string (e.g., "March 04, 2016"),
             or None if the date cannot be found or an error occurs.
    """
    try:
        # Define the exact search string we are looking for in the raw HTML
        search_string = "Posted&nbsp;"

        # Find the position of our search string
        start_index = html_text.find(search_string)

        if start_index != -1:
            # If the string is found, calculate the start of the actual date text
            date_start = start_index + len(search_string)

            # Find the end of the date, which is the next period character
            end_index = html_text.find('.', date_start)

            if end_index != -1:
                # Extract the date substring and remove any leading/trailing whitespace
                date_str = html_text[date_start:end_index].strip()
                return date_str

        # If the search string or the trailing period isn't found, return None
        return None

    except Exception as e:
        # Handle any potential errors during string manipulation
        print(f"An error occurred during parsing: {e}")
        return None

def find_date_from_date_span(soup: BeautifulSoup) -> str | None:
    """
    Finds and returns a unique publication date from a parsed BeautifulSoup object.

    This function finds all <span> tags with itemprop="datePublished", converts
    each date to 'YYYY-MM-DD' format, and collects the unique dates. If exactly
    one unique date is found, it is returned. Otherwise, None is returned.

    Args:
        soup: A BeautifulSoup object representing the parsed HTML of a webpage.

    Returns:
        A string containing the unique date in 'YYYY-MM-DD' format if found,
        otherwise None.
    """
    try:
        # TODO: ADD dc:date
        # Find all <span> tags that might contain a publication date.
        date_spans = soup.find_all('span', attrs={'itemprop': 'datePublished'})

        if not date_spans:
            # Return None immediately if no matching tags are found.
            return None

        unique_dates = set()
        for span in date_spans:
            date_str = span.get_text(strip=True)
            try:
                unique_dates.add(to_ymd(date_str))
            except ValueError:
                # Ignore any strings that don't match the expected date format.
                print(f"Warning: Could not parse date string: '{date_str}'")
                continue

        # Check the number of unique dates found.
        if len(unique_dates) == 1:
            # If there is exactly one unique date, return it.
            return unique_dates.pop()

        # If there are 0 or more than 1 unique dates, return None.
        return None

    except Exception as e:
        # Basic error handling in case of unexpected issues with the soup object.
        print(f"An error occurred while processing the HTML: {e}")
        return None

def find_citation_publication_date(soup: BeautifulSoup) -> str | None:
    low_priority_keys = [
        "citation_publication_date","citation_date","prism.publicationDate"
    ]
    low_priority_dates = set()
    for key in low_priority_keys:
        # Note: This loop only checks for the 'name' attribute as in the original code.
        # Some keys (e.g., 'og:published_time') are typically found with the 'property' attribute.
        tag = soup.find('meta', attrs={'name': key})
        if tag and (date_str := tag.get('content')):
            try:
                # Parse and normalize the date, then add to the set.
                parsed_date = to_ymd(date_str)
                low_priority_dates.add(parsed_date)
            except parser.ParserError:
                # Ignore tags with unparseable date strings.
                continue

    # If any dates were found, assert they are all the same before returning.
    if low_priority_dates:
        return min(low_priority_dates)

    return None # Return None if no reliable date was found

def find_youtube_date(soup: BeautifulSoup) -> str | None:
    """
    Parses a YouTube watch page's BeautifulSoup object to find the publication date.

    Args:
        soup: A BeautifulSoup object of a YouTube /watch page.

    Returns:
        The publication date as a string (e.g., "Nov 17, 2017"), or None if not found.
    """
    try:
        # Find all script tags
        scripts = soup.find_all('script')

        # Target the script containing 'ytInitialData'
        data_script = None
        for script in scripts:
            # Ensure the script has content before searching
            if script.string and 'ytInitialData' in script.string:
                data_script = script.string
                break

        if not data_script:
            print("Error: Could not find the ytInitialData script tag.")
            return None

        # Use a regular expression to find the JSON object assigned to ytInitialData
        # This is more robust than simple string splitting.
        match = re.search(r"var ytInitialData = ({.*?});", data_script)
        if not match:
            print("Error: Could not extract JSON from the script tag.")
            return None

        # The actual JSON data is in the first capture group
        json_data_str = match.group(1)

        # Parse the JSON string into a Python dictionary
        data = json.loads(json_data_str)

        # Navigate through the nested dictionary to find the date
        # The path can vary, so we check for keys safely
        primary_info = data.get('contents', {})\
                           .get('twoColumnWatchNextResults', {})\
                           .get('results', {})\
                           .get('results', {})\
                           .get('contents', [{}])[0]\
                           .get('videoPrimaryInfoRenderer', {})

        if primary_info:
            date_text = primary_info.get('dateText', {}).get('simpleText')
            if date_text:
                return date_text

        print("Error: Could not find the date in the expected JSON structure.")
        return None

    except (json.JSONDecodeError, IndexError, AttributeError, TypeError) as e:
        print(f"An error occurred while parsing: {e}")
        return None




def find_date_html(url2raw):
    url2date_found = {} # Dictionary to store successfully found dates
    url_ld_json = []
    url_meta_tags = []
    # url_time_tags = []
    url_visible_tags = []
    ld_meta_tags = []
    # url_dd_tags = []
    # url_div_tags = []
    # url_script_tags = []
    success_urls = []
    urls_with_no_date = [] # List for URLs where no date could be found

    # for url in tqdm(url2raw, desc="Processing URLs"):
    for url in tqdm(url2raw, desc="Processing URLs"):
        date_published_processed = None
        date_published_processed_ld_json = None
        date_published_processed_meta_tags = None
        curr_html = url2raw.get(url, "")
        if not curr_html:
            continue
        soup = BeautifulSoup(curr_html, 'html.parser')
        if 'Failed with ZenRows' in url2raw[url]:
            continue
        if 'trialbulletin' in url:
            continue
        if 'ijgc.bmj' in url:
            continue
        if 'biorxiv' in url or 'medrxiv' in url:
            raw_date = find_biorxiv_date(url2raw[url])
            if raw_date:
                date_published_processed = clean_date(raw_date)
                if date_published_processed: success_urls.append(url)

        if not date_published_processed and "youtube" in url:
            raw_date = find_youtube_date(soup)
            if raw_date:
                date_published_processed = clean_date(raw_date)
            if date_published_processed: success_urls.append(url)



        # if 'biorxiv' in url or 'medrxiv' in url:
        #     # TODO: query from database, use title or doi
        #     continue

        if not date_published_processed:
            raw_date = find_date_ld_json(soup)
            if raw_date:
                date_published_processed_ld_json = clean_date(raw_date)
            if date_published_processed_ld_json: success_urls.append(url)
            if date_published_processed_ld_json: url_ld_json.append(url)

            raw_date2 = find_date_in_meta_tags(soup)
            if raw_date2:
                if raw_date2 == "more than one publish date found":
                    print("More than one publish date found in meta tags for",url)
                else:
                    date_published_processed_meta_tags = clean_date(raw_date2)
            if date_published_processed_meta_tags: success_urls.append(url)
            if date_published_processed_meta_tags: url_meta_tags.append(url)
            if date_published_processed_ld_json and date_published_processed_meta_tags:
                date_published_processed = min(date_published_processed_ld_json, date_published_processed_meta_tags)
            elif date_published_processed_ld_json:
                date_published_processed = date_published_processed_ld_json
            elif date_published_processed_meta_tags:
                date_published_processed = date_published_processed_meta_tags
            if date_published_processed: ld_meta_tags.append(url)

        if not date_published_processed:
            raw_date = find_date_in_publish_tags(soup)
            if raw_date:
                date_published_processed = clean_date(raw_date)
            if date_published_processed: success_urls.append(url)

        if not date_published_processed and "sciencedaily" in url:
            raw_date = get_dd_date(soup)
            if raw_date:
                date_published_processed = clean_date(raw_date)
            if date_published_processed: success_urls.append(url)

        if not date_published_processed:
            raw_date = find_citation_publication_date(soup)
            if raw_date:
                date_published_processed = clean_date(raw_date)
            if date_published_processed: success_urls.append(url)

        if not date_published_processed:
            raw_date = find_div_publish_date(soup)
            if raw_date:
                date_published_processed = clean_date(raw_date)
            if date_published_processed: success_urls.append(url)

        if not date_published_processed:
            raw_date = get_first_published_date_from_script(soup)
            if raw_date:
                date_published_processed = clean_date(raw_date)
            if date_published_processed: success_urls.append(url)

        if not date_published_processed:
            raw_date = find_date_from_date_span(soup)
            if raw_date:
                date_published_processed = clean_date(raw_date)
            if date_published_processed: success_urls.append(url)

        if not date_published_processed:
            raw_date = find_earliest_date(soup)
            if raw_date:
                date_published_processed = clean_date(raw_date)
            if date_published_processed: success_urls.append(url)

        if not date_published_processed:
            raw_date = find_date_in_visible_text(soup)
            if raw_date:
                date_published_processed = clean_date(raw_date)
            if date_published_processed: success_urls.append(url)

        if date_published_processed:
            url2date_found[url] = to_ymd(date_published_processed)
        else:
            urls_with_no_date.append(url)

    return url2date_found, urls_with_no_date

# def find_date_html_multithreaded(url2raw, max_workers=16):
    # """
    # Threaded extractor: returns {url: 'YYYY-MM-DD' | None} for every key in url2raw.
    # Assumes helper functions exist: clean_date, to_ymd, and all find_* helpers.
    # """
    # results = {}

    # def _process(url, curr_html):
    #     try:
    #         if not curr_html:
    #             return url, None
    #         if 'Failed with ZenRows' in curr_html:
    #             return url, None

    #         # Domain skips (still return None for the URL)
    #         if 'trialbulletin' in url or 'ijgc.bmj' in url:
    #             return url, None

    #         date_published_processed = None
    #         date_published_processed_ld_json = None
    #         date_published_processed_meta_tags = None

    #         # Special cases that don't need soup first
    #         if ('biorxiv' in url) or ('medrxiv' in url):
    #             raw_date = find_biorxiv_date(curr_html)
    #             if raw_date:
    #                 date_published_processed = clean_date(raw_date)

    #         soup = None
    #         def _soup():
    #             nonlocal soup
    #             if soup is None:
    #                 soup = BeautifulSoup(curr_html, 'html.parser')
    #             return soup

    #         if not date_published_processed and "youtube" in url:
    #             raw_date = find_youtube_date(_soup())
    #             if raw_date:
    #                 date_published_processed = clean_date(raw_date)

    #         if not date_published_processed:
    #             raw_date = find_date_ld_json(_soup())
    #             if raw_date:
    #                 date_published_processed_ld_json = clean_date(raw_date)

    #             raw_date2 = find_date_in_meta_tags(_soup())
    #             if raw_date2 and raw_date2 != "more than one publish date found":
    #                 date_published_processed_meta_tags = clean_date(raw_date2)

    #             if date_published_processed_ld_json and date_published_processed_meta_tags:
    #                 date_published_processed = min(
    #                     date_published_processed_ld_json,
    #                     date_published_processed_meta_tags
    #                 )
    #             elif date_published_processed_ld_json:
    #                 date_published_processed = date_published_processed_ld_json
    #             elif date_published_processed_meta_tags:
    #                 date_published_processed = date_published_processed_meta_tags

    #         if not date_published_processed:
    #             raw_date = find_date_in_publish_tags(_soup())
    #             if raw_date:
    #                 date_published_processed = clean_date(raw_date)

    #         if not date_published_processed and "sciencedaily" in url:
    #             raw_date = get_dd_date(_soup())
    #             if raw_date:
    #                 date_published_processed = clean_date(raw_date)

    #         if not date_published_processed:
    #             raw_date = find_citation_publication_date(_soup())
    #             if raw_date:
    #                 date_published_processed = clean_date(raw_date)

    #         if not date_published_processed:
    #             raw_date = find_div_publish_date(_soup())
    #             if raw_date:
    #                 date_published_processed = clean_date(raw_date)

    #         if not date_published_processed:
    #             raw_date = get_first_published_date_from_script(_soup())
    #             if raw_date:
    #                 date_published_processed = clean_date(raw_date)

    #         if not date_published_processed:
    #             raw_date = find_date_from_date_span(_soup())
    #             if raw_date:
    #                 date_published_processed = clean_date(raw_date)

    #         if not date_published_processed:
    #             raw_date = find_earliest_date(_soup())
    #             if raw_date:
    #                 date_published_processed = clean_date(raw_date)

    #         if not date_published_processed:
    #             raw_date = find_date_in_visible_text(_soup())
    #             if raw_date:
    #                 date_published_processed = clean_date(raw_date)

    #         return url, (to_ymd(date_published_processed) if date_published_processed else None)

    #     except Exception:
    #         # Never let one bad page kill the batch; just return None for that URL.
    #         return url, None

    # if max_workers is None:
    #     max_workers = min(32, max(1, len(url2raw)))

    # with ThreadPoolExecutor(max_workers=max_workers) as ex:
    #     futures = [ex.submit(_process, url, html) for url, html in url2raw.items()]
    #     for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting dates", unit="url"):
    #         url, ymd = fut.result()
    #         results[url] = ymd

    # return results

# Top-level worker so it can be pickled by ProcessPoolExecutor
def _find_date_worker(args):
    url, curr_html = args
    try:
        if not curr_html:
            return url, None
        if 'Failed with ZenRows' in curr_html:
            return url, None

        # Domain skips (still return None for the URL)
        if 'trialbulletin' in url or 'ijgc.bmj' in url:
            return url, None

        date_published_processed = None
        date_published_processed_ld_json = None
        date_published_processed_meta_tags = None

        # Special cases that don't need soup first
        if ('biorxiv' in url) or ('medrxiv' in url):
            raw_date = find_biorxiv_date(curr_html)
            if raw_date:
                date_published_processed = clean_date(raw_date)

        soup = None
        def _soup():
            nonlocal soup
            if soup is None:
                soup = BeautifulSoup(curr_html, 'html.parser')
            return soup

        if not date_published_processed and "youtube" in url:
            raw_date = find_youtube_date(_soup())
            if raw_date:
                date_published_processed = clean_date(raw_date)

        if not date_published_processed:
            raw_date = find_date_ld_json(_soup())
            if raw_date:
                date_published_processed_ld_json = clean_date(raw_date)

            raw_date2 = find_date_in_meta_tags(_soup())
            if raw_date2 and raw_date2 != "more than one publish date found":
                date_published_processed_meta_tags = clean_date(raw_date2)

            if date_published_processed_ld_json and date_published_processed_meta_tags:
                date_published_processed = min(
                    date_published_processed_ld_json,
                    date_published_processed_meta_tags
                )
            elif date_published_processed_ld_json:
                date_published_processed = date_published_processed_ld_json
            elif date_published_processed_meta_tags:
                date_published_processed = date_published_processed_meta_tags

        if not date_published_processed:
            raw_date = find_date_in_publish_tags(_soup())
            if raw_date:
                date_published_processed = clean_date(raw_date)

        if not date_published_processed and "sciencedaily" in url:
            raw_date = get_dd_date(_soup())
            if raw_date:
                date_published_processed = clean_date(raw_date)

        if not date_published_processed:
            raw_date = find_citation_publication_date(_soup())
            if raw_date:
                date_published_processed = clean_date(raw_date)

        if not date_published_processed:
            raw_date = find_div_publish_date(_soup())
            if raw_date:
                date_published_processed = clean_date(raw_date)

        if not date_published_processed:
            raw_date = get_first_published_date_from_script(_soup())
            if raw_date:
                date_published_processed = clean_date(raw_date)

        if not date_published_processed:
            raw_date = find_date_from_date_span(_soup())
            if raw_date:
                date_published_processed = clean_date(raw_date)

        if not date_published_processed:
            raw_date = find_earliest_date(_soup())
            if raw_date:
                date_published_processed = clean_date(raw_date)

        if not date_published_processed:
            raw_date = find_date_in_visible_text(_soup())
            if raw_date:
                date_published_processed = clean_date(raw_date)

        return url, (to_ymd(date_published_processed) if date_published_processed else None)

    except Exception:
        # Never let one bad page kill the batch; just return None for that URL.
        return url, None


def find_date_html_multithreaded(url2raw, max_workers=16):
    """
    Process-pooled extractor: returns {url: 'YYYY-MM-DD' | None} for every key in url2raw.
    Assumes helper functions exist: clean_date, to_ymd, and all find_* helpers.
    """
    results = {}

    if max_workers is None:
        max_workers = min(32, max(1, len(url2raw)))

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_find_date_worker, (url, html)) for url, html in url2raw.items()]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting dates", unit="url"):
            url, ymd = fut.result()
            results[url] = ymd

    return results




# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--url2raw", type=str, default="data/jul29_url2raw.pickle")
#     parser.add_argument("--output_file", type=str, default="data/aug11_url2date_found.pickle")
#     args = parser.parse_args()

#     with open(args.url2raw,'rb') as f:
#         url2raw = pickle.load(f)

    # url2date_found = {} # Dictionary to store successfully found dates
    # url_ld_json = []
    # url_meta_tags = []
    # # url_time_tags = []
    # url_visible_tags = []
    # ld_meta_tags = []
    # # url_dd_tags = []
    # # url_div_tags = []
    # # url_script_tags = []
    # success_urls = []
    # urls_with_no_date = [] # List for URLs where no date could be found

    # for url in tqdm(url2raw, desc="Processing URLs"):
    #     date_published_processed = None
    #     date_published_processed_ld_json = None
    #     date_published_processed_meta_tags = None
    #     curr_html = url2raw.get(url, "")
    #     if not curr_html:
    #         continue
    #     soup = BeautifulSoup(curr_html, 'html.parser')
    #     if 'Failed with ZenRows' in url2raw[url]:
    #         continue
    #     if 'trialbulletin' in url:
    #         continue
    #     if 'ijgc.bmj' in url:
    #         continue
    #     if 'biorxiv' in url or 'medrxiv' in url:
    #         raw_date = find_biorxiv_date(url2raw[url])
    #         if raw_date:
    #             date_published_processed = clean_date(raw_date)
    #             if date_published_processed: success_urls.append(url)

    #     if not date_published_processed and "youtube" in url:
    #         raw_date = find_youtube_date(soup)
    #         if raw_date:
    #             date_published_processed = clean_date(raw_date)
    #         if date_published_processed: success_urls.append(url)



    #     # if 'biorxiv' in url or 'medrxiv' in url:
    #     #     # TODO: query from database, use title or doi
    #     #     continue

    #     if not date_published_processed:
    #         raw_date = find_date_ld_json(soup)
    #         if raw_date:
    #             date_published_processed_ld_json = clean_date(raw_date)
    #         if date_published_processed_ld_json: success_urls.append(url)
    #         if date_published_processed_ld_json: url_ld_json.append(url)

    #         raw_date2 = find_date_in_meta_tags(soup)
    #         if raw_date2:
    #             if raw_date2 == "more than one publish date found":
    #                 print("More than one publish date found in meta tags for",url)
    #             else:
    #                 date_published_processed_meta_tags = clean_date(raw_date2)
    #         if date_published_processed_meta_tags: success_urls.append(url)
    #         if date_published_processed_meta_tags: url_meta_tags.append(url)
    #         if date_published_processed_ld_json and date_published_processed_meta_tags:
    #             date_published_processed = min(date_published_processed_ld_json, date_published_processed_meta_tags)
    #         elif date_published_processed_ld_json:
    #             date_published_processed = date_published_processed_ld_json
    #         elif date_published_processed_meta_tags:
    #             date_published_processed = date_published_processed_meta_tags
    #         if date_published_processed: ld_meta_tags.append(url)

    #     if not date_published_processed:
    #         raw_date = find_date_in_publish_tags(soup)
    #         if raw_date:
    #             date_published_processed = clean_date(raw_date)
    #         if date_published_processed: success_urls.append(url)

    #     if not date_published_processed and "sciencedaily" in url:
    #         raw_date = get_dd_date(soup)
    #         if raw_date:
    #             date_published_processed = clean_date(raw_date)
    #         if date_published_processed: success_urls.append(url)

    #     if not date_published_processed:
    #         raw_date = find_citation_publication_date(soup)
    #         if raw_date:
    #             date_published_processed = clean_date(raw_date)
    #         if date_published_processed: success_urls.append(url)

    #     if not date_published_processed:
    #         raw_date = find_div_publish_date(soup)
    #         if raw_date:
    #             date_published_processed = clean_date(raw_date)
    #         if date_published_processed: success_urls.append(url)

    #     if not date_published_processed:
    #         raw_date = get_first_published_date_from_script(soup)
    #         if raw_date:
    #             date_published_processed = clean_date(raw_date)
    #         if date_published_processed: success_urls.append(url)

    #     if not date_published_processed:
    #         raw_date = find_date_from_date_span(soup)
    #         if raw_date:
    #             date_published_processed = clean_date(raw_date)
    #         if date_published_processed: success_urls.append(url)

    #     if not date_published_processed:
    #         raw_date = find_earliest_date(soup)
    #         if raw_date:
    #             date_published_processed = clean_date(raw_date)
    #         if date_published_processed: success_urls.append(url)

    #     if not date_published_processed:
    #         raw_date = find_date_in_visible_text(soup)
    #         if raw_date:
    #             date_published_processed = clean_date(raw_date)
    #         if date_published_processed: success_urls.append(url)

    #     if date_published_processed:
    #         url2date_found[url] = date_published_processed
    #     else:
    #         urls_with_no_date.append(url)

    # with open(args.output_file, "wb") as f:
    #     pickle.dump({"url2date_found": url2date_found, "urls_with_no_date": urls_with_no_date}, f)
def normalize_to_url_html_map(obj):
    """
    Accepts either:
      1) {'scraped_results_batch': { url: {content: str, method: str}, ... }}
         -> keeps content IFF method != 'failed'
      2) { url: html_str }  (legacy) OR { url: {content, method?} } mixed
    Returns: dict[url] = html_str  (only URLs we should process)
    """
    url_html = {}

    if isinstance(obj, dict) and 'scraped_results_batch' in obj:
        batch = obj.get('scraped_results_batch', {})
        if isinstance(batch, dict):
            for url, rec in batch.items():
                if not isinstance(rec, dict):
                    continue
                method = (rec.get('method') or '').lower()
                if method == 'failed':
                    continue
                content = rec.get('content')
                if content:
                    url_html[url] = content
        return url_html

    return url_html

if __name__ == "__main__":
    # TODO: check the urls with no date found and make sure they don't have pub date
    import argparse
    cli_args = argparse.ArgumentParser()
    cli_args.add_argument("--input_file", type=str)
    cli_args.add_argument("--output_file", type=str)
    args = cli_args.parse_args()

    with open(args.input_file, 'rb') as f:
        data = pickle.load(f)
    url2raw = normalize_to_url_html_map(data)

    print(f"Found {len(url2raw)} URLs to process")

    url2date_found = find_date_html_multithreaded(url2raw)
    # url2date_found = find_date_html(url2raw)

    with open(args.output_file, "wb") as f:
        pickle.dump(url2date_found, f)
