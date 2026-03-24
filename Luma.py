#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests
import extruct
import re
import json
from datetime import datetime
from urllib.parse import urlparse
from pathlib import Path

import streamlit as st
import plotly.express as px
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(
    page_title="Luma: RDA Metadata Quality Auditor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CONFIGURATION
# =========================================================
DEFAULT_INPUT_PATH = r"C:\1 ARDC\Metadata Curation\MetadataList.xlsx"
DEFAULT_OUTPUT_PATH = r"C:\1 ARDC\Metadata Curation\RDA_Metadata_Audit_Gold.xlsx"

USER_AGENT = "ARDC-RDA-Metadata-Auditor/11.0"
TIMEOUT = 20

# =========================================================
# STYLING
# =========================================================
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1rem;
            max-width: 1500px;
        }
        .metric-card {
            background: #ffffff;
            border: 1px solid #e9edf3;
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04);
        }
        .small-label {
            color: #5b6577;
            font-size: 0.88rem;
            margin-bottom: 0.15rem;
        }
        .big-number {
            color: #111827;
            font-size: 1.9rem;
            font-weight: 700;
            line-height: 1.1;
        }
        .subtle {
            color: #6b7280;
            font-size: 0.92rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e9edf3;
            padding: 14px 16px;
            border-radius: 18px;
            box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04);
        }
        .title-wrap {
            padding: 0.2rem 0 0.8rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HTTP SESSION
# =========================================================
def build_session():
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session

SESSION = build_session()

# =========================================================
# GENERIC HELPERS
# =========================================================
def safe_str(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


def normalise_to_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def dedupe_keep_order(values):
    out = []
    seen = set()
    for v in values:
        s = str(v).strip()
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return out


def flatten_jsonld_nodes(jsonld_items):
    """
    Recursively flatten JSON-LD objects, including @graph.
    """
    flattened = []

    def walk(node):
        if isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            flattened.append(node)
            if "@graph" in node:
                walk(node["@graph"])

    walk(jsonld_items)
    return flattened


def get_types(node):
    t = node.get("@type", [])
    if isinstance(t, list):
        return [str(x) for x in t]
    if t:
        return [str(t)]
    return []


def infer_record_id_from_url(url):
    path = urlparse(url).path.strip("/")
    m = re.search(r"/(\d+)$", "/" + path)
    return m.group(1) if m else ""

# =========================================================
# GOLD STANDARD
# =========================================================
def infer_functional_profile(record_type, distributions, variable_count, licence_present):
    """
    Override semantic type using metadata behaviour.
    """
    if record_type == "Dataset":
        if variable_count == 0 and len(distributions) == 0:
            return "Collection-like Dataset"
        if variable_count == 0:
            return "Thin Dataset"
        if len(distributions) > 0:
            return "Operational Dataset"
        return "Dataset"

    return record_type

# =========================================================
# JSON-LD GRAPH HELPERS
# =========================================================
def build_id_index(nodes):
    """
    Build index of @id -> node for graph enrichment.
    """
    index = {}
    for node in nodes:
        if isinstance(node, dict):
            node_id = node.get("@id")
            if node_id:
                index[str(node_id)] = node
    return index


def resolve_reference(value, id_index):
    """
    Resolve {"@id": "..."} references when possible.
    """
    if isinstance(value, dict) and set(value.keys()) == {"@id"}:
        return id_index.get(str(value["@id"]), value)
    return value


def resolve_references_in_list(values, id_index):
    return [resolve_reference(v, id_index) for v in normalise_to_list(values)]

# =========================================================
# RECORD TYPE + CANDIDATE DETECTION
# =========================================================
def classify_record_type_from_types(types_lower):
    if "dataset" in types_lower:
        return "Dataset"
    if "collection" in types_lower:
        return "Collection"
    if "datafeed" in types_lower:
        return "DataFeed"
    if "datacatalog" in types_lower:
        return "DataCatalog"
    if "creativework" in types_lower:
        return "CreativeWork"
    return "Other"


def looks_like_record_object(node):
    """
    RDA-aware record detection.
    Broader than Dataset, but still requires substantial metadata substance.
    """
    types = {t.lower() for t in get_types(node)}
    recordish_types = {
        "dataset", "collection", "creativework", "datafeed", "datacatalog", "thing"
    }
    keys = {k.lower() for k in node.keys()}

    informative_fields = {
        "name", "headline", "description", "identifier", "@id",
        "license", "creator", "author", "distribution",
        "spatialcoverage", "temporalcoverage", "variablemeasured",
        "keywords", "includedindatacatalog"
    }

    if types & recordish_types and len(keys & informative_fields) >= 3:
        return True

    score = 0
    if "name" in node or "headline" in node:
        score += 2
    if "description" in node:
        score += 2
    if "identifier" in node or "@id" in node:
        score += 2
    if "creator" in node or "author" in node:
        score += 1
    if "license" in node:
        score += 1
    if "distribution" in node:
        score += 2
    if "spatialCoverage" in node:
        score += 1
    if "temporalCoverage" in node:
        score += 1
    if "variableMeasured" in node:
        score += 2

    return score >= 5


def candidate_strength(node):
    """
    Rank record candidates.
    """
    types = {t.lower() for t in get_types(node)}
    score = 0

    if "dataset" in types:
        score += 100
    elif "collection" in types:
        score += 60
    elif "creativework" in types:
        score += 40
    elif "datafeed" in types:
        score += 50
    elif "datacatalog" in types:
        score += 30
    else:
        score += 10

    for field, weight in [
        ("name", 10),
        ("description", 10),
        ("identifier", 12),
        ("@id", 8),
        ("license", 10),
        ("creator", 10),
        ("author", 8),
        ("distribution", 18),
        ("variableMeasured", 18),
        ("spatialCoverage", 8),
        ("temporalCoverage", 8),
        ("keywords", 5),
        ("includedInDataCatalog", 5),
    ]:
        if field in node and node.get(field):
            score += weight

    return score


def pick_best_record_candidate(nodes):
    candidates = [n for n in nodes if looks_like_record_object(n)]
    if not candidates:
        return {}
    return max(candidates, key=candidate_strength)

# =========================================================
# ENRICHMENT
# =========================================================
def enrich_candidate(candidate, id_index):
    """
    Resolve simple linked references for key fields.
    Keeps the original node, but expands common reference-only structures.
    """
    if not candidate:
        return {}

    enriched = dict(candidate)

    for field in ["identifier", "creator", "author", "license", "distribution", "spatialCoverage"]:
        if field in enriched:
            enriched[field] = resolve_references_in_list(enriched[field], id_index)

    return enriched

# =========================================================
# EXTRACTION HELPERS
# =========================================================
def extract_identifiers(node):
    values = []

    for key in ["identifier", "@id"]:
        val = node.get(key)
        if val is None:
            continue

        for item in normalise_to_list(val):
            if isinstance(item, dict):
                for k in ["value", "@id", "url", "identifier", "name"]:
                    if item.get(k):
                        values.append(str(item[k]))
            else:
                values.append(str(item))

    return dedupe_keep_order(values)


def has_doi(identifiers):
    doi_patterns = [
        r"https?://(dx\.)?doi\.org/10\.\S+",
        r"\b10\.\d{4,9}/[^\s\"<>]+\b"
    ]
    combined = " | ".join(identifiers)
    return any(re.search(pattern, combined, re.I) for pattern in doi_patterns)


def extract_doi_from_page_metadata(html):
    """
    Detect DOI outside JSON-LD blocks only.
    """
    cleaned_html = re.sub(
        r'<script[^>]*type="application/ld\+json"[^>]*>.*?</script>',
        '',
        html,
        flags=re.DOTALL | re.IGNORECASE
    )

    doi_patterns = [
        r'https?://(?:dx\.)?doi\.org/10\.\d{4,9}/[^\s"<>]+',
        r'\b10\.\d{4,9}/[^\s"<>]+\b'
    ]

    for pattern in doi_patterns:
        match = re.search(pattern, cleaned_html, re.I)
        if match:
            return match.group(0).strip()

    return ""


def extract_people_field(node):
    people = []
    for field in ["creator", "author"]:
        for item in normalise_to_list(node.get(field)):
            people.append(item)
    return people


def extract_person_names(people):
    names = []
    for p in people:
        if isinstance(p, dict):
            if p.get("name"):
                names.append(str(p["name"]))
            elif p.get("creator") and isinstance(p["creator"], dict) and p["creator"].get("name"):
                names.append(str(p["creator"]["name"]))
        elif isinstance(p, str):
            names.append(p)
    return dedupe_keep_order(names)


def has_orcid(people):
    text = safe_str(people).lower()
    return "orcid.org/" in text


def has_role_based_semantics(people):
    text = safe_str(people)
    return ("roleName" in text) or ('"@type": "Role"' in text) or ('"Role"' in text)


def extract_licenses(node):
    out = []
    for lic in normalise_to_list(node.get("license")):
        if isinstance(lic, dict):
            for k in ["@id", "url", "name", "value"]:
                if lic.get(k):
                    out.append(str(lic[k]))
        else:
            out.append(str(lic))
    return dedupe_keep_order(out)


def extract_distributions(node):
    distributions = []
    for d in normalise_to_list(node.get("distribution")):
        if not isinstance(d, dict):
            continue
        item = {
            "name": d.get("name", ""),
            "contentUrl": d.get("contentUrl", ""),
            "encodingFormat": d.get("encodingFormat", ""),
            "url": d.get("url", ""),
            "@type": safe_str(d.get("@type", ""))
        }
        distributions.append(item)
    return distributions

# =========================================================
# SCORING HELPERS
# =========================================================
def classify_distribution_strength(distributions):
    """
    Score machine actionability conservatively.
    """
    if not distributions:
        return 0, "Low", [], []

    score = 10
    direct_links = []
    formats = []

    strong_extensions = [
        ".csv", ".json", ".jsonl", ".ndjson", ".xml", ".rdf", ".ttl",
        ".zip", ".xlsx", ".xls", ".txt", ".tsv", ".parquet", ".feather",
        ".nc", ".h5", ".hdf5", ".geojson", ".gpkg", ".shp", ".tif", ".tiff"
    ]

    strong_mime_keywords = [
        "csv", "json", "xml", "tab-separated", "excel", "netcdf",
        "parquet", "geojson", "rdf", "zip", "geotiff", "tiff"
    ]

    seen_strong = 0
    seen_content = 0

    for d in distributions:
        content_url = safe_str(d.get("contentUrl")).lower().strip()
        fallback_url = safe_str(d.get("url")).lower().strip()
        fmt = safe_str(d.get("encodingFormat")).lower().strip()

        if fmt:
            formats.append(fmt)

        target = content_url or fallback_url
        if target:
            direct_links.append(target)

        extension_hit = any(ext in target for ext in strong_extensions)
        mime_hit = any(keyword in fmt for keyword in strong_mime_keywords)

        if content_url:
            seen_content += 1
        if extension_hit or mime_hit:
            seen_strong += 1

    if distributions:
        score += 15
    if seen_content > 0:
        score += 25
    if seen_strong > 0:
        score += 25
    if seen_content > 1:
        score += 10
    if seen_strong > 1:
        score += 10

    score = min(score, 100)

    if score >= 65:
        label = "High"
    elif score >= 40:
        label = "Medium"
    else:
        label = "Low"

    return score, label, dedupe_keep_order(direct_links), dedupe_keep_order(formats)


def evaluate_temporal_quality(node):
    temp = safe_str(node.get("temporalCoverage")).strip()
    if not temp:
        return 0, "Missing"

    iso_patterns = [
        r"\b\d{4}-\d{2}-\d{2}/\d{4}-\d{2}-\d{2}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{4}/\d{4}\b",
        r"\b\d{4}-\d{2}\b"
    ]

    if any(re.search(p, temp) for p in iso_patterns):
        return 20, "Standard / Structured"
    if re.search(r"\b\d{4}\b", temp):
        return 10, "Partially Structured"
    return 5, "Free Text"


def evaluate_spatial_quality(node):
    spatial = node.get("spatialCoverage")
    if not spatial:
        return 0, "Missing"

    text = safe_str(spatial).lower()
    if any(k in text for k in ["geo", "latitude", "longitude", "polygon", "place"]):
        return 15, "Structured"
    return 8, "Present"


def evaluate_variables(node):
    variables = node.get("variableMeasured")
    if not variables:
        return 0, "None", 0

    if isinstance(variables, list):
        count = len(variables)
        if count >= 10:
            return 45, f"{count} variables", count
        if count >= 3:
            return 35, f"{count} variables", count
        return 25, f"{count} variables", count

    return 20, "Defined", 1


def classify_record_profile(record_type):
    """
    Used for interpretation, not detection.
    """
    if record_type == "Dataset":
        return "Dataset-oriented"
    if record_type in {"Collection", "CreativeWork"}:
        return "Collection-oriented"
    if record_type in {"DataFeed", "DataCatalog"}:
        return "Service / feed-oriented"
    return "Other"

# =========================================================
# SCORING MODELS
# =========================================================
def score_machine_readable(jsonld_items, all_nodes, best):
    score = 0
    if jsonld_items:
        score += 40
    if all_nodes:
        score += 20
    if best:
        score += 25
        types = {t.lower() for t in get_types(best)}
        if "dataset" in types:
            score += 15
    return min(score, 100)


def score_fair_proxy(identifiers, has_doi_flag, has_orcid_flag, licences):
    """
    FAIR proxy: stronger emphasis on DOI + licence, lighter on generic identifiers.
    """
    score = 0

    if identifiers:
        score += 10
    if has_doi_flag:
        score += 35
    if licences:
        score += 30
    if has_orcid_flag:
        score += 15

    if len(identifiers) >= 3:
        score += 10

    return min(score, 100)


def score_cdif_ai_ready(record_type, variable_score, temporal_score, spatial_score, role_score):
    """
    Stricter model:
    - For Dataset records, missing variables is a major limitation.
    - For Collection-like records, variables matter less, but direct AI/CDIF suitability is still capped.
    """
    score = 0
    score += variable_score
    score += temporal_score
    score += spatial_score
    score += role_score

    if record_type == "Dataset" and variable_score == 0:
        score = max(score - 30, 0)
    elif record_type in {"Collection", "CreativeWork"}:
        score = min(score, 55)

    return min(score, 100)


def score_total(machine_readable_score, fair_proxy_score, machine_actionable_score, cdif_score, record_type):
    """
    Composite uses FAIR proxy + actionable + CDIF, plus a small machine-readable contribution.
    """
    base = (
        fair_proxy_score * 0.30 +
        machine_actionable_score * 0.30 +
        cdif_score * 0.30 +
        machine_readable_score * 0.10
    )

    if record_type == "Dataset" and machine_actionable_score == 0:
        base -= 5

    return round(max(min(base, 100), 0), 2)


def classify_curation_effort(total_score):
    if total_score < 40:
        return "High"
    if total_score < 75:
        return "Medium"
    return "Low"


def classify_overall_quality(total_score):
    if total_score < 35:
        return "Low"
    if total_score < 70:
        return "Moderate"
    return "High"

# =========================================================
# RECOMMENDATIONS
# =========================================================
def build_path_to_100(report):
    recs = []

    if report["Machine_Readable_Score"] < 90:
        recs.append("Expose a stronger and more complete Schema.org JSON-LD record on the landing page.")

    if report["Has_DOI_JSONLD"] == "No":
        if report["Has_DOI_Page_Metadata"] == "Yes":
            recs.append("Expose the existing DOI inside the JSON-LD Dataset identifier field.")
        else:
            recs.append("Add a resolvable DOI in the JSON-LD Dataset identifier field where applicable.")

    if report["Has_ORCID"] == "No":
        recs.append("Include creator ORCIDs to strengthen provenance and authority.")

    if report["License_Status"] == "Missing":
        recs.append("Provide a machine-readable licence URI or licence object.")

    if report["Machine_Actionable"] != "High":
        recs.append("Add distribution entries with direct contentUrl links to downloadable files or machine endpoints.")

    if "Dataset" in report["Record_Profile"] and report["Variable_Description"] == "None":
        recs.append("Map internal variables or columns to variableMeasured.")

    if report["Temporal_Quality"] == "Missing":
        recs.append("Add structured temporalCoverage using ISO-aligned values.")
    elif report["Temporal_Quality"] == "Partially Structured":
        recs.append("Tighten temporalCoverage formatting to consistent machine-readable date ranges.")

    if report["Spatial_Data"] == "Missing":
        recs.append("Add spatialCoverage using structured place or geospatial representations.")

    if report["Structural_Type"] != "Semantic Role-Based":
        recs.append("Represent contributors using Role objects where contributor roles matter.")

    return " ".join(recs) if recs else "Strong web-exposed metadata profile."

# =========================================================
# NOTES TAB
# =========================================================
NOTES_DATA = {
    "Column Name": [
        "Machine_Readable_Score",
        "FAIR_Proxy_Score",
        "Machine_Actionable_Score",
        "CDIF_AI_Ready_Score",
        "Total_Composite_Score",
        "Record_Type",
        "Record_Profile",
        "JSONLD_Found",
        "RDFa_Present",
        "Microdata_Present",
        "Dataset_Candidate_Found",
        "Has_DOI",
        "Has_DOI_JSONLD",
        "Has_DOI_Page_Metadata",
        "Page_DOI_Value",
        "Has_ORCID",
        "License_Status",
        "Distribution_Count",
        "Direct_Link_Count",
        "Variable_Description",
        "Temporal_Quality",
        "Spatial_Data",
        "Structural_Type",
        "Path_to_100",
        "Curation_Effort",
        "Overall_Quality"
    ],
    "Metric Meaning": [
        "Strength of machine-readable JSON-LD exposure on the landing page.",
        "Proxy FAIR score based on DOI, identifiers, ORCIDs, and machine-readable licence metadata.",
        "Ability for a machine to discover direct data access points and usable formats.",
        "Proxy AI/CDIF readiness based on variables, temporal structure, spatial structure, and semantic contributor roles.",
        "Weighted composite score across machine readability, FAIR proxy, machine actionability, and AI/CDIF readiness.",
        "Primary interpreted record type from detected JSON-LD types.",
        "Broad orientation of the record for interpretation purposes.",
        "Whether JSON-LD was found on the page.",
        "Whether RDFa was detected on the page.",
        "Whether Microdata was detected on the page.",
        "Whether a strong record-like JSON-LD object was found.",
        "Backward-compatible DOI flag used in the report. For scoring, this follows DOI presence in JSON-LD.",
        "Whether DOI-like identifiers were found in the JSON-LD graph. This is the DOI signal used for scoring.",
        "Whether a DOI-like identifier was found elsewhere in page-level metadata outside the JSON-LD identifier graph.",
        "The DOI value detected from page-level metadata when present.",
        "Whether ORCID identifiers were found for creators/authors.",
        "Whether licence metadata is present.",
        "Number of distribution objects found.",
        "Number of direct URLs found in distribution metadata.",
        "Whether variableMeasured is present and how many variables appear to be defined.",
        "Quality of temporalCoverage structure.",
        "Whether spatialCoverage is present and how structured it appears to be.",
        "Whether contributors appear to use Role-based semantics.",
        "Priority remediation advice to improve landing-page metadata quality.",
        "Estimated manual curation workload based on total maturity.",
        "High-level interpretation of total composite score."
    ]
}

# =========================================================
# AUDIT ENGINE
# =========================================================
def evaluate_rda_record(url, current, total):
    report = {
        "URL": url,
        "HTTP_Status": 0,
        "Record_ID": "",
        "JSONLD_Found": "No",
        "RDFa_Present": "No",
        "Microdata_Present": "No",
        "Dataset_Candidate_Found": "No",
        "Detected_Types": "",
        "Record_Type": "Unknown",
        "Record_Profile": "Unknown",
        "Machine_Readable_Score": 0,
        "FAIR_Proxy_Score": 0,
        "Machine_Actionable_Score": 0,
        "CDIF_AI_Ready_Score": 0,
        "Total_Composite_Score": 0,
        "Overall_Quality": "Unknown",
        "Has_DOI": "No",
        "Has_DOI_JSONLD": "No",
        "Has_DOI_Page_Metadata": "No",
        "Page_DOI_Value": "",
        "Has_ORCID": "No",
        "Identifier_Count": 0,
        "Identifier_Values": "",
        "Creator_Count": 0,
        "Creator_Names": "",
        "License_Status": "Missing",
        "License_Values": "",
        "Distribution_Count": 0,
        "Direct_Link_Count": 0,
        "Distribution_Formats": "",
        "Machine_Actionable": "Low",
        "Variable_Description": "None",
        "Variable_Count": 0,
        "Spatial_Data": "Missing",
        "Temporal_Quality": "Missing",
        "Structural_Type": "Flat",
        "Best_Candidate_Score": 0,
        "Critical_Gaps": "",
        "Path_to_100": "",
        "Curation_Effort": "N/A",
        "System_Error": ""
    }

    try:
        report["Record_ID"] = infer_record_id_from_url(url)

        response = SESSION.get(url, timeout=TIMEOUT)
        report["HTTP_Status"] = response.status_code

        if response.status_code != 200:
            report["Critical_Gaps"] = f"HTTP {response.status_code}"
            return report

        extracted = extruct.extract(
            response.text,
            base_url=url,
            syntaxes=["json-ld", "microdata", "opengraph", "rdfa"]
        )

        jsonld_items = extracted.get("json-ld", [])
        all_nodes = flatten_jsonld_nodes(jsonld_items)
        id_index = build_id_index(all_nodes)

        if jsonld_items:
            report["JSONLD_Found"] = "Yes"
        if extracted.get("rdfa"):
            report["RDFa_Present"] = "Yes"
        if extracted.get("microdata"):
            report["Microdata_Present"] = "Yes"

        best = pick_best_record_candidate(all_nodes)
        report["Machine_Readable_Score"] = score_machine_readable(jsonld_items, all_nodes, best)

        if not best:
            report["Critical_Gaps"] = "No record-like JSON-LD object found."
            report["Path_to_100"] = "Expose a stronger Schema.org JSON-LD record on the page."
            return report

        best = enrich_candidate(best, id_index)

        report["Dataset_Candidate_Found"] = "Yes"
        report["Best_Candidate_Score"] = candidate_strength(best)

        detected_types = get_types(best)
        report["Detected_Types"] = ", ".join(detected_types)

        record_type = classify_record_type_from_types({t.lower() for t in detected_types})
        report["Record_Type"] = record_type

        all_identifiers = []
        for node in all_nodes:
            all_identifiers.extend(extract_identifiers(node))
        all_identifiers = dedupe_keep_order(all_identifiers)

        report["Identifier_Count"] = len(all_identifiers)
        report["Identifier_Values"] = " | ".join(all_identifiers[:10])

        doi_jsonld_flag = has_doi(all_identifiers)
        if doi_jsonld_flag:
            report["Has_DOI_JSONLD"] = "Yes"

        page_doi = extract_doi_from_page_metadata(response.text)
        if page_doi:
            report["Has_DOI_Page_Metadata"] = "Yes"
            report["Page_DOI_Value"] = page_doi

        if doi_jsonld_flag:
            report["Has_DOI"] = "Yes"

        people = extract_people_field(best)
        creator_names = extract_person_names(people)
        report["Creator_Count"] = len(creator_names)
        report["Creator_Names"] = " | ".join(creator_names[:10])

        orcid_flag = has_orcid(people)
        if orcid_flag:
            report["Has_ORCID"] = "Yes"

        if has_role_based_semantics(people):
            report["Structural_Type"] = "Semantic Role-Based"

        licences = extract_licenses(best)
        if licences:
            report["License_Status"] = "Present"
            report["License_Values"] = " | ".join(licences[:10])

        report["FAIR_Proxy_Score"] = score_fair_proxy(
            identifiers=all_identifiers,
            has_doi_flag=doi_jsonld_flag,
            has_orcid_flag=orcid_flag,
            licences=licences
        )

        distributions = extract_distributions(best)
        report["Distribution_Count"] = len(distributions)

        ma_score, ma_label, direct_links, formats = classify_distribution_strength(distributions)
        report["Machine_Actionable_Score"] = ma_score
        report["Machine_Actionable"] = ma_label
        report["Direct_Link_Count"] = len(direct_links)
        report["Distribution_Formats"] = " | ".join(formats[:10])

        var_score, var_desc, var_count = evaluate_variables(best)
        report["Variable_Description"] = var_desc
        report["Variable_Count"] = var_count

        functional_profile = infer_functional_profile(
            record_type,
            distributions,
            var_count,
            bool(licences)
        )
        report["Record_Profile"] = functional_profile

        temp_score, temp_label = evaluate_temporal_quality(best)
        report["Temporal_Quality"] = temp_label

        spatial_score, spatial_label = evaluate_spatial_quality(best)
        report["Spatial_Data"] = spatial_label

        role_score = 10 if report["Structural_Type"] == "Semantic Role-Based" else 0

        report["CDIF_AI_Ready_Score"] = score_cdif_ai_ready(
            record_type=record_type,
            variable_score=var_score,
            temporal_score=temp_score,
            spatial_score=spatial_score,
            role_score=role_score
        )

        report["Total_Composite_Score"] = score_total(
            machine_readable_score=report["Machine_Readable_Score"],
            fair_proxy_score=report["FAIR_Proxy_Score"],
            machine_actionable_score=report["Machine_Actionable_Score"],
            cdif_score=report["CDIF_AI_Ready_Score"],
            record_type=record_type
        )

        report["Overall_Quality"] = classify_overall_quality(report["Total_Composite_Score"])
        report["Curation_Effort"] = classify_curation_effort(report["Total_Composite_Score"])

        gaps = []
        if report["Has_DOI_JSONLD"] == "No":
            if report["Has_DOI_Page_Metadata"] == "Yes":
                gaps.append("DOI Missing in JSON-LD")
            else:
                gaps.append("No DOI/PID")

        if report["Has_ORCID"] == "No":
            gaps.append("No ORCID")
        if report["License_Status"] == "Missing":
            gaps.append("No Licence")
        if report["Distribution_Count"] == 0:
            gaps.append("No Distribution")
        elif report["Machine_Actionable"] != "High":
            gaps.append("Weak Direct Access")
        if "Dataset" in report["Record_Profile"] and report["Variable_Description"] == "None":
            gaps.append("No Variables")
        if report["Temporal_Quality"] == "Missing":
            gaps.append("No Temporal")
        if report["Spatial_Data"] == "Missing":
            gaps.append("No Spatial")

        report["Critical_Gaps"] = " | ".join(gaps)
        report["Path_to_100"] = build_path_to_100(report)
        return report

    except Exception as e:
        report["System_Error"] = str(e)
        report["Critical_Gaps"] = f"System Error: {str(e)}"
        return report


def run_master_audit_from_df(input_df: pd.DataFrame):
    url_list = input_df.iloc[:, 0].dropna().astype(str).tolist()
    total = len(url_list)
    results = [evaluate_rda_record(url, i, total) for i, url in enumerate(url_list, start=1)]
    final_df = pd.DataFrame(results)

    summary_data = {
        "Metric": [
            "Total Records Audited",
            "Average Machine Readable Score",
            "Average FAIR Proxy Score",
            "Average Machine Actionable Score",
            "Average CDIF/AI-Ready Score",
            "Average Composite Score",
            "Records with JSON-LD",
            "Records with RDFa",
            "Records with Microdata",
            "Records with Record Candidate",
            "Records with DOI",
            "Records with DOI in JSON-LD",
            "Records with DOI in Page Metadata",
            "Records with ORCID",
            "Records with Distribution",
            "Records with variableMeasured",
            "Dataset Records",
            "Collection Records"
        ],
        "Value": [
            total,
            round(final_df["Machine_Readable_Score"].mean(), 2),
            round(final_df["FAIR_Proxy_Score"].mean(), 2),
            round(final_df["Machine_Actionable_Score"].mean(), 2),
            round(final_df["CDIF_AI_Ready_Score"].mean(), 2),
            round(final_df["Total_Composite_Score"].mean(), 2),
            int((final_df["JSONLD_Found"] == "Yes").sum()),
            int((final_df["RDFa_Present"] == "Yes").sum()),
            int((final_df["Microdata_Present"] == "Yes").sum()),
            int((final_df["Dataset_Candidate_Found"] == "Yes").sum()),
            int((final_df["Has_DOI"] == "Yes").sum()),
            int((final_df["Has_DOI_JSONLD"] == "Yes").sum()),
            int((final_df["Has_DOI_Page_Metadata"] == "Yes").sum()),
            int((final_df["Has_ORCID"] == "Yes").sum()),
            int((final_df["Distribution_Count"] > 0).sum()),
            int((final_df["Variable_Count"] > 0).sum()),
            int((final_df["Record_Type"] == "Dataset").sum()),
            int((final_df["Record_Type"] == "Collection").sum())
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    notes_df = pd.DataFrame(NOTES_DATA)
    return summary_df, final_df, notes_df


def workbook_bytes(summary_df, final_df, notes_df):
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Executive Summary", index=False)
        final_df.to_excel(writer, sheet_name="Detailed Audit", index=False)
        notes_df.to_excel(writer, sheet_name="Notes", index=False)
    output.seek(0)
    return output.getvalue()

# =========================================================
# DASHBOARD HELPERS
# =========================================================
def detect_score_column(df: pd.DataFrame) -> str:
    for col in ["Total_Composite_Score", "Total Composite Score"]:
        if col in df.columns:
            return col
    raise ValueError("Could not find the composite score column.")


@st.cache_data(show_spinner=False)
def load_audit_workbook(file_obj):
    if str(getattr(file_obj, "name", "")).lower().endswith(".csv"):
        return pd.read_csv(file_obj)
    return pd.read_excel(file_obj, sheet_name="Detailed Audit")


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    defaults = {
        "Record_Type": "Unknown",
        "Record_Profile": "Unknown",
        "Overall_Quality": "Unknown",
        "Machine_Actionable": "Unknown",
        "Has_DOI_JSONLD": "No",
        "Has_DOI_Page_Metadata": "No",
        "Has_ORCID": "No",
        "License_Status": "Missing",
        "Critical_Gaps": "",
        "Path_to_100": "",
        "Variable_Count": 0,
        "Distribution_Count": 0,
        "Direct_Link_Count": 0,
        "Machine_Readable_Score": 0,
        "FAIR_Proxy_Score": 0,
        "Machine_Actionable_Score": 0,
        "CDIF_AI_Ready_Score": 0,
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    score_col = detect_score_column(df)
    numeric_cols = [
        score_col,
        "Machine_Readable_Score",
        "FAIR_Proxy_Score",
        "Machine_Actionable_Score",
        "CDIF_AI_Ready_Score",
        "Variable_Count",
        "Distribution_Count",
        "Direct_Link_Count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "DOI_Exposure" not in df.columns:
        df["DOI_Exposure"] = df.apply(
            lambda r: "JSON-LD DOI"
            if r.get("Has_DOI_JSONLD", "No") == "Yes"
            else ("Page DOI only" if r.get("Has_DOI_Page_Metadata", "No") == "Yes" else "No DOI"),
            axis=1,
        )

    if "Gap_Count" not in df.columns:
        df["Gap_Count"] = df["Critical_Gaps"].fillna("").apply(
            lambda x: 0 if not str(x).strip() else len([g for g in str(x).split("|") if g.strip()])
        )

    return df


def pct(series_bool: pd.Series) -> float:
    if len(series_bool) == 0:
        return 0.0
    return round(float(series_bool.mean() * 100), 1)


def build_kpi(label: str, value: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='small-label'>{label}</div>
            <div class='big-number'>{value}</div>
            <div class='subtle'>{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plot(fig):
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(size=13),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#edf2f7")
    return fig

# =========================================================
# APP LAYOUT
# =========================================================
st.markdown(
    """
    <div class='title-wrap'>
        <h1 style='margin-bottom:0.2rem;'>Luma: RDA Metadata Quality Auditor</h1>
        <div class='subtle'>Machine Readability, FAIR Proxy, Machine Actionability, and AI / CDIF Readiness</div>
    </div>
    """,
    unsafe_allow_html=True,
)

mode = st.sidebar.radio(
    "Mode",
    ["Dashboard from existing audit", "Run audit and open dashboard"],
)

audit_df = None
summary_df = None
notes_df = None

if "audit_df" not in st.session_state:
    st.session_state.audit_df = None
if "summary_df" not in st.session_state:
    st.session_state.summary_df = None
if "notes_df" not in st.session_state:
    st.session_state.notes_df = None
if "audit_mode_loaded" not in st.session_state:
    st.session_state.audit_mode_loaded = False
    
if mode == "Dashboard from existing audit":
    uploaded_audit = st.sidebar.file_uploader("Upload audit workbook or CSV", type=["xlsx", "xls", "csv"])
    if uploaded_audit is None:
        st.info("Upload an existing audit workbook or CSV to explore the dashboard.")

        st.markdown("---")
        st.markdown(
            """
            <div style='background-color:#f3f4f6; padding:18px; border-radius:12px; border:1px solid #e5e7eb; margin-top:20px;'>
                <div style='font-size:1rem; font-weight:600; margin-bottom:6px;'>
                    ⚠️ Demonstration Tool
                </div>
                <div style='font-size:0.95rem; color:#374151;'>
                    Luma has been developed by the Translational Research Data Challenges – ARDC. 
                    The tool is provided for demonstration purposes only, to make the concept tangible. 
                    Results should be interpreted as indicative rather than authoritative.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.stop()
    audit_df = load_audit_workbook(uploaded_audit)

else:
    uploaded_input = st.sidebar.file_uploader("Upload URL list workbook", type=["xlsx", "xls", "csv"])
    if uploaded_input is None:
        st.info("Upload the workbook containing the list of RDA URLs in the first column.")

        st.markdown("---")
        st.markdown(
            """
            <div style='text-align:center; color:#6b7280; font-size:0.9rem; margin-top:20px;'>
            Luma has been developed by the Translational Research Data Challenges – ARDC. 
            The tool is provided for demonstration purposes only, to make the concept tangible. 
            Results should be interpreted as indicative rather than authoritative.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.stop()

    if str(uploaded_input.name).lower().endswith(".csv"):
        input_df = pd.read_csv(uploaded_input)
    else:
        input_df = pd.read_excel(uploaded_input)

    run_now = st.sidebar.button("Run audit")

    if run_now:
        progress = st.progress(0)
        status = st.empty()

        urls = input_df.iloc[:, 0].dropna().astype(str).tolist()
        results = []
        total = len(urls)

        for i, url in enumerate(urls, start=1):
            status.write(f"Auditing {i} of {total}: {url}")
            results.append(evaluate_rda_record(url, i, total))
            progress.progress(i / total)

        st.session_state.audit_df = pd.DataFrame(results)
        st.session_state.summary_df = pd.DataFrame({
            "Metric": [
                "Total Records Audited",
                "Average Machine Readable Score",
                "Average FAIR Proxy Score",
                "Average Machine Actionable Score",
                "Average CDIF/AI-Ready Score",
                "Average Composite Score",
                "Records with JSON-LD",
                "Records with RDFa",
                "Records with Microdata",
                "Records with Record Candidate",
                "Records with DOI",
                "Records with DOI in JSON-LD",
                "Records with DOI in Page Metadata",
                "Records with ORCID",
                "Records with Distribution",
                "Records with variableMeasured",
                "Dataset Records",
                "Collection Records"
            ],
            "Value": [
                len(st.session_state.audit_df),
                round(st.session_state.audit_df["Machine_Readable_Score"].mean(), 2),
                round(st.session_state.audit_df["FAIR_Proxy_Score"].mean(), 2),
                round(st.session_state.audit_df["Machine_Actionable_Score"].mean(), 2),
                round(st.session_state.audit_df["CDIF_AI_Ready_Score"].mean(), 2),
                round(st.session_state.audit_df["Total_Composite_Score"].mean(), 2),
                int((st.session_state.audit_df["JSONLD_Found"] == "Yes").sum()),
                int((st.session_state.audit_df["RDFa_Present"] == "Yes").sum()),
                int((st.session_state.audit_df["Microdata_Present"] == "Yes").sum()),
                int((st.session_state.audit_df["Dataset_Candidate_Found"] == "Yes").sum()),
                int((st.session_state.audit_df["Has_DOI"] == "Yes").sum()),
                int((st.session_state.audit_df["Has_DOI_JSONLD"] == "Yes").sum()),
                int((st.session_state.audit_df["Has_DOI_Page_Metadata"] == "Yes").sum()),
                int((st.session_state.audit_df["Has_ORCID"] == "Yes").sum()),
                int((st.session_state.audit_df["Distribution_Count"] > 0).sum()),
                int((st.session_state.audit_df["Variable_Count"] > 0).sum()),
                int((st.session_state.audit_df["Record_Type"] == "Dataset").sum()),
                int((st.session_state.audit_df["Record_Type"] == "Collection").sum()),
            ]
        })
        st.session_state.notes_df = pd.DataFrame(NOTES_DATA)
        st.session_state.audit_mode_loaded = True

    if st.session_state.audit_mode_loaded and st.session_state.audit_df is not None:
        audit_df = st.session_state.audit_df
        summary_df = st.session_state.summary_df
        notes_df = st.session_state.notes_df

        excel_bytes = workbook_bytes(summary_df, audit_df, notes_df)
        st.download_button(
            label="Download audit workbook",
            data=excel_bytes,
            file_name="RDA_Metadata_Audit_Gold.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("Upload the workbook and click 'Run audit' to generate results.")
        st.stop()
        
# =========================================================
# DASHBOARD
# =========================================================
df = ensure_columns(audit_df)
score_col = detect_score_column(df)

st.sidebar.markdown("## Filters")
record_types = sorted(df["Record_Type"].dropna().astype(str).unique().tolist())
profiles = sorted(df["Record_Profile"].dropna().astype(str).unique().tolist())
qualities = sorted(df["Overall_Quality"].dropna().astype(str).unique().tolist())
actionability_levels = sorted(df["Machine_Actionable"].dropna().astype(str).unique().tolist())
doi_exposures = sorted(df["DOI_Exposure"].dropna().astype(str).unique().tolist())
license_levels = sorted(df["License_Status"].dropna().astype(str).unique().tolist())

selected_record_types = st.sidebar.multiselect("Record Type", record_types, default=record_types)
selected_profiles = st.sidebar.multiselect("Record Profile", profiles, default=profiles)
selected_qualities = st.sidebar.multiselect("Overall Quality", qualities, default=qualities)
selected_actionability = st.sidebar.multiselect("Machine Actionable", actionability_levels, default=actionability_levels)
selected_doi = st.sidebar.multiselect("DOI Exposure", doi_exposures, default=doi_exposures)
selected_licence = st.sidebar.multiselect("Licence Status", license_levels, default=license_levels)
score_range = st.sidebar.slider("Composite Score Range", 0, 100, (0, 100))
search_text = st.sidebar.text_input("Search URL / gaps / recommendation")

filtered = df[
    df["Record_Type"].astype(str).isin(selected_record_types)
    & df["Record_Profile"].astype(str).isin(selected_profiles)
    & df["Overall_Quality"].astype(str).isin(selected_qualities)
    & df["Machine_Actionable"].astype(str).isin(selected_actionability)
    & df["DOI_Exposure"].astype(str).isin(selected_doi)
    & df["License_Status"].astype(str).isin(selected_licence)
    & df[score_col].between(score_range[0], score_range[1])
].copy()

if search_text.strip():
    q = search_text.strip().lower()
    mask = (
        filtered.get("URL", "").astype(str).str.lower().str.contains(q, na=False)
        | filtered.get("Critical_Gaps", "").astype(str).str.lower().str.contains(q, na=False)
        | filtered.get("Path_to_100", "").astype(str).str.lower().str.contains(q, na=False)
        | filtered.get("Creator_Names", "").astype(str).str.lower().str.contains(q, na=False)
    )
    filtered = filtered[mask]

st.sidebar.markdown(f"**Records in view:** {len(filtered):,}")

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    build_kpi("Records in view", f"{len(filtered):,}")
with col2:
    build_kpi("Average score", f"{filtered[score_col].mean():.1f}")
with col3:
    build_kpi("DOI in JSON-LD", f"{pct(filtered['Has_DOI_JSONLD'].eq('Yes'))}%")
with col4:
    build_kpi("Licence present", f"{pct(filtered['License_Status'].eq('Present'))}%")
with col5:
    build_kpi("High actionability", f"{pct(filtered['Machine_Actionable'].eq('High'))}%")
with col6:
    build_kpi("Variables present", f"{pct(filtered['Variable_Count'].gt(0))}%")

left, right = st.columns(2)
with left:
    score_band = pd.cut(filtered[score_col], bins=[-0.1, 34.999, 69.999, 100], labels=["Low", "Moderate", "High"])
    band_df = score_band.value_counts(dropna=False).rename_axis("Band").reset_index(name="Count")
    fig = px.bar(band_df, x="Band", y="Count", title="Overall Quality Distribution")
    st.plotly_chart(style_plot(fig), use_container_width=True)
with right:
    type_df = filtered["Record_Profile"].value_counts().rename_axis("Record Profile").reset_index(name="Count")
    fig = px.pie(type_df, names="Record Profile", values="Count", hole=0.6, title="Record Profile Breakdown")
    style_plot(fig)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

c1, c2, c3 = st.columns(3)
with c1:
    doi_df = filtered["DOI_Exposure"].value_counts().rename_axis("DOI Exposure").reset_index(name="Count")
    fig = px.bar(doi_df, x="DOI Exposure", y="Count", title="DOI Exposure")
    st.plotly_chart(style_plot(fig), use_container_width=True)
with c2:
    all_gaps = []
    for val in filtered["Critical_Gaps"].fillna(""):
        all_gaps.extend([g.strip() for g in str(val).split("|") if g.strip()])
    gap_df = pd.Series(all_gaps).value_counts().head(10).rename_axis("Gap").reset_index(name="Count")
    if len(gap_df) == 0:
        gap_df = pd.DataFrame({"Gap": ["No gaps in current view"], "Count": [0]})
    fig = px.bar(gap_df, x="Count", y="Gap", orientation="h", title="Most Common Critical Gaps")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(style_plot(fig), use_container_width=True)
with c3:
    act_df = filtered["Machine_Actionable"].value_counts().rename_axis("Level").reset_index(name="Count")
    fig = px.bar(act_df, x="Level", y="Count", title="Machine Actionability")
    st.plotly_chart(style_plot(fig), use_container_width=True)

st.markdown("### Score Composition")
score_avg = pd.DataFrame(
    {
        "Metric": ["Machine Readable", "FAIR Proxy", "Machine Actionable", "AI / CDIF Ready", "Composite"],
        "Average Score": [
            filtered["Machine_Readable_Score"].mean(),
            filtered["FAIR_Proxy_Score"].mean(),
            filtered["Machine_Actionable_Score"].mean(),
            filtered["CDIF_AI_Ready_Score"].mean(),
            filtered[score_col].mean(),
        ],
    }
)
fig = px.bar(score_avg, x="Metric", y="Average Score", title="Average Scores by Dimension")
st.plotly_chart(style_plot(fig), use_container_width=True)

st.markdown("### Record Explorer")
view_cols = [
    c for c in [
        "URL", "Record_Type", "Record_Profile", score_col, "Overall_Quality",
        "FAIR_Proxy_Score", "Machine_Actionable_Score", "CDIF_AI_Ready_Score",
        "Has_DOI_JSONLD", "Has_DOI_Page_Metadata", "License_Status",
        "Machine_Actionable", "Critical_Gaps",
    ] if c in filtered.columns
]

st.dataframe(filtered[view_cols].sort_values(score_col, ascending=False), use_container_width=True, hide_index=True)

st.markdown("### Record Detail")
record_options = filtered["URL"].dropna().astype(str).tolist()
if record_options:
    selected_url = st.selectbox("Select a record", record_options)
    row = filtered.loc[filtered["URL"].astype(str) == selected_url].iloc[0]

    d1, d2 = st.columns([1.1, 1])
    with d1:
        st.markdown("#### Summary")
        summary_items = {
            "URL": row.get("URL", ""),
            "Record Type": row.get("Record_Type", ""),
            "Record Profile": row.get("Record_Profile", ""),
            "Overall Quality": row.get("Overall_Quality", ""),
            "Composite Score": row.get(score_col, ""),
            "Machine Actionable": row.get("Machine_Actionable", ""),
            "DOI in JSON-LD": row.get("Has_DOI_JSONLD", ""),
            "DOI in Page Metadata": row.get("Has_DOI_Page_Metadata", ""),
            "Licence": row.get("License_Status", ""),
            "Creators": row.get("Creator_Names", ""),
        }
        for k, v in summary_items.items():
            st.markdown(f"**{k}:** {v}")

    with d2:
        st.markdown("#### Scores")
        score_detail = pd.DataFrame(
            {
                "Metric": ["Machine Readable", "FAIR Proxy", "Machine Actionable", "AI / CDIF Ready", "Composite"],
                "Score": [
                    row.get("Machine_Readable_Score", 0),
                    row.get("FAIR_Proxy_Score", 0),
                    row.get("Machine_Actionable_Score", 0),
                    row.get("CDIF_AI_Ready_Score", 0),
                    row.get(score_col, 0),
                ],
            }
        )
        fig = px.bar(score_detail, x="Metric", y="Score", title="Record Score Profile")
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(style_plot(fig), use_container_width=True)

    x1, x2 = st.columns(2)
    with x1:
        st.markdown("#### Critical Gaps")
        st.info(row.get("Critical_Gaps", "No critical gaps recorded."))
    with x2:
        st.markdown("#### Path to 100")
        st.success(row.get("Path_to_100", "No recommendation available."))

    st.markdown("#### Metadata Diagnostics")
    diag_cols = [
        c for c in [
            "Has_DOI_JSONLD", "Has_DOI_Page_Metadata", "Page_DOI_Value", "Has_ORCID",
            "Identifier_Count", "Identifier_Values", "License_Status", "License_Values",
            "Distribution_Count", "Direct_Link_Count", "Distribution_Formats",
            "Variable_Description", "Variable_Count", "Spatial_Data", "Temporal_Quality",
            "Structural_Type",
        ] if c in filtered.columns
    ]
    st.dataframe(
        pd.DataFrame({"Field": diag_cols, "Value": [row.get(c, "") for c in diag_cols]}),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.warning("No records match the current filters.")

st.markdown("### Export")
export_csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download filtered results as CSV",
    data=export_csv,
    file_name="rda_metadata_filtered_results.csv",
    mime="text/csv",
)


st.markdown("---")

st.markdown(
    """
    <div style='text-align:center; color:#6b7280; font-size:0.85rem; margin-top:20px;'>
    Luma has been developed by the Translational Research Data Challenges – ARDC. 
    The tool is provided for demonstration purposes only, to make the concept tangible. 
    Results should be interpreted as indicative rather than authoritative.
    </div>
    """,
    unsafe_allow_html=True,
)
