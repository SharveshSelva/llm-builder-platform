import re
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

_analyzer = AnalyzerEngine()
_anonymizer = AnonymizerEngine()

INJECTION_PATTERNS = [
    r"ignore.{0,40}instructions?",
    r"disregard.{0,30}instructions?",
    r"you are now",
    r"act as (a |an )?",
    r"pretend (you are|to be)",
    r"system prompt",
    r"jailbreak",
    r"dan mode",
]

# Demographic bias — stereotyping statements about gender, race, age, religion
BIAS_PATTERNS = [
    r"\b(women|men|females|males)\s+are\s+(bad|worse|better|superior|inferior|not)\s+at\b",
    r"\ball\s+(asians?|blacks?|whites?|hispanics?|latinos?|arabs?|jews?|muslims?|christians?)\s+are\b",
    r"\b(old|elderly|young)\s+people\s+(can'?t|cannot|don'?t|are unable to)\b",
    r"\b(blacks?|whites?|asians?|hispanics?)\s+(people\s+)?(are\s+)?(always|never|typically|naturally)\b",
    r"\b(gender|race|religion|ethnicity)\s+(determines?|decides?|makes?)\b",
    r"\bnaturally\s+(better|worse|smarter|dumber|lazier|more violent)\b",
]

TOXICITY_PATTERNS = [
    r"\b(i hate|we hate)\s+all\b",
    r"\b(kill|murder|destroy|eliminate)\s+all\s+\w+",
    r"\bgo\s+(kill|die|hang)\s+yourself\b",
    r"\b(worthless|subhuman|inferior\s+race)\b",
    r"\b(f[u\*]ck|sh[i\*]t|b[i\*]tch|a[s\*]{2}hole)\b",
]


def detect_prompt_injection(text: str) -> bool:
    """Returns True if prompt injection is detected."""
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def detect_bias(text: str) -> bool:
    """
    Returns True if demographic bias or stereotyping language is detected.
    Runs on both inputs and model outputs for responsible AI compliance.
    """
    text_lower = text.lower()
    for pattern in BIAS_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def check_toxicity(text: str) -> bool:
    """
    Returns True if toxic, hateful, or abusive language is detected.
    Runs on model outputs as a post-generation safety check.
    """
    text_lower = text.lower()
    for pattern in TOXICITY_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def redact_pii(text: str) -> tuple[str, list[str]]:
    """
    Redacts PII (names, emails, phone numbers, SSNs, credit cards).
    Returns (redacted_text, list_of_entity_types_found).
    """
    results = _analyzer.analyze(text=text, language="en")
    if not results:
        return text, []
    entity_types = list({r.entity_type for r in results})
    anonymized = _anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text, entity_types


def validate_output(text: str) -> tuple[bool, str]:
    """
    Full output validation: length, refusal, bias, toxicity.
    Returns (is_valid, reason).
    """
    if len(text.strip()) < 5:
        return False, "Response too short"
    if text.strip().lower().startswith("i cannot") and len(text) < 100:
        return False, "Model refused without explanation"
    if detect_bias(text):
        return False, "Bias detected in output"
    if check_toxicity(text):
        return False, "Toxic content detected in output"
    return True, "ok"
