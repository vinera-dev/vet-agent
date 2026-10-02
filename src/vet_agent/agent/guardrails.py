import re
import unicodedata

INJECTION_PATTERNS = [
    r"ignor\w* (todas |tudo |as |suas |essas |estas |seus )*(instruc|regra|orientac|diretriz)",
    r"esquec\w* (todas |tudo |as |suas )*(instruc|regra|orientac)",
    r"ignore (all |any )?(previous|prior|above) (instructions|rules)",
    r"disregard (all |your )?(previous |prior )?(instructions|rules)",
    r"(system|developer) prompt",
    r"prompt (do|de) sistema",
    r"(revel|mostr|exib|repit)\w* (o |seu |suas |as )*(prompt|instruc)",
    r"voce agora (e|sera|vai ser)",
    r"a partir de agora,? (voce|aja|atue|responda)",
    r"\b(aja|atue|responda) como (se fosse )?(um|uma|o|a)\b",
    r"finj\w* (que )?(voce )?(e|ser|esta)\b",
    r"modo (desenvolvedor|dev|deus|sem restric)",
    r"jailbreak",
    r"\bdan\b.*(mode|modo)",
    r"sem (nenhuma )?restric",
]
EMERGENCY_PATTERNS = [
    r"convuls",
    r"envenen",
    r"intoxic",
    r"veneno",
    r"atropel",
    r"sangr\w* (muito|bastante|sem parar|intensa)",
    r"hemorragi",
    r"(nao|n) (esta|ta) respirando",
    r"(dificuldade|falta|problema) (de|para|pra|em) respirar",
    r"falta de ar",
    r"desmai",
    r"engasg",
    r"(comeu|ingeriu|engoliu|tomou) (\w+ ){0,3}(chocolate|veneno|remedio|uva|passas|cebola)",
    r"\b(barriga|abdomen|abdome)\b.{0,25}\b(inchad|distendid|dur[ao]\b)",
]
OUTPUT_POLICY_PATTERNS = [
    r"\b\d+([.,]\d+)?\s*(mg|mcg|ml|ui)\b",
    r"(?<!nao )(?<!nem )(?<!jamais )\b(recomendo|indico|sugiro|receito)\b.{0,40}"
    r"\b(medicament|remedio|antibiotic|anti-?inflamator)",
    r"\b(provavelmente|certamente|com certeza)\b.{0,30}\b(tem|esta com|sofre de|e)\b.{0,20}"
    r"\b(doenca|infeccao|virose|cinomose|parvovirose)",
    r"\bo diagnostico (e|seria)\b",
]
CPF_PATTERN = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_PATTERN = re.compile(r"(\+?55\s?)?\(?\b\d{2}\)?\s?9?\d{4}[-\s]?\d{4}\b")

_INJECTION = [re.compile(pattern) for pattern in INJECTION_PATTERNS]
_EMERGENCY = [re.compile(pattern) for pattern in EMERGENCY_PATTERNS]
_OUTPUT_POLICY = [re.compile(pattern) for pattern in OUTPUT_POLICY_PATTERNS]


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def detect_injection(text: str) -> bool:
    normalized = normalize(text)
    return any(pattern.search(normalized) for pattern in _INJECTION)


def detect_emergency(text: str) -> bool:
    normalized = normalize(text)
    return any(pattern.search(normalized) for pattern in _EMERGENCY)


def screen_input(text: str) -> str | None:
    if detect_emergency(text):
        return "emergency"
    if detect_injection(text):
        return "injection"
    return None


def violates_output_policy(text: str) -> bool:
    normalized = normalize(text)
    return any(pattern.search(normalized) for pattern in _OUTPUT_POLICY)


def redact_pii(text: str, keep_phone: bool = False) -> str:
    text = CPF_PATTERN.sub("[CPF]", text)
    text = EMAIL_PATTERN.sub("[EMAIL]", text)
    if not keep_phone:
        text = PHONE_PATTERN.sub("[TELEFONE]", text)
    return text
