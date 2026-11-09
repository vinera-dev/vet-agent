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
    r"(responda|aja|atue|funcione|opere)\w* sem (nenhuma )?restric",
    r"sem (nenhuma )?restric\w* (de|para|nas?) (resposta|responder|conteudo|filtro)",
    r"desconsider\w* (o que|tudo|as |todas|suas|essas)",
    r"prompt (inicial|original|interno|oculto)",
    r"modo (administrador|admin|root|mestre)",
    r"\bpretend (you|to|that)\b",
    r"(you have no|without any|ignore your) (rules|restrictions|guidelines)",
    r"from now on,? you",
    r"repit\w* (literalmente |exatamente )?(tudo|o texto|a mensagem)",
    r"nao precisa\w* seguir (suas|as|nenhuma) (diretriz|regra|instruc|orientac)",
    r"\[(system|sistema|admin)\]",
    r"(texto|conteudo|detalhes) d[ao]s? suas? (instruc|regras|diretriz|orientac)",
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
    r"\bataques?\b.{0,30}\b(trem|convuls|espum)",
    r"\b(trem\w*|espum\w*) sem parar",
    r"(bebeu|tomou|lambeu|ingeriu|comeu|engoliu) (\w+ ){0,3}(agua sanitaria|alvejante|cloro|"
    r"desinfetante|soda caustica|anticongelante|raticida)",
    r"(mordid|picad)\w* (\w+ ){0,3}(cobra|escorpiao|aranha)",
    r"\b(cobra|escorpiao|aranha) (o |a |me )?(mordeu|picou)",
    r"nao (para|parou) de sangrar",
    r"\bsufoc",
    r"(lingua|gengiva|mucosa)s? (roxa|azul|palida|branca)",
    r"\bofegante\b",
    r"caiu d[aeo]s?\w* (janela|andar|sacada|laje|alto|telhado)",
    r"trabalho de parto.{0,40}(horas|nada nasce|nao nasce)",
    r"(corpo mole|inconsciente|desacordad\w*)",
    r"\b(ele|ela|gato|gata|cao|cachorro|cachorra|filhote)\b.{0,30}\bnao (responde|reage)\b",
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
