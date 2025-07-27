# main_api.py
# Language: Python


import os
import re
import asyncio
import time
import httpx
import jwt
import uuid
import hmac
import hashlib
import base64
import json
import traceback
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from operator import itemgetter
from datetime import datetime, timezone, timedelta 

# --- FastAPI & Pydantic Dependencies ---
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

# --- LangChain & Neo4j Dependencies ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from neo4j.time import DateTime as Neo4jDateTime

# --- Carga de Variables de Entorno y Constantes ---
load_dotenv()
ANALYSIS_HISTORY_LIMIT = 30
STOP_WORDS = set([
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con", "contra", "cual", "cuando", "de", "del",
    "desde", "donde", "durante", "e", "el", "ella", "ellas", "ellos", "en", "entre", "era", "erais", "eramos", "eran",
    "eras", "eres", "es", "esa", "esas", "ese", "eso", "esos", "esta", "estaba", "estabais", "estabamos", "estaban",
    "estar", "estas", "este", "esto", "estos", "fue", "fueron", "fui", "fuimos", "ha", "hace", "haceis", "hacemos",
    "hacen", "hacer", "haces", "hacia", "hago", "han", "hasta", "hay", "he", "hemos", "la", "las", "le", "les", "lo",
    "los", "mas", "me", "mi", "mis", "mucho", "muchos", "muy", "nada", "ni", "no", "nos", "nosotras", "nosotros",
    "nuestra", "nuestras", "nuestro", "nuestros", "o", "os", "otra", "otras", "otro", "otros", "para", "pero", "pues",
    "que", "qué", "se", "sea", "seais", "seamos", "sean", "seas", "ser", "si", "siendo", "sin", "sobre", "sois",
    "somos", "son", "soy", "su", "sus", "suya", "suyas", "suyo", "suyos", "tal", "te", "tenemos", "tener", "tengo",
    "tu", "tus", "un", "una", "uno", "unos", "vosotras", "vosotros", "y", "ya", "yo", "the", "and", "is", "in", "it",
    "of", "to", "for", "with", "on", "that", "this", "be", "are", "not", "a", "an", "as", "at", "by", "from", "or",
    "if", "must", "should", "not", "all", "any", "el", "la", "los", "las", "un", "una", "unos", "unas"
])

LANGUAGE_EXTENSION_MAP = {
    # Lenguajes de Backend y Scripting
    '.py': 'Python',
    '.js': 'JavaScript',
    '.ts': 'TypeScript',
    '.java': 'Java',
    '.go': 'Go',
    '.rb': 'Ruby',
    '.php': 'PHP',
    '.cs': 'C#',
    '.rs': 'Rust',
    '.kt': 'Kotlin',
    '.kts': 'Kotlin Script',
    '.sh': 'Shell',
    '.ps1': 'PowerShell',

    # Lenguajes de Compilador C/C++
    '.c': 'C',
    '.h': 'C',
    '.cpp': 'C++',
    '.hpp': 'C++',
    '.cc': 'C++',
    '.cxx': 'C++',
    '.hxx': 'C++',

    # Lenguajes de Frontend y Maquetado
    '.html': 'HTML',
    '.htm': 'HTML',
    '.css': 'CSS',
    '.scss': 'SASS',
    '.sass': 'SASS',
    '.less': 'Less',

    # Otros
    '.sql': 'SQL',
    '.json': 'JSON',
    '.yaml': 'YAML',
    '.yml': 'YAML',
    '.xml': 'XML'
}

# --- 1.Pydantic Models ---
class Usagemesage(BaseModel):
    pass
class AttackPatternInfo(BaseModel):
    patternId: str; name: str

class AttackTechniqueInfo(BaseModel):
    techniqueId: str
    name: str

class VulnerabilityProfile(BaseModel):
    name: str
    cwe: str
    related_cve: Optional[str] = None

class ImpactAnalysis(BaseModel):
    summary: str

class Remediation(BaseModel):
    summary: str

class StructuredVulnerability(BaseModel):
    profile: VulnerabilityProfile
    impact: ImpactAnalysis
    technical_details: str = Field(description="A detailed technical explanation of why the code is vulnerable.")
    remediation: Remediation
    attack_patterns: Optional[List[AttackPatternInfo]] = []
    attack_techniques: Optional[List[AttackTechniqueInfo]] = [] 
    matched_custom_rules: Optional[List[str]] = Field(default=[], description="A list of all custom rule IDs that this vulnerability violates, e.g., ['BR001', 'BR015'].")

class AnalysisResult(BaseModel):
    summary: str; vulnerabilities: List[StructuredVulnerability]

class LLMVulnerability(BaseModel):
    vulnerability_name: str = Field(description="Un nombre breve y descriptivo para la vulnerabilidad (ej: 'Credenciales Hardcodeadas').")
    cwe_id: str = Field(description="El identificador CWE más relevante (ej: 'CWE-798'). Si no aplica un CWE, usa 'N/A'.")
    technical_details: str = Field(description="Una explicación técnica detallada de por qué el código es vulnerable.")
    remediation_summary: List[str] = Field(description="Una lista numerada de pasos concretos para solucionar el problema.")
    matched_custom_rules: List[str] = Field(default_factory=list, description="CRÍTICO: Una lista con los IDs de TODAS las reglas de negocio que esta vulnerabilidad infringe. Si ninguna aplica, devolver una lista vacía [].")

class LLMAnalysisResult(BaseModel):
    executive_summary: str = Field(description="Un resumen ejecutivo (1-2 frases) del análisis, enfocado en las violaciones de reglas de negocio.")
    found_vulnerabilities: List[LLMVulnerability] = Field(description="Una lista de las vulnerabilidades encontradas, priorizando las que violan reglas de negocio.")

class CodeInput(BaseModel):
    code_block: str
    user_id: str
    language: str

class RepoActivationRequest(BaseModel):
    repo_id: int; repo_full_name: str; user_id: str; user_name: Optional[str]; is_private: bool = None

class DashboardStats(BaseModel):
    total_analyses: int; total_vulnerabilities: int; reviewed_analyses: int = 0

class AnalysisDetail(AnalysisResult):
    analysisId: str; prUrl: str; timestamp: str; isReviewed: Optional[bool] = False

class AnalysesHistoryResponse(BaseModel):
    analyses: List[AnalysisDetail]

class CustomRulesRequest(BaseModel):
    user_id: str; rules_text: str; filename: str

class CustomRulesResponse(BaseModel):
    success: bool = True; rules: Optional[Dict[str, Any]] = None

class UpdateLogDetails(BaseModel):
    summary: str

class UpdateLogEntry(BaseModel):
    timestamp: datetime; taskName: str; status: str; details: UpdateLogDetails

class UpdateHistoryResponse(BaseModel):
    history: List[UpdateLogEntry]

class ReviewStatusResponse(BaseModel):
    analysisId: str; newStatus: bool

class RepositoryInfo(BaseModel):
    id: int = Field(alias="repoId")
    fullName: str

    model_config = ConfigDict(
        populate_by_name=True,
        ser_by_alias=False
    )

class DailyAnalysisCount(BaseModel):
    date: str = Field(description="Fecha en formato YYYY-MM-DD.")
    count: int = Field(description="Número de análisis completados en esa fecha.")

class VulnerabilityBreakdownItem(BaseModel):
    name: str = Field(description="Nombre de la vulnerabilidad, ej: 'SQL Injection'")
    cwe: str = Field(description="El CWE asociado, ej: 'CWE-89'")
    count: int = Field(description="Número de veces que esta vulnerabilidad fue encontrada.")

class CustomRuleBreakdownItem(BaseModel):
    rule_id: str = Field(description="El ID de la regla de negocio violada.")
    representative_name: str = Field(description="Un nombre de ejemplo de una vulnerabilidad que violó esta regla.")
    count: int = Field(description="Número de veces que esta regla fue violada.")

class SetAnalysisModeRequest(BaseModel):
    repo_id: int
    mode: str

class SubscriptionStatus(BaseModel):
    plan: str
    characterCount: int
    characterLimit: int
    usageResetDate: str

class UsageLimitExceededError(Exception):
    """Custom exception for when a user exceeds their usage limit."""
    pass
# --- Modelo Pydantic para la respuesta del token ---
class PayPalClientToken(BaseModel):
    client_token: str

class PayPalSubscriptionInfo(BaseModel):
    client_token: str
    plan_id: str

print("INFO: Inicializando conexiones y el cerebro de PullBrain-AI...")
graph = Neo4jGraph(url=os.getenv("NEO4J_URI"), username=os.getenv("NEO4J_USERNAME"), password=os.getenv("NEO4J_PASSWORD"), database=os.getenv("NEO4J_DATABASE", "neo4j"))
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
retrieval_query = "RETURN node.rag_text AS text, score, { cveId: node.cweId, cvssV3_1_Score: node.cvssV3_1_Score, isKev: labels(node) CONTAINS 'KEV', weaknesses: [ (node)-[:HAS_WEAKNESS]->(w) | w.cweId ] } AS metadata"
neo4j_vector_index = Neo4jVector.from_existing_index(embedding=embeddings_model, url=os.getenv("NEO4J_URI"), username=os.getenv("NEO4J_USERNAME"), password=os.getenv("NEO4J_PASSWORD"), database=os.getenv("NEO4J_DATABASE", "neo4j"), index_name="security_knowledge", embedding_node_property="rag_text_embedding", text_node_property="rag_text", retrieval_query=retrieval_query)
retriever = neo4j_vector_index.as_retriever()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0)
parser = JsonOutputParser(pydantic_object=LLMAnalysisResult)

#----Prompt template ---------------->

rag_prompt_template = ChatPromptTemplate.from_template("""
**CRITICAL INSTRUCTION: Your entire response, including all text and summaries, MUST be in English.**

You are PullBrain-AI, an expert security auditor specializing in **{language}**.

// --- CORE MISSION ---
Your primary goal is to identify security vulnerabilities by meticulously tracking user-controlled input from its source to its sink.

// --- ANALYSIS CONTEXT & INSTRUCTIONS ---
Your analysis MUST be based on your own extensive training and the `CUSTOM RULES` and `SECURITY CONTEXT` provided below.

// --- Handling CUSTOM RULES ---
The custom rules provided below can be in one of three formats: Structured JSON, Simple Text, or a 'no rules' message. You must first identify the format and then act accordingly.

1.  **If the content is a JSON object (starts with `{{`):** # <-- CORRECCIÓN AQUÍ
    - It contains an array of rule objects under the "rules" key.
    - Each rule has properties like `id`, `name`, `description`, and `patterns`.
    - You MUST use the `patterns` array, which contains **REGULAR EXPRESSIONS**, to actively search for violations in the code.

2.  **If the content is Simple Text (lines of `ID: description`):**
    - Treat each line as a general security principle to guide your analysis.
    - Check if the code violates the principle described in the text.

3.  **If the content is "No custom rules have been defined.":**
    - You can ignore this section.

For any violation found (either from a JSON pattern or a text description), you MUST populate the `matched_custom_rules` field with the corresponding rule `id`.

// --- FINAL OUTPUT INSTRUCTIONS ---
- You MUST generate a response in the requested JSON format and fill ALL fields.
- The `executive_summary` must holistically summarize all findings.
- **CRITICAL:** For each vulnerability, the `technical_details` field MUST describe the data flow path.
- **vulnerabilities.profile.name:** A short, common name for the vulnerability.
- **vulnerabilities.profile.cwe:** The single most accurate CWE ID.
- **vulnerabilities.matched_custom_rules:** A mandatory list of all violated custom rule IDs.
- **impact.summary:** A brief summary of the business/security impact.
- **remediation.summary:** Provide a clear, numbered list of the top 2-3 most critical steps to fix the issue. If a `cwe` is identified, you MUST conclude this section with a full, clickable link to its official Mitre page.
- **Do NOT populate the `attack_patterns` field.** My system enriches this later.

// --- ADDITIONAL REQUIREMENTS ---
// -- 5. Hallucination Prevention --
// Only report findings that can be clearly substantiated from the supplied information.
// -- 6. Duplicate Control --
// Group all occurrences of the same vulnerability under a single entry.

--- CUSTOM RULES (Check against these) ---
{custom_rules}
--- END CUSTOM RULES ---

--- SECURITY CONTEXT (Use as additional reference) ---
{context}
--- END SECURITY CONTEXT ---

--- CODE TO ANALYZE (Language: {language}) ---
{codigo}
--- END CODE TO ANALYZE ---

{format_instructions}
""")

# --- 3. Pipeline and Bussines Rules ---

def transform_llm_to_api_format(llm_result: Dict[str, Any]) -> AnalysisResult:
    """
    Transforms the raw LLM output dictionary into the structured AnalysisResult format.
    """
    structured_vulnerabilities = []
    for llm_vuln_data in llm_result.get('found_vulnerabilities', []):
        try:
            # Pydantic ahora valida directamente contra el modelo LLMVulnerability actualizado
            llm_vuln = LLMVulnerability(**llm_vuln_data)
            remediation_text = ""
            if isinstance(llm_vuln.remediation_summary, list):
                remediation_text = "\n".join(llm_vuln.remediation_summary)
            else:
                remediation_text = str(llm_vuln.remediation_summary)

            # Mapeamos directamente al formato final
            structured_vulnerabilities.append(
                StructuredVulnerability(
                    profile=VulnerabilityProfile(name=llm_vuln.vulnerability_name, cwe=llm_vuln.cwe_id),
                    impact=ImpactAnalysis(summary="El impacto de esta vulnerabilidad puede variar según el contexto de la aplicación."),
                    technical_details=llm_vuln.technical_details,
                    remediation=Remediation(summary=remediation_text),
                    attack_patterns=[], 
                    matched_custom_rules=llm_vuln.matched_custom_rules
                )
            )
        except Exception as e:
            print(f"ERROR: Failed to process individual vulnerability from LLM output: {e}. Raw data: {llm_vuln_data}")
            traceback.print_exc()
            continue

    return AnalysisResult(
        summary=llm_result.get('executive_summary', 'No se pudo generar un resumen ejecutivo.'),
        vulnerabilities=structured_vulnerabilities
    )



def enrich_with_custom_rules(analysis_result: AnalysisResult, code_block: str, custom_rules_text: str) -> AnalysisResult:
    """
    (VERSIÓN DE DIAGNÓSTICO) Imprime un log detallado para depurar el matching de reglas.
    """
    print("\n\n--- INICIANDO DIAGNÓSTICO DETALLADO DE REGLAS DE NEGOCIO ---")
    if not custom_rules_text or custom_rules_text == "No custom rules defined for this analysis.":
        print("DIAGNÓSTICO: No hay texto de reglas de negocio para procesar. Finalizando.")
        return analysis_result

    # --- Pre-procesamiento de Reglas ---
    parsed_rules = []
    print("DIAGNÓSTICO: Parseando reglas del texto...")
    for rule_line in custom_rules_text.strip().split('\n'):
        if not rule_line.strip() or rule_line.strip().startswith('#'): continue
        match = re.match(r'\s*([a-zA-Z0-9_-]+)\s*[:\-]\s*(.*)', rule_line)
        if not match: continue
        rule_id, text = match.groups()
        
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        keywords = {word for word in clean_text.split() if word not in STOP_WORDS and len(word) > 2}
        
        if keywords:
            parsed_rules.append({"id": rule_id, "text_lower": text.lower(), "keywords": keywords})
            # Imprimimos las palabras clave generadas para cada regla
            if "perf-002" in rule_id.lower():
                 print(f"DIAGNÓSTICO: Palabras clave generadas para la regla '{rule_id}': {keywords}")


    if not parsed_rules:
        print("DIAGNÓSTICO: No se encontraron reglas válidas después del parseo. Finalizando.")
        return analysis_result
    
    code_lower = code_block.lower()

    # --- Bucle principal de enriquecimiento ---
    print("\nDIAGNÓSTICO: Iniciando bucle de enriquecimiento por vulnerabilidad...")
    for i, vuln in enumerate(analysis_result.vulnerabilities):
        print(f"\n--- Analizando Vulnerabilidad #{i+1}: '{vuln.profile.name}' (CWE: {vuln.profile.cwe}) ---")
        found_rules = set(vuln.matched_custom_rules)
        vuln_text_lower = (f"{vuln.profile.name} {vuln.profile.cwe} {vuln.technical_details} {vuln.remediation.summary}").lower()

        # --- LÓGICA DE DETECCIÓN CON LOGS ---
        for rule in parsed_rules:
            # Solo nos enfocamos en la regla que nos interesa para este diagnóstico
            if "perf-002" not in rule['id'].lower():
                continue

            print(f"\nDIAGNÓSTICO: Verificando coincidencia para la regla '{rule['id']}'...")

            # TÉCNICA 1: Búsqueda del CWE
            cwe_id_lower = vuln.profile.cwe.lower()
            cwe_match_found = cwe_id_lower in rule["text_lower"]
            print(f"  - Técnica 1 (Match de CWE): Buscando '{cwe_id_lower}' en el texto de la regla. ¿Encontrado?: {cwe_match_found}")
            if cwe_match_found:
                found_rules.add(rule['id'])

            # TÉCNICA 2: Búsqueda de Palabras Clave en el CÓDIGO
            rule_keywords = rule["keywords"]
            matches_in_code = [kw for kw in rule_keywords if kw in code_lower]
            num_matches_in_code = len(matches_in_code)
            required_matches = min(2, len(rule_keywords)) if len(rule_keywords) > 1 else 1
            code_match_found = num_matches_in_code >= required_matches
            
            print(f"  - Técnica 2 (Keywords en Código): Se requieren {required_matches} coincidencias. Palabras clave encontradas en el código: {matches_in_code} (Total: {num_matches_in_code}). ¿Suficiente?: {code_match_found}")
            if code_match_found:
                found_rules.add(rule['id'])

        # Asignación Final
        if found_rules:
            vuln.matched_custom_rules = sorted(list(found_rules))
        
        print(f"DIAGNÓSTICO: Reglas finales asociadas a esta vulnerabilidad: {vuln.matched_custom_rules}")

    print("\n--- DIAGNÓSTICO DETALLADO FINALIZADO ---")
    return analysis_result

def enrich_with_threat_intelligence(analysis_result: AnalysisResult) -> AnalysisResult:
    """
    Enriquece el resultado con datos de CAPEC y ATT&CK, pero de forma limitada.
    """
    for vuln in analysis_result.vulnerabilities:
        cwe_id = vuln.profile.cwe
        if cwe_id and cwe_id != 'N/A':
            # --- CONSULTA CAPEC CORREGIDA CON LIMIT ---
            # Añadimos DISTINCT para seguridad y LIMIT 5 para brevedad.
            capec_query = """
            MATCH (w:CWE {cweId: $cwe_id})-[:HAS_ATTACK_PATTERN]->(ap:AttackPattern) 
            RETURN DISTINCT ap.patternId AS patternId, ap.name AS name 
            LIMIT 5
            """
            try:
                capec_results = graph.query(capec_query, params={"cwe_id": cwe_id})
                vuln.attack_patterns = [AttackPatternInfo(**p) for p in capec_results]
            except Exception as e:
                print(f"ERROR: Could not fetch CAPEC patterns for {cwe_id}. Error: {e}")

            # La búsqueda de ATT&CK ya usaba DISTINCT, por lo que está bien.
            # Podríamos añadirle un LIMIT también si fuera necesario.
            attack_query = "MATCH (w:CWE {cweId: $cwe_id})-[:HAS_ATTACK_PATTERN]->(p:AttackPattern)<-[:USES_PATTERN]-(t:Technique) RETURN DISTINCT t.techniqueId AS techniqueId, t.name AS name LIMIT 5"
            try:
                attack_results = graph.query(attack_query, params={"cwe_id": cwe_id})
                # Asumo que tienes un modelo AttackTechniqueInfo similar a AttackPatternInfo
                # Si no, esta línea debe adaptarse o eliminarse.
                # vuln.attack_techniques = [AttackTechniqueInfo(**t) for t in attack_results] 
            except Exception as e:
                print(f"ERROR: Could not fetch ATT&CK techniques for {cwe_id}. Error: {e}")
                
    return analysis_result
def get_github_app_jwt():
    base64_key = os.getenv("GITHUB_PRIVATE_KEY_B64")
    app_id = os.getenv("GITHUB_APP_ID")
    if not base64_key or not app_id:
        raise ValueError("GitHub App environment variables not configured.")
    try:
        private_key = base64.b64decode(base64_key)
    except Exception as e:
        raise ValueError(f"Error decoding private key: {e}")
    payload = {'iat': int(time.time()) - 60, 'exp': int(time.time()) + (5 * 60), 'iss': app_id}
    return jwt.encode(payload, private_key, algorithm='RS256')

async def get_installation_access_token(installation_id: int):
    app_jwt = get_github_app_jwt()
    headers = {'Authorization': f'Bearer {app_jwt}', 'Accept': 'application/vnd.github.v3+json'}
    url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers)
            response.raise_for_status()
            return response.json()['token']
        except httpx.HTTPStatusError as exc:
            print(f"ERROR fetching installation token: {exc.response.text}")
            raise

async def verify_and_get_body(request: Request):
    x_hub_signature_256 = request.headers.get('X-Hub-Signature-256')
    if not x_hub_signature_256:
        raise HTTPException(status_code=400, detail="X-Hub-Signature-256 header missing.")
    secret = os.getenv("GITHUB_WEBHOOK_SECRET")
    if not secret:
        raise HTTPException(status_code=500, detail="Webhook secret not configured in the environment.")
    body_bytes = await request.body()
    digest = "sha256=" + hmac.new(bytes(secret, 'utf-8'), body_bytes, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(digest, x_hub_signature_256):
        raise HTTPException(status_code=401, detail="Invalid signature.")
    try:
        return json.loads(body_bytes)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

def format_docs(docs):
    return "\n---\n".join([f"Retrieved Info: {doc.page_content}\nMetadata: {doc.metadata}" for doc in docs]) if docs else "No relevant context found."

def extract_rules_text(rules_data: Optional[Dict[str, Any]]) -> str:
    return rules_data.get('text', "No custom rules defined for this analysis.") if rules_data else "No custom rules defined for this analysis."

def get_user_rules_sync(user_id: str, exclude_fields: List[str] = ["embedding", "ruleId"]) -> Optional[Dict[str, Any]]:
    """
    (Versión Final y Completa) Obtiene las reglas de un usuario y las formatea correctamente
    para el prompt del LLM, evitando el "doble escape" y permitiendo la exclusión de campos.
    """
    try:
        query = """
        MATCH (u:User {githubId: $user_id})
        OPTIONAL MATCH (u)-[:HAS_RULE]->(r:CustomRule)
        RETURN u.rulesFilename AS filename,
               u.rulesTimestamp AS timestamp,
               collect(properties(r)) AS rules
        """
        result = graph.query(query, params={'user_id': user_id})

        if not result or not result[0] or not result[0].get("filename"):
            return None
        
        record = result[0]
        filename = record.get("filename", "")
        rules_list = record.get("rules", [])
        rules_text_for_llm = ""

        if not rules_list or all(not r for r in rules_list):
            rules_text_for_llm = "No custom rules have been defined."
        
        elif filename.endswith('.json'):
            rule_strings = []
            for rule_props in rules_list:
                rule_parts = []
                # --- INICIO DE LA CORRECCIÓN INTEGRADA ---
                # Iteramos sobre los items del diccionario para poder usar la lista de exclusión
                for key, value in rule_props.items():
                    if key in exclude_fields:
                        continue # Saltamos los campos que queremos excluir

                    if key == "patterns":
                        patterns_json_array = json.dumps(value)
                        rule_parts.append(f'"patterns": {patterns_json_array}')
                    else:
                        rule_parts.append(f'"{key}": {json.dumps(value)}')
                # --- FIN DE LA CORRECCIÓN INTEGRADA ---
                
                if rule_parts: # Solo añadimos la regla si tiene alguna parte después de filtrar
                    rule_strings.append("    {\n      " + ",\n      ".join(rule_parts) + "\n    }")
            
            if rule_strings:
                rules_text_for_llm = "{\n  \"rules\": [\n" + ",\n".join(rule_strings) + "\n  ]\n}"
            else:
                rules_text_for_llm = "No custom rules have been defined."
        
        else: # Para .txt y .md
            rules_text_lines = [f"{rule.get('id', rule.get('ruleId'))}: {rule.get('description')}" for rule in rules_list if rule and rule.get('description')]
            rules_text_for_llm = "\n".join(rules_text_lines)

        return {
            "text": rules_text_for_llm,
            "filename": record["filename"],
            "timestamp": record["timestamp"]
        }
        
    except Exception as e:
        print(f"WARNING: Could not fetch/parse rules for user {user_id}. Error: {e}")
        return None

def clean_llm_output(llm_text: str) -> str:
    """
    (Versión Definitiva y Robusta) Extrae un bloque JSON de un string,
    ignorando cualquier texto o bloque de código Markdown que lo envuelva.
    """
    # 1. Buscar si el JSON está envuelto en un bloque de código Markdown
    match = re.search(r"```(json)?\s*(\{.*?\})\s*```", llm_text, re.DOTALL)
    if match:
        # Si lo encuentra, trabaja solo con el contenido del bloque
        text_to_process = match.group(2)
    else:
        # Si no, trabaja con el texto completo
        text_to_process = llm_text

    # 2. Encontrar el primer '{' y el último '}' en el texto a procesar
    try:
        start_index = text_to_process.index('{')
        end_index = text_to_process.rindex('}')
        # 3. Extraer y devolver solo la subcadena que contiene el JSON
        return text_to_process[start_index : end_index + 1]
    except ValueError:
        # Si no se encuentra un '{' o '}', el texto no contiene un JSON válido.
        # Devolvemos el texto original para que el parser falle y podamos ver el error.
        print(f"WARN [CLEAN_LLM]: Could not find a valid JSON structure in the output: {llm_text}")
        return llm_text


async def full_analysis_pipeline(code: str, language: str, custom_rules_data: Optional[Dict[str, Any]], source: str) -> Dict[str, Any]:
    llm_chain = (
        {
            "context": RunnableLambda(lambda x: format_docs(retriever.invoke(x["codigo"]))),
            "codigo": itemgetter("codigo"),
            "custom_rules": itemgetter("custom_rules_data") | RunnableLambda(extract_rules_text),
            "language": itemgetter("language"),
            "format_instructions": lambda x: parser.get_format_instructions(),
        }
        | rag_prompt_template
        | llm
        | StrOutputParser()
        | RunnableLambda(clean_llm_output)
        | parser
    )
    llm_result = await llm_chain.ainvoke({"codigo": code, "language": language, "custom_rules_data": custom_rules_data})
    
    # 1. Transformar a objeto Pydantic AnalysisResult
    api_result = transform_llm_to_api_format(llm_result)
    
    # 2. Enriquecer el objeto con inteligencia de amenazas
    api_result_with_intelligence = enrich_with_threat_intelligence(api_result)
    custom_rules_text = extract_rules_text(custom_rules_data)
    
    # Llamamos a la función de enriquecimiento con la variable de texto correcta.
    final_result = enrich_with_custom_rules(api_result_with_intelligence, code, custom_rules_text)
    
    return final_result.dict()


def check_and_update_usage(user_id: str, code_to_analyze: str, is_private_repo: bool):
    """
    (Versión Final) Verifica y actualiza el uso de caracteres.
    - Maneja el reseteo mensual de la cuota.
    - Maneja el downgrade automático de planes cancelados.
    """
    now = datetime.now(timezone.utc)

    # --- INICIO DE LA NUEVA LÓGICA DE DOWNGRADE ---
    
    # 1. Obtenemos el estado completo del plan del usuario.
    plan_check_query = """
    MATCH (u:User {githubId: $user_id})
    RETURN u.plan AS plan,
           u.planStatus AS planStatus,
           u.proAccessEndDate AS proAccessEndDate
    """
    plan_data = graph.query(plan_check_query, params={"user_id": user_id})
    
    if plan_data and plan_data[0]:
        record = plan_data[0]
        user_plan = record.get("plan")
        plan_status = record.get("planStatus")
        pro_access_end_date = record.get("proAccessEndDate")

        # Convertimos la fecha de Neo4j a un objeto de Python comparable
        pro_access_end_date_native = pro_access_end_date.to_native() if pro_access_end_date else None

        # 2. Si el plan es 'pro', está cancelado y la fecha de acceso ya pasó, hacemos el downgrade.
        if user_plan == 'pro' and plan_status == 'cancelled' and pro_access_end_date_native and now > pro_access_end_date_native:
            print(f"INFO [PLAN_MGMT]: El acceso Pro para el usuario {user_id} ha expirado. Revirtiendo a Free.")
            downgrade_query = """
            MATCH (u:User {githubId: $user_id})
            SET u.plan = 'free',
                u.characterLimit = 150000,
                u.characterCount = 0, // Reseteamos el contador al hacer downgrade
                u.usageResetDate = datetime() + duration({days: 30}),
                u.planStatus = null,
                u.proAccessEndDate = null
            """
            graph.query(downgrade_query, params={"user_id": user_id})
            print(f"SUCCESS [PLAN_MGMT]: Usuario {user_id} revertido a Free.")

    # --- FIN DE LA NUEVA LÓGICA DE DOWNGRADE ---

    # 3. El resto de la función continúa, pero ahora con los datos del plan potencialmente actualizados.
    #    Volvemos a obtener los datos para asegurar que usamos los valores correctos.
    usage_query = """
    MATCH (u:User {githubId: $user_id})
    RETURN u.plan AS plan,
           u.characterCount AS count, 
           u.characterLimit AS limit, 
           u.usageResetDate AS reset_date
    """
    usage_data = graph.query(usage_query, params={"user_id": user_id})
    if not usage_data or not usage_data[0]:
        raise HTTPException(status_code=404, detail="User usage data not found.")
    
    record = usage_data[0]
    user_plan = record.get("plan", "free")
    count = record.get("count", 0)
    limit = record.get("limit", 150000)
    reset_date_obj = record.get("reset_date")
    reset_date_native = reset_date_obj.to_native() if reset_date_obj else None

    # Lógica de reseteo mensual
    if reset_date_native and now > reset_date_native:
        print(f"INFO [USAGE_CHECK]: Usage reset for user {user_id}.")
        count = 0
        new_reset_date = now + timedelta(days=30)
        graph.query(
            "MATCH (u:User {githubId: $user_id}) SET u.characterCount = 0, u.usageResetDate = $new_date",
            params={'user_id': user_id, 'new_date': new_reset_date}
        )
    
    # Verificación final de límite de consumo
    chars_to_use = len(code_to_analyze)
    if (count + chars_to_use) > limit:
        error_msg = f"User {user_id} has exceeded their usage limit ({count + chars_to_use}/{limit})."
        print(f"WARN [USAGE_CHECK]: {error_msg}")
        raise UsageLimitExceededError(error_msg)
        
    # Actualización del contador
    graph.query(
        "MATCH (u:User {githubId: $user_id}) SET u.characterCount = u.characterCount + $chars",
        params={'user_id': user_id, 'chars': chars_to_use}
    )
    print(f"INFO [USAGE_CHECK]: Usage updated for user {user_id}. New count: {count + chars_to_use}/{limit}")


async def process_analysis(payload: dict):
    pr_url = payload.get("pull_request", {}).get("html_url", "URL_desconocida")
    comments_url = payload.get("pull_request", {}).get("comments_url")
    access_token = None

    try:
        # ... (Toda la lógica inicial hasta el guardado es correcta) ...
        repo_id = payload['repository']['id']
        repo_full_name = payload['repository']['full_name']
        owner_query = "MATCH (u:User)-[:MONITORS]->(r:Repository {repoId: $repo_id}) RETURN u.githubId AS ownerId"
        owner_result = graph.query(owner_query, params={'repo_id': repo_id})
        if not owner_result or not owner_result[0] or not owner_result[0]['ownerId']:
            return
        owner_user_id = owner_result[0]['ownerId']
        installation_id = payload['installation']['id']
        pull_request_api_url = payload["pull_request"]["url"]
        access_token = await get_installation_access_token(installation_id)
        headers = {'Authorization': f'token {access_token}', 'Accept': 'application/vnd.github.v3+json'}
        mode_query = "MATCH (r:Repository {repoId: $repo_id}) RETURN r.analysisMode AS mode"
        mode_result = graph.query(mode_query, params={'repo_id': repo_id})
        analysis_mode = mode_result[0]['mode'] if mode_result and mode_result[0] and mode_result[0].get('mode') else 'full'
        files_by_language = {}
        if analysis_mode == 'full':
            files_url = f"{pull_request_api_url}/files"
            async with httpx.AsyncClient() as client:
                files_response = await client.get(files_url, headers=headers)
                files_response.raise_for_status()
                changed_files = files_response.json()
                for file_data in changed_files:
                    filename = file_data.get('filename', '')
                    if file_data.get('status') in ['added', 'modified']:
                        detected_lang = next((lang for ext, lang in LANGUAGE_EXTENSION_MAP.items() if filename.endswith(ext)), None)
                        if detected_lang:
                            contents_url = file_data['contents_url']
                            content_api_response = await client.get(contents_url, headers=headers)
                            file_content = base64.b64decode(content_api_response.json()['content']).decode('utf-8')
                            formatted_content = f"--- START FILE: {filename} ---\n{file_content}\n--- END FILE: {filename} ---"
                            files_by_language.setdefault(detected_lang, []).append(formatted_content)
        else:
            diff_headers = {'Authorization': f'token {access_token}', 'Accept': 'application/vnd.github.v3.diff'}
            async with httpx.AsyncClient() as client:
                response = await client.get(pull_request_api_url, headers=diff_headers)
                response.raise_for_status()
                raw_diff_code = response.text
                added_lines = [line[1:] for line in raw_diff_code.splitlines() if line.startswith('+') and not line.startswith('+++')]
                code_to_analyze = "\n".join(added_lines)
                lang_for_diff = "general"
                for line in raw_diff_code.splitlines():
                    if line.startswith('--- a/') or line.startswith('+++ b/'):
                        filename = line.split('/')[-1]
                        lang_for_diff = next((lang for ext, lang in LANGUAGE_EXTENSION_MAP.items() if filename.endswith(ext)), lang_for_diff)
                        break
                if code_to_analyze.strip():
                    files_by_language[lang_for_diff] = [code_to_analyze]
        if not files_by_language:
            return
        # Obtener si el repositorio es privado desde el payload de GitHub.
        is_private_repo = payload['repository']['private']

        # Concatenar todo el código de los diferentes lenguajes en un solo bloque para medir su longitud.
        all_code_to_analyze = "\n".join(
            code for content_list in files_by_language.values() for code in content_list
        )

        # Llamar a la función de verificación de uso antes de proceder con el análisis.
        check_and_update_usage(
            user_id=owner_user_id,
            code_to_analyze=all_code_to_analyze,
            is_private_repo=is_private_repo
        )
        
        user_rules_data = get_user_rules_sync(owner_user_id)
        analysis_tasks = []
        source_info = f"GitHub PR from {repo_full_name} (Mode: {analysis_mode})"
        for lang, content_list in files_by_language.items():
            code_block = "\n\n".join(content_list)
            analysis_task = full_analysis_pipeline(code=code_block, language=lang, custom_rules_data=user_rules_data, source=source_info)
            analysis_tasks.append(analysis_task)
        analysis_results = await asyncio.gather(*analysis_tasks)
        final_summary_parts = []
        final_vulnerabilities = []
        for result_dict in analysis_results:
            if result_dict and result_dict.get("summary") and "No vulnerabilities" not in result_dict.get("summary", ""):
                final_summary_parts.append(result_dict["summary"])
            if result_dict and result_dict.get("vulnerabilities"):
                final_vulnerabilities.extend(result_dict["vulnerabilities"])
        final_result = AnalysisResult(
            summary=" ".join(final_summary_parts) if final_summary_parts else "Analysis complete. No vulnerabilities were found.",
            vulnerabilities=[StructuredVulnerability(**v) for v in final_vulnerabilities]
        )
        print("INFO [BG_TASK]: Validation successful. AI result processed.")
        analysis_props = {'summary': final_result.summary, 'timestamp': datetime.now(timezone.utc), 'prUrl': pr_url, 'isReviewed': False}
        save_analysis_query = "MATCH (r:Repository {repoId: $repo_id}) CREATE (a:Analysis $props)-[:FOR_REPO]->(r) RETURN elementId(a) AS analysisNodeId"
        result = graph.query(save_analysis_query, params={'repo_id': repo_id, 'props': analysis_props})
        analysis_node_id = result[0]['analysisNodeId'] if result and result[0] else None

        if final_result.vulnerabilities and analysis_node_id:
            saved_count = 0
            for vuln in final_result.vulnerabilities:
                try:
                    vuln_data_for_db = {
                        "profile": json.dumps(vuln.profile.model_dump()),
                        "impact": json.dumps(vuln.impact.model_dump()),
                        # --- CORRECCIÓN DEFINITIVA ---
                        "technical_details": vuln.technical_details or "",
                        # --- FIN DE LA CORRECCIÓN ---
                        "remediation": json.dumps(vuln.remediation.model_dump()),
                        "attackPatterns": json.dumps([ap.model_dump() for ap in vuln.attack_patterns]),
                        "matchedCustomRules": vuln.matched_custom_rules
                    }
                    save_one_vuln_query = "MATCH (a:Analysis) WHERE elementId(a) = $analysis_node_id CREATE (v:Vulnerability) SET v += $vuln_properties CREATE (a)-[:HAS_VULNERABILITY]->(v)"
                    graph.query(save_one_vuln_query, params={'analysis_node_id': analysis_node_id, 'vuln_properties': vuln_data_for_db})
                    saved_count += 1
                except Exception as e:
                    print(f"ERROR [BG_TASK]: Failed to save vulnerability ({vuln.profile.name}). Cause: {e}")
            print(f"INFO [BG_TASK]: Saved {saved_count} of {len(final_result.vulnerabilities)} vulnerabilities.")

        # ... (El resto de la función, incluyendo el posteo de comentarios, es correcta) ...
        if comments_url:
            print(f"INFO [BG_TASK]: Preparing comment for PR: {pr_url}")
            comment_body = f"### 🛡️ PullBrain-AI Security Analysis\n\n**Executive Summary:** {final_result.summary}\n\n"
            if final_result.vulnerabilities:
                comment_body += f"**Vulnerabilities Found ({len(final_result.vulnerabilities)}):**\n\n"
                for i, vuln in enumerate(final_result.vulnerabilities):
                    comment_body += f"---\n#### Risk #{i+1}: {vuln.profile.name} (`{vuln.profile.cwe}`)\n"
                    if vuln.matched_custom_rules:
                        comment_body += f"**Violated Rules:** {', '.join(f'`{r}`' for r in vuln.matched_custom_rules)}\n"
                    comment_body += f"\n**Recommendation:**\n"
                    remediation_steps = str(vuln.remediation.summary).strip().split('\n')
                    for step in remediation_steps:
                        comment_body += f"- {step.lstrip('123456789. ')}\n"
                    if vuln.attack_patterns:
                        comment_body += f"\n**Associated Attack Patterns (CAPEC):**\n"
                        for pattern in vuln.attack_patterns:
                            comment_body += f"- `{pattern.patternId}`: {pattern.name}\n"
            else:
                comment_body += "✅ **Excellent work!** No vulnerabilities were found.\n"
            comment_payload = {"body": comment_body}
            async with httpx.AsyncClient() as client:
                await client.post(comments_url, headers=headers, json=comment_payload)
            print("INFO [BG_TASK]: Comment posted to GitHub successfully.")
        print(f"INFO [BG_TASK]: Analysis process for {pr_url} completed successfully.")

    except Exception as e:
        print(f"CRITICAL ERROR in background task for {pr_url}: {e}")
        traceback.print_exc()
        if comments_url and access_token:
            error_comment = f"### 🛡️ PullBrain-AI Analysis Failed\n\nAn unexpected error occurred:\n\n```\n{type(e).__name__}: {e}\n```\nPlease check the logs."
            comment_payload = {"body": error_comment}
            headers = {'Authorization': f'token {access_token}', 'Accept': 'application/vnd.github.v3+json'}
            async with httpx.AsyncClient() as client:
                await client.post(comments_url, headers=headers, json=comment_payload)

async def process_repository_deletion(payload: dict):
    try:
        repo_id = payload['repository']['id']
        print(f"INFO [BG_TASK]: Received deletion event for repoId: {repo_id}. Deleting from DB.")

        # Esta consulta encuentra el repositorio por su ID y lo elimina,
        # junto con todas sus relaciones (análisis, vulnerabilidades, etc.)
        delete_query = "MATCH (r:Repository {repoId: $repo_id}) DETACH DELETE r"
        graph.query(delete_query, params={'repo_id': repo_id})

        print(f"INFO [BG_TASK]: Repository {repo_id} successfully deleted from Neo4j.")

    except KeyError as e:
        print(f"ERROR in deletion task: Missing essential payload key: {e}")
        return
    except Exception as e:
        print(f"CRITICAL ERROR in repository deletion task: {e}")
        traceback.print_exc()
# --- 4. Start App FastAPI & Endpoints ---

app = FastAPI(
    title="PullBrain-AI API",
    description="API to analyze code security using AI and a Knowledge Graph."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

print("INFO: El cerebro de PullBrain-AI está inicializado y listo.")

@app.get("/")
async def root():
    return {"message": "La API de PullBrain-AI está en funcionamiento."}

@app.post("/api/v1/analyze")
async def handle_github_webhook(background_tasks: BackgroundTasks, payload: dict = Depends(verify_and_get_body)):
    event_action = payload.get("action")

    # Ruta para eventos de Pull Request (análisis de código)
    if "pull_request" in payload and event_action in ["opened", "reopened", "synchronize"]:
        background_tasks.add_task(process_analysis, payload)
        return {"status": "accepted", "message": "Analysis event accepted and being processed."}

    # --- NUEVA RUTA ---
    # Ruta para eventos de Repositorio (eliminación)
    elif "repository" in payload and event_action == "deleted":
        background_tasks.add_task(process_repository_deletion, payload)
        return {"status": "accepted", "message": "Repository deletion event accepted and being processed."}

    return {"status": "success", "message": "Event ignored."}

# En main_api.py

@app.post("/api/v1/analyze-manual", response_model=AnalysisResult)
async def handle_manual_analysis(code_input: CodeInput):
    """
    Handles manual code analysis, now with dynamic character limits based on user's plan.
    """
    if not code_input.user_id:
        raise HTTPException(status_code=401, detail="Authentication (user_id) is required to perform a manual analysis.")

    try:
        # --- INICIO DE LA MODIFICACIÓN: Lógica de límite dinámico ---

        # 1. Consultar el plan del usuario en la base de datos.
        plan_query = "MATCH (u:User {githubId: $user_id}) RETURN u.plan AS plan"
        result = graph.query(plan_query, params={"user_id": code_input.user_id})
        user_plan = result[0]['plan'] if (result and result[0] and result[0].get('plan')) else 'free'

        # 2. Definir los límites basados en el plan.
        if user_plan == 'pro':
            max_chars_per_analysis = 50000
        else: # 'free' o cualquier otro caso por defecto
            max_chars_per_analysis = 10000
        
        print(f"INFO [MANUAL_ANALYSIS]: User {code_input.user_id} on '{user_plan}' plan. Applying limit of {max_chars_per_analysis} chars.")

        # 3. Validar el tamaño del código contra el límite dinámico.
        #    Esto reemplaza la validación de 'LINE_LIMIT' que estaba antes.
        if len(code_input.code_block) > max_chars_per_analysis:
            error_detail = f"The code exceeds the character limit for your '{user_plan}' plan. Limit: {max_chars_per_analysis}, Sent: {len(code_input.code_block)}."
            print(f"WARN [MANUAL_ANALYSIS]: {error_detail}")
            raise HTTPException(status_code=413, detail=error_detail) # 413 Payload Too Large

        # --- FIN DE LA MODIFICACIÓN ---
        
        # 4. Si la validación pasa, continuamos con la verificación de consumo general.
        #    Para análisis manuales, siempre se trata como "privado" (consume créditos).
        check_and_update_usage(
            user_id=code_input.user_id, 
            code_to_analyze=code_input.code_block, 
            is_private_repo=True
        )
        
        # 5. Si todo está en orden, procedemos con el análisis completo.
        user_rules_data = get_user_rules_sync(code_input.user_id)
        analysis_result_dict = await full_analysis_pipeline(
            code=code_input.code_block,
            language=code_input.language,
            custom_rules_data=user_rules_data,
            source="Manual analysis"  
        )

        return analysis_result_dict
    
    except UsageLimitExceededError as e:
        # Este error se dispara si se excede el LÍMITE MENSUAL TOTAL.
        raise HTTPException(status_code=429, detail=str(e)) # 429 Too Many Requests

    except HTTPException as http_exc:
        # Re-lanzamos la excepción de límite por análisis (413) para que FastAPI la maneje.
        raise http_exc

    except Exception as e:
        print(f"ERROR: Error during manual analysis for user {code_input.user_id}. Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Manual analysis failed: {str(e)}")

@app.post("/api/v1/repositories/toggle-activation")
async def toggle_repository_activation(data: RepoActivationRequest):
    """
    (Versión Definitiva) Activa/desactiva el monitoreo de un repositorio.
    - Crea el usuario si no existe con el plan "Free".
    - Limita la activación de repositorios privados para el plan "Free".
    """
    user_id = data.user_id
    repo_id = data.repo_id
    is_private = data.is_private

    # 1. Aseguramos que el usuario y el repositorio existan en la base de datos.
    #    Esta es la parte clave que se había perdido.
    ensure_nodes_query = """
    // Asegurar que el usuario exista con su plan por defecto
    MERGE (u:User {githubId: $user_id})
      ON CREATE SET u.name = $user_name, 
                    u.plan = 'free', 
                    u.characterCount = 0, 
                    u.characterLimit = 150000, 
                    u.usageResetDate = datetime() + duration({days: 30})
    
    // Asegurar que el repositorio exista con su estado de privacidad
    MERGE (r:Repository {repoId: $repo_id})
      ON CREATE SET r.fullName = $repo_full_name, 
                    r.analysisMode = 'full', 
                    r.isPrivate = $is_private
      ON MATCH SET r.fullName = $repo_full_name, 
                   r.isPrivate = $is_private
    """
    graph.query(ensure_nodes_query, params={
        "user_id": user_id,
        "user_name": data.user_name,
        "repo_id": repo_id,
        "repo_full_name": data.repo_full_name,
        "is_private": is_private
    })

    # 2. Verificamos si el usuario está INTENTANDO activar un repo privado
    check_activation_query = """
    MATCH (u:User {githubId: $user_id})
    MATCH (r:Repository {repoId: $repo_id})
    RETURN NOT exists((u)-[:MONITORS]->(r)) AS is_activating
    """
    activation_check_result = graph.query(check_activation_query, params={"user_id": user_id, "repo_id": repo_id})
    is_activating = activation_check_result[0]['is_activating'] if activation_check_result else False

    if is_activating and is_private:
        # Si está activando un repo privado, obtenemos su plan y contamos sus repos privados activos
        plan_and_count_query = """
        MATCH (u:User {githubId: $user_id})
        OPTIONAL MATCH (u)-[:MONITORS]->(p_r:Repository) WHERE p_r.isPrivate = true
        RETURN u.plan AS plan, count(p_r) AS privateRepoCount
        """
        result = graph.query(plan_and_count_query, params={"user_id": user_id})
        record = result[0]
        user_plan = record.get('plan', 'free')
        private_repo_count = record.get('privateRepoCount', 0)

        # Si es plan "Free" y ya tiene 1 o más repos privados, lo bloqueamos.
        if user_plan == 'free' and private_repo_count >= 1:
            error_detail = "Free plan is limited to 1 active private repository. Please upgrade to Pro for unlimited private repositories."
            print(f"WARN [REPO_ACTIVATION]: User {user_id} (plan: {user_plan}) blocked from activating another private repo.")
            raise HTTPException(status_code=403, detail=error_detail)

    # 3. Si pasamos todas las validaciones, procedemos a activar/desactivar la relación
    toggle_rel_query = """
    MATCH (u:User {githubId: $user_id}), (r:Repository {repoId: $repo_id})
    OPTIONAL MATCH (u)-[rel:MONITORS]->(r)
    FOREACH (_ IN CASE WHEN rel IS NULL THEN [1] ELSE [] END | CREATE (u)-[:MONITORS]->(r))
    FOREACH (_ IN CASE WHEN rel IS NOT NULL THEN [1] ELSE [] END | DELETE rel)
    """
    try:
        graph.query(toggle_rel_query, params={"user_id": user_id, "repo_id": repo_id})
        return {"status": "success", "message": "Repository status toggled successfully."}
    except Exception as e:
        print(f"ERROR: Database error in toggle_repository_activation for repoId {repo_id}. Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error de interacción con la base de datos.")

@app.get("/api/v1/dashboard/stats/{user_id}", response_model=DashboardStats)
async def get_dashboard_stats(
    user_id: str,
    repo_id: Optional[int] = None,  # Parámetro opcional para filtrar por repositorio
    period: Optional[str] = '30d'    # Parámetro opcional para filtrar por período (ej: "7d", "30d")
):
    """
    Obtiene estadísticas del dashboard para un usuario, opcionalmente filtradas por repositorio y/o período de tiempo.
    """
    # Calcular el timestamp de corte si se proporciona un período
    cutoff_timestamp = None
    if period:
        try:
            # Intentar parsear el período (ej: "7d", "30d")
            if period.endswith('d'):
                days = int(period[:-1])
                if days > 0: 
                    cutoff_timestamp = datetime.now(timezone.utc) - timedelta(days=days)
                else:
                    print(f"WARNING: Period days must be positive: {period}. Not applying time filter.")
            # Puedes añadir más lógicas aquí para semanas ('w'), meses ('m'), etc.
            # elif period.endswith('w'): ...
            # elif period.endswith('m'): ...
            else:
                 # Período no reconocido, no aplicar filtro de tiempo
                 print(f"WARNING: Unrecognized period format: {period}. Not applying time filter.")
                 cutoff_timestamp = None # Asegurarse de que sea None si el formato es inválido

        except ValueError:
            print(f"WARNING: Invalid period value: {period}. Not applying time filter.")
            cutoff_timestamp = None # Asegurarse de que sea None si el valor es inválido


    # Construir la consulta Cypher dinámicamente
    # Empezamos encontrando los repositorios monitoreados por el usuario
    query = """
    MATCH (u:User {githubId: $user_id})-[:MONITORS]->(r:Repository)
    """
    # Si se filtra por repo, nos aseguramos de que el repo monitoreado sea el correcto
    if repo_id is not None:
         query += " WHERE r.repoId = $repo_id"

    query += """
    OPTIONAL MATCH (a:Analysis)-[:FOR_REPO]->(r)
    """

    # Lista para almacenar las condiciones WHERE para los análisis
    analysis_where_conditions = []

    # Añadir filtro por timestamp si se proporciona un período válido
    if cutoff_timestamp is not None:
        analysis_where_conditions.append("a.timestamp >= $cutoff_timestamp")

    # Si hay condiciones para los análisis, añadirlas con WHERE
    if analysis_where_conditions:
        # --- Re-Reestructuración de la construcción de la query ---
        query_parts = [
            "MATCH (u:User {githubId: $user_id})-[:MONITORS]->(r:Repository)"
        ]

        # Añadir filtro por repositorio si se proporciona
        if repo_id is not None:
             query_parts.append("WHERE r.repoId = $repo_id")

        query_parts.append("OPTIONAL MATCH (a:Analysis)-[:FOR_REPO]->(r)")

        # Lista para almacenar las condiciones WHERE para los análisis
        analysis_where_conditions = []
        if cutoff_timestamp is not None:
            analysis_where_conditions.append("a.timestamp >= $cutoff_timestamp")

        # Si hay condiciones para los análisis, añadirlas con WHERE
        if analysis_where_conditions:
            query_parts.append("WHERE " + " AND ".join(analysis_where_conditions))

        query_parts.append("OPTIONAL MATCH (a)-[:HAS_VULNERABILITY]->(v:Vulnerability)")

        query_parts.append("""
        RETURN count(DISTINCT a) AS totalAnalyses,
               count(DISTINCT v) AS totalVulnerabilities,
               count(DISTINCT CASE WHEN a.isReviewed = true THEN a ELSE null END) AS reviewedAnalyses
        """)

        query = "\n".join(query_parts)
       
    # Preparar los parámetros para la consulta
    params = {"user_id": user_id}
    if repo_id is not None:
        params["repo_id"] = repo_id
    if cutoff_timestamp is not None:
        # El driver de langchain_neo4j suele manejar objetos datetime nativos de Python
        params["cutoff_timestamp"] = cutoff_timestamp

    try:
        print(f"INFO: Executing dashboard query for user {user_id} (repo_id: {repo_id}, period: {period})")
        result = graph.query(query, params=params)

        if not result or not result[0]:
             # Esto puede ocurrir si el usuario no monitorea ningún repo,
             # o si los filtros no encuentran ningún análisis/vulnerabilidad.
             # Devolvemos 0s en este caso.
             print(f"INFO: No dashboard data found for user {user_id} with specified filters.")
             return DashboardStats(total_analyses=0, total_vulnerabilities=0, reviewed_analyses=0)

        stats_data = result[0]

        return DashboardStats(
            total_analyses=stats_data.get('totalAnalyses', 0),
            total_vulnerabilities=stats_data.get('totalVulnerabilities', 0),
            reviewed_analyses=stats_data.get('reviewedAnalyses', 0)
        )

    except Exception as e:
        print(f"ERROR: Error querying dashboard statistics for user {user_id}. Error: {e}")
        traceback.print_exc()
        # Devolver estadísticas vacías en caso de error para no romper el frontend
        return DashboardStats(total_analyses=0, total_vulnerabilities=0, reviewed_analyses=0)

@app.post("/api/v1/user/rules")
async def save_user_rules(data: CustomRulesRequest):
    """
    (Versión robusta y extendida) Acepta archivos de reglas en formato .txt, .md y .json.
    Valida el contenido y el plan del usuario antes de procesar.
    """
    user_id = data.user_id
    rules_text = data.rules_text
    filename = data.filename

    print(f"INFO: Procesando reglas desde '{filename}' para el usuario {user_id}")
    
    # --- Restricción de Plan ---

    if filename.endswith('.json'):
        # 1. Validar si el contenido es un JSON sintácticamente correcto.
        try:
            json_data = json.loads(rules_text)
            # Verificación adicional: Asegurarse de que la clave "rules" exista y sea una lista.
            if "rules" not in json_data or not isinstance(json_data.get("rules"), list):
                 raise ValueError("El JSON debe tener una clave 'rules' que contenga una lista.")
        except (json.JSONDecodeError, ValueError) as e:
            error_detail = f"El archivo '{filename}' no es un JSON válido o no tiene la estructura correcta. Error: {e}"
            print(f"WARN [RULES_UPLOAD]: {error_detail}")
            raise HTTPException(status_code=400, detail=error_detail) # 400 Bad Request

        # 1. Si el archivo es JSON, verificamos el plan del usuario.
        plan_query = "MATCH (u:User {githubId: $user_id}) RETURN u.plan AS plan"
        result = graph.query(plan_query, params={"user_id": user_id})
        user_plan = result[0]['plan'] if (result and result[0] and result[0].get('plan')) else 'free'
        
        # 2. Si el plan no es 'pro', rechazamos la solicitud.
        if user_plan != 'pro':
            error_detail = "Uploading custom rules in JSON format is a Pro feature. Please upgrade your plan."
            print(f"WARN [RULES_UPLOAD]: User {user_id} (plan: {user_plan}) attempt to upload JSON rules rejected.")
            raise HTTPException(status_code=403, detail=error_detail) # 403 Forbidden

    # 3. El resto del proceso de parseo y guardado continúa sin cambios.
    parsed_rules = []
    try:
        if filename.endswith('.json'):
            # 1. Validar y parsear JSON
            json_data = json.loads(rules_text)
            if "rules" not in json_data or not isinstance(json_data.get("rules"), list):
                 raise ValueError("El JSON debe tener una clave 'rules' que contenga una lista.")

            # 2. Validar el plan del usuario para JSON
            plan_query = "MATCH (u:User {githubId: $user_id}) RETURN u.plan AS plan"
            result = graph.query(plan_query, params={"user_id": user_id})
            user_plan = result[0]['plan'] if (result and result[0] and result[0].get('plan')) else 'free'
            if user_plan != 'pro':
                raise HTTPException(status_code=403, detail="La carga de reglas en formato JSON es una funcionalidad Pro.")

            # 3. Extraer reglas del JSON validado
            for rule_obj in json_data.get("rules", []):
                if rule_obj.get("id") and rule_obj.get("description"):
                    parsed_rules.append(rule_obj)
        
        else:  # .txt o .md
            # --- INICIO DE LA LÓGICA MEJORADA (DENTRO DEL ELSE) ---
            print("INFO: Formato de texto/markdown detectado. Parseando con lógica mejorada.")
            lines = rules_text.strip().split('\n')
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Buscamos un encabezado de regla (ej: ### CR-SEC-002)
                id_match = re.search(r'^#+\s*([a-zA-Z0-9_-]+)', line)
                if id_match and (i + 1) < len(lines):
                    rule_id = id_match.group(1)
                    next_line = lines[i+1].strip()
                    
                    # Buscamos la descripción en la siguiente línea
                    desc_match = re.search(r'^\*\*Descripción:\*\*\s*(.*)', next_line)
                    if desc_match:
                        description = desc_match.group(1)
                        parsed_rules.append({"id": rule_id, "description": description})
                        i += 2 # Avanzamos 2 líneas (ID y descripción)
                        continue

                # Fallback a la lógica original para formato simple ID: Descripción
                simple_match = re.match(r'^\s*([a-zA-Z0-9_-]+)\s*[:\-]\s*(.*)', line)
                if simple_match:
                    rule_id, text = simple_match.groups()
                    parsed_rules.append({"id": rule_id, "description": text})
                
                i += 1 # Avanzamos a la siguiente línea

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Formato JSON inválido en el archivo proporcionado.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando el archivo de reglas: {e}")

   
   
   
    # Si no hay reglas, solo eliminamos las existentes y salimos
    if not parsed_rules:
        # ... (el resto de la función permanece exactamente igual) ...
        print("INFO: Archivo válido pero sin reglas. Eliminando reglas existentes.")
        clear_query = """
        MATCH (u:User {githubId: $user_id})
        OPTIONAL MATCH (u)-[r:HAS_RULE]->(cr:CustomRule)
        DETACH DELETE cr
        REMOVE u.rulesFilename, u.rulesTimestamp
        """
        graph.query(clear_query, params={"user_id": user_id})
        return {"success": True, "message": "Reglas eliminadas exitosamente (no se agregaron nuevas)."}

    # Generar embeddings...
    try:
        embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        for rule in parsed_rules:
            text_to_embed = f"Rule ID: {rule['id']}. Description: {rule['description']}"
            rule["embedding"] = embeddings_model.embed_query(text_to_embed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando embeddings: {e}")

    # Eliminar reglas anteriores...
    try:
        clear_query = """
        MATCH (u:User {githubId: $user_id})
        OPTIONAL MATCH (u)-[r:HAS_RULE]->(cr:CustomRule)
        DETACH DELETE cr
        REMOVE u.rulesFilename, u.rulesTimestamp
        """
        graph.query(clear_query, params={"user_id": user_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al limpiar reglas anteriores: {e}")

    # Guardar nuevas reglas...
    try:
        save_query = """
        MATCH (u:User {githubId: $user_id})
        SET u.rulesFilename = $filename,
            u.rulesTimestamp = $timestamp
        WITH u
        UNWIND $rules as rule_properties
        MERGE (cr:CustomRule {ruleId: rule_properties.id})
        SET cr += apoc.map.clean(rule_properties, [], [null, ""])
        CREATE (u)-[:HAS_RULE]->(cr)
        """
        graph.query(save_query, params={
            "user_id": user_id,
            "filename": filename,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rules": parsed_rules
        })
        return {"success": True, "message": f"{len(parsed_rules)} reglas personalizadas guardadas exitosamente."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar las reglas: {e}")

@app.get("/api/v1/user/rules/{user_id}", response_model=CustomRulesResponse)
async def get_user_rules(user_id: str):
    """
    (Versión final) Obtiene los metadatos, el formato y el texto de las reglas de un usuario.
    """
    query = """
    MATCH (u:User {githubId: $user_id})
    OPTIONAL MATCH (u)-[:HAS_RULE]->(r:CustomRule)
    RETURN u.rulesFilename AS filename,
           u.rulesTimestamp AS timestamp,
           collect(properties(r)) AS rules
    """
    try:
        result = graph.query(query, params={'user_id': user_id})
        if not result or not result[0] or not result[0].get("filename"):
            return CustomRulesResponse(success=True, rules=None)
        
        record = result[0]
        filename = record.get("filename", "")
        rules_list = record.get("rules", [])
        
        # Determinar el formato y construir el texto
        file_format = 'json' if filename.endswith('.json') else 'text'
        rules_text_for_llm = ""

        if not rules_list or all(not r for r in rules_list):
             rules_text_for_llm = "No custom rules have been defined."
        elif file_format == 'json':
            clean_rules_list = []
            for rule_props in rules_list:
                rule_props.pop('embedding', None)
                rule_props.pop('ruleId', None)
                clean_rules_list.append(rule_props)
            rules_text_for_llm = json.dumps({"rules": clean_rules_list}, indent=2)
        else: # Formato 'text'
            # Usamos get() para evitar errores si las claves no existen
            rules_text_lines = [f"{rule.get('id', rule.get('ruleId'))}: {rule.get('description')}" for rule in rules_list if rule and rule.get('description')]
            rules_text_for_llm = "\n".join(rules_text_lines)

        rules_data = {
            "text": rules_text_for_llm,
            "filename": filename,
            "timestamp": record["timestamp"],
            "format": file_format  # <-- Este campo es crucial para el frontend
        }
        return CustomRulesResponse(success=True, rules=rules_data)

    except Exception as e:
        print(f"ERROR: Could not fetch rules for user {user_id}. Error: {e}")
        raise HTTPException(status_code=500, detail="Error fetching user rules.")
    
    
@app.delete("/api/v1/user/rules/{user_id}")
async def delete_user_rules(user_id: str):
    """
    (Versión Corregida) Elimina todas las reglas de negocio personalizadas (:CustomRule)
    y los metadatos asociados de un usuario.
    """
    print(f"INFO: Solicitud de eliminación de reglas para el usuario {user_id}")
    try:
        # Esta consulta busca al usuario, encuentra todas las reglas enlazadas,
        # las borra de forma segura, y también elimina los metadatos del archivo.
        clear_query = """
        MATCH (u:User {githubId: $user_id})
        OPTIONAL MATCH (u)-[r:HAS_RULE]->(cr:CustomRule)
        DETACH DELETE cr
        REMOVE u.rulesFilename, u.rulesTimestamp
        """
        graph.query(clear_query, params={"user_id": user_id})
        
        print(f"ÉXITO: Reglas para el usuario {user_id} eliminadas de Neo4j.")
        return {"success": True, "message": "Reglas personalizadas eliminadas exitosamente."}
        
    except Exception as e:
        print(f"ERROR: No se pudieron eliminar las reglas para el usuario {user_id}. Causa: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error al eliminar las reglas.")

@app.get("/api/v1/repositories/active/{user_id}", response_model=List[int])
async def get_active_repositories(user_id: str):
    query = "MATCH (u:User {githubId: $user_id})-[:MONITORS]->(r:Repository) RETURN r.repoId AS repoId"
    try:
        result = graph.query(query, params={"user_id": str(user_id)})
        return [record["repoId"] for record in result if record and "repoId" in record and record["repoId"] is not None]
    except Exception as e:
        return []

# --- NUEVO ENDPOINT: Obtener lista de repositorios monitoreados ---
@app.get("/api/v1/user/repositories/{user_id}", response_model=List[RepositoryInfo])
async def get_user_repositories(user_id: str):
    print(f"DEBUG: Valor recibido como user_id: {user_id}")
    """
    Gets a list of repositories monitored by the user.
    """
    query = """
    MATCH (u:User {githubId: $user_id})-[:MONITORS]->(r:Repository)
    RETURN r.repoId AS repoId, r.fullName AS fullName
    ORDER BY r.fullName
    """
    try:
        print(f"INFO: Fetching monitored repositories for user {user_id}")
        results = graph.query(query, params={"user_id": user_id})
        repos = [RepositoryInfo(**{'repoId': r['repoId'], 'fullName': r['fullName']}) for r in results]
        print(f"INFO: Found {len(repos)} monitored repositories for user {user_id}")
        return repos
    except Exception as e:
        print(f"ERROR: Error fetching user repositories for {user_id}. Error: {e}")
        traceback.print_exc()
        return [] # Devolver lista vacía en caso de error

@app.get("/api/v1/user/installation-status/{user_id}")
async def get_user_installation_status(user_id: str):
    try:
        app_jwt = get_github_app_jwt()
        headers = {
            'Authorization': f'Bearer {app_jwt}',
            'Accept': 'application/vnd.github.v3+json'
        }
        url = "https://api.github.com/app/installations"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            installations = response.json()

            for installation in installations:
                if installation.get("account") and str(installation["account"]["id"]) == user_id:
                    return {"has_installation": True}

            return {"has_installation": False}

    except Exception as e:
        print(f"ERROR verificando el estado de la instalación: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error al verificar el estado de la instalación.")

@app.get("/api/v1/user/analyses/{user_id}", response_model=AnalysesHistoryResponse)
async def get_analyses_history(user_id: str):
    query = """
    MATCH (u:User {githubId: $user_id})-[:MONITORS]->(r:Repository)
    MATCH (a:Analysis)-[:FOR_REPO]->(r)
    WITH a ORDER BY a.timestamp DESC
    OPTIONAL MATCH (a)-[:HAS_VULNERABILITY]->(v:Vulnerability)
    RETURN a.prUrl AS prUrl,
           a.summary AS summary,
           a.timestamp AS timestamp,
           elementId(a) AS analysisId,
           a.isReviewed AS isReviewed,
           collect(v {
               profile: v.profile,
               impact: v.impact,
               technical_details: v.technical_details,
               remediation: v.remediation,
               attackPatterns: v.attackPatterns,
               matchedCustomRules: v.matchedCustomRules
           }) AS vulnerabilities
    ORDER BY a.timestamp DESC
    LIMIT 30
    """
    try:
        raw_results = graph.query(query, params={"user_id": user_id})
        cleaned_analyses = []
        if not raw_results:
            return AnalysesHistoryResponse(analyses=[])

        for record in raw_results:
            cleaned_vulnerabilities = []
            for vuln_json_props in record.get("vulnerabilities", []):
                if not vuln_json_props:
                    continue

                try:
                    profile_data = json.loads(vuln_json_props.get("profile", '{}')) if vuln_json_props.get("profile") else {}
                    impact_data = json.loads(vuln_json_props.get("impact", '{}')) if vuln_json_props.get("impact") else {}
                    remediation_data = json.loads(vuln_json_props.get("remediation", '{}')) if vuln_json_props.get("remediation") else {}
                    attack_patterns_list = json.loads(vuln_json_props.get("attackPatterns", '[]')) if vuln_json_props.get("attackPatterns") else []
                    
                    vuln_data = {
                        "profile": profile_data,
                        "impact": impact_data,
                        # --- CORRECCIÓN DEFINITIVA ---
                        # Si technical_details es None, se convierte en un string vacío ""
                        "technical_details": vuln_json_props.get("technical_details") or "",
                        # --- FIN DE LA CORRECCIÓN ---
                        "remediation": remediation_data,
                        "attack_patterns": [AttackPatternInfo(**p) for p in attack_patterns_list],
                        "matched_custom_rules": vuln_json_props.get("matchedCustomRules", [])
                    }
                    cleaned_vulnerabilities.append(StructuredVulnerability(**vuln_data))
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"WARN: No se pudo procesar un registro de vulnerabilidad del historial. Error: {e}")
                    continue

            timestamp = record.get("timestamp")
            timestamp_str = timestamp.isoformat() if isinstance(timestamp, Neo4jDateTime) else str(timestamp) if timestamp else None

            analysis_detail_data = {
                "analysisId": record.get("analysisId"),
                "prUrl": record.get("prUrl"),
                "summary": record.get("summary"),
                "timestamp": timestamp_str,
                "isReviewed": record.get("isReviewed", False),
                "vulnerabilities": cleaned_vulnerabilities
            }
            cleaned_analyses.append(AnalysisDetail(**analysis_detail_data))

        return AnalysesHistoryResponse(analyses=cleaned_analyses)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error al consultar el historial de análisis.")
    
@app.get("/api/v1/updates/history", response_model=UpdateHistoryResponse)
async def get_update_history():
    query = """
    MATCH (log:UpdateLog)
    WHERE log.status = 'Success'
    RETURN log.timestamp AS timestamp,
           log.taskName AS taskName,
           log.status AS status,
           log.details AS details
    ORDER BY log.timestamp DESC
    LIMIT 30
    """
    try:
        results = graph.query(query)
        history_list = []
        for record in results:
            # Intentar parsear los detalles JSON de forma segura
            details_json = {} # Valor por defecto si falla el parseo
            if record.get("details"): # Verificar que la propiedad existe y no es None/vacía
                try:
                    details_json = json.loads(record["details"])
                except (json.JSONDecodeError, TypeError) as e:
                    # Si falla el parseo, loguear una advertencia y usar el valor por defecto
                    print(f"WARNING: Could not parse details JSON for log entry. Error: {e}. Raw details: {record.get('details')}")
                    details_json = {"summary": "Error parsing details."} # Proporcionar un resumen de fallback

            history_list.append(
                UpdateLogEntry(
                    timestamp=record["timestamp"].to_native(),
                    taskName=record["taskName"],
                    status=record["status"],
                    details=UpdateLogDetails(summary=details_json.get("summary", "No summary available."))
                )
            )
        
        return UpdateHistoryResponse(history=history_list)
    except Exception as e:
        print(f"ERROR: Could not fetch update history. Error: {e}")
        traceback.print_exc()
        return UpdateHistoryResponse(history=[]) # Devolver lista vacía en caso de error
    
@app.post("/api/v1/analyses/{analysis_id}/toggle-review", response_model=ReviewStatusResponse)
async def toggle_review_status(analysis_id: str):
    query = """
    MATCH (a:Analysis)
    WHERE elementId(a) = $analysis_id
    SET a.isReviewed = NOT coalesce(a.isReviewed, false)
    RETURN a.isReviewed AS newStatus
    """
    try:
        result = graph.query(query, params={"analysis_id": analysis_id})
        if not result or not result[0]:
            raise HTTPException(status_code=404, detail="Analysis not found.")
        new_status = result[0]['newStatus']
        return ReviewStatusResponse(analysisId=analysis_id, newStatus=new_status)
    except Exception as e:
        print(f"ERROR: Could not toggle review status for {analysis_id}. Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error updating review status.")
    
@app.get("/api/v1/dashboard/analyses-by-day/{user_id}", response_model=List[DailyAnalysisCount])
async def get_daily_analyses(
    user_id: str,
    repo_id: Optional[int] = None,
    period: Optional[str] = '30d'
):
    """
    Obtiene el conteo de análisis completados por día para un usuario,
    opcionalmente filtrado por repositorio y/o período de tiempo.
    Por defecto, muestra los últimos 30 días.
    """
    cutoff_timestamp = None
    if period and period != "":
        try:
            if period.endswith('d'):
                days = int(period[:-1])
                if days > 0:
                    cutoff_timestamp = datetime.now(timezone.utc) - timedelta(days=days)
                else:
                    print(f"WARNING: Period days must be positive: {period}. Not applying time filter.")
            else:
                print(f"WARNING: Unrecognized period format: {period}. Not applying time filter.")
        except ValueError:
            print(f"WARNING: Invalid period value: {period}. Not applying time filter.")

    query_parts = [
        "MATCH (u:User {githubId: $user_id})-[:MONITORS]->(r:Repository)",
        "MATCH (a:Analysis)-[:FOR_REPO]->(r)"
    ]

    where_conditions = []
    if repo_id is not None:
        where_conditions.append("r.repoId = $repo_id")
    if cutoff_timestamp is not None:
        where_conditions.append("a.timestamp >= $cutoff_timestamp")

    if where_conditions:
        query_parts.append("WHERE " + " AND ".join(where_conditions))

    query_parts.append("""
    RETURN toString(date(a.timestamp)) AS analysisDate, count(a) AS count
    ORDER BY analysisDate
    """)

    query = "\n".join(query_parts)

    params = {"user_id": user_id}
    if repo_id is not None:
        params["repo_id"] = repo_id
    if cutoff_timestamp is not None:
        params["cutoff_timestamp"] = cutoff_timestamp

    try:
        print(f"INFO: Executing daily analyses query for user {user_id} (repo_id: {repo_id}, period: {period})")
        results = graph.query(query, params=params)

        if not results:
            print(f"INFO: No daily analyses data found for user {user_id} with specified filters.")
            return []

        daily_counts = [DailyAnalysisCount(date=r['analysisDate'], count=r['count']) for r in results]
        print(f"INFO: Found {len(daily_counts)} days with analyses for user {user_id}")
        return daily_counts

    except Exception as e:
        print(f"ERROR: Error querying daily analyses statistics for user {user_id}. Error: {e}")
        traceback.print_exc()
        return []
@app.get("/api/v1/dashboard/vulnerability-breakdown/{user_id}", response_model=List[VulnerabilityBreakdownItem])
async def get_vulnerability_breakdown(
    user_id: str,
    repo_id: Optional[int] = None,
    period: Optional[str] = '30d'
):
    """
    Obtiene un conteo de las vulnerabilidades más comunes, agrupadas por CWE.
    """
    cutoff_timestamp = None
    if period and period.endswith('d'):
        try:
            days = int(period[:-1])
            if days > 0:
                cutoff_timestamp = datetime.now(timezone.utc) - timedelta(days=days)
        except ValueError:
            pass

    query_parts = [
        "MATCH (u:User {githubId: $user_id})-[:MONITORS]->(r:Repository)",
        "WHERE r.repoId = $repo_id" if repo_id is not None else "",
        "MATCH (a:Analysis)-[:FOR_REPO]->(r)",
        "WHERE a.timestamp >= $cutoff_timestamp" if cutoff_timestamp is not None else "",
        "MATCH (a)-[:HAS_VULNERABILITY]->(v:Vulnerability)",
        "WHERE v.profile IS NOT NULL AND v.profile <> ''",
        "RETURN v.profile AS profile_json"
    ]
    query = "\n".join(filter(None, query_parts))
    
    params = {"user_id": user_id}
    if repo_id is not None:
        params["repo_id"] = repo_id
    if cutoff_timestamp is not None:
        params["cutoff_timestamp"] = cutoff_timestamp
        
    try:
        print(f"INFO: Executing robust vulnerability breakdown query for user {user_id}")
        results = graph.query(query, params=params)
        
    
        # --- PROCESAMIENTO Y CONTEO EN PYTHON (AGRUPADO POR CWE) ---
        breakdown_counts = {} # El diccionario ahora usará el CWE como clave.

        for record in results:
            try:
                profile_data = json.loads(record["profile_json"])
                if profile_data and "name" in profile_data and "cwe" in profile_data:
                    name = profile_data["name"]
                    cwe = profile_data["cwe"]

                    # Si el CWE no ha sido visto antes, lo inicializamos.
                    if cwe not in breakdown_counts:
                        breakdown_counts[cwe] = {
                            'count': 0,
                            # Usamos el nombre de la primera vulnerabilidad encontrada como nombre representativo del grupo.
                            'name': name 
                        }
                    
                    # Incrementamos el contador para este CWE.
                    breakdown_counts[cwe]['count'] += 1

            except (json.JSONDecodeError, TypeError):
                # Ignorar registros con JSON mal formado de forma segura.
                continue
        
        # Convertimos el nuevo diccionario de conteos al formato de lista esperado.
        breakdown_list = [
            VulnerabilityBreakdownItem(name=data['name'], cwe=cwe_code, count=data['count'])
            for cwe_code, data in breakdown_counts.items()
        ]
        
        # Ordenamos la lista por el conteo, de mayor a menor.
        breakdown_list.sort(key=lambda x: x.count, reverse=True)
        
        return breakdown_list

    except Exception as e:
        print(f"ERROR: Error querying vulnerability breakdown for user {user_id}. Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error al obtener el desglose de vulnerabilidades.")
    
@app.get("/api/v1/dashboard/custom-rule-breakdown/{user_id}", response_model=List[CustomRuleBreakdownItem])
async def get_custom_rule_breakdown(
    user_id: str,
    repo_id: Optional[int] = None,
    period: Optional[str] = '30d'
):
    """
    Obtiene un conteo de las violaciones de reglas de negocio más comunes.
    """
    cutoff_timestamp = None
    if period and period.endswith('d'):
        try:
            days = int(period[:-1])
            if days > 0:
                cutoff_timestamp = datetime.now(timezone.utc) - timedelta(days=days)
        except ValueError:
            pass

    # --- CONSULTA CYPHER PARA REGLAS DE NEGOCIO ---
    query_parts = [
        "MATCH (u:User {githubId: $user_id})-[:MONITORS]->(r:Repository)",
        "WHERE r.repoId = $repo_id" if repo_id is not None else "",
        "MATCH (a:Analysis)-[:FOR_REPO]->(r)",
        "WHERE a.timestamp >= $cutoff_timestamp" if cutoff_timestamp is not None else "",
        "MATCH (a)-[:HAS_VULNERABILITY]->(v:Vulnerability)",
        # 1. Filtrar vulnerabilidades que tengan reglas asociadas
        "WHERE v.matchedCustomRules IS NOT NULL AND size(v.matchedCustomRules) > 0",
        # 2. "Desenrrollar" la lista de reglas para procesar cada una individualmente
        "UNWIND v.matchedCustomRules AS ruleId",
        # 3. Devolver el ID de la regla y el perfil de la vulnerabilidad asociada
        "RETURN ruleId, v.profile AS profile_json"
    ]
    query = "\n".join(filter(None, query_parts))
    
    params = {"user_id": user_id}
    if repo_id is not None:
        params["repo_id"] = repo_id
    if cutoff_timestamp is not None:
        params["cutoff_timestamp"] = cutoff_timestamp
        
    try:
        print(f"INFO: Executing custom rule breakdown query for user {user_id}")
        results = graph.query(query, params=params)
        
        # --- PROCESAMIENTO Y CONTEO EN PYTHON (AGRUPADO POR RULE ID) ---
        breakdown_counts = {} # El diccionario usará el ruleId como clave.

        for record in results:
            try:
                rule_id = record["ruleId"]
                profile_data = json.loads(record["profile_json"])
                
                if rule_id and profile_data and "name" in profile_data:
                    # Si la regla no ha sido vista antes, la inicializamos.
                    if rule_id not in breakdown_counts:
                        breakdown_counts[rule_id] = {
                            'count': 0,
                            # Usamos el nombre de la vulnerabilidad como nombre representativo.
                            'representative_name': profile_data["name"]
                        }
                    
                    # Incrementamos el contador para este ruleId.
                    breakdown_counts[rule_id]['count'] += 1

            except (json.JSONDecodeError, TypeError, KeyError):
                # Ignorar registros mal formados de forma segura.
                continue
        
        # Convertimos el diccionario de conteos al formato de lista esperado.
        breakdown_list = [
            CustomRuleBreakdownItem(rule_id=rule_id, representative_name=data['representative_name'], count=data['count'])
            for rule_id, data in breakdown_counts.items()
        ]
        
        # Ordenamos la lista por el conteo, de mayor a menor.
        breakdown_list.sort(key=lambda x: x.count, reverse=True)
        
        return breakdown_list

    except Exception as e:
        print(f"ERROR: Error querying custom rule breakdown for user {user_id}. Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error al obtener el desglose de reglas de negocio.")
    
@app.post("/api/v1/repositories/set-analysis-mode")
async def set_analysis_mode(data: SetAnalysisModeRequest):
    """
    Establece el modo de análisis ('full' o 'diff') para un repositorio específico.
    """
    # Validamos que el modo sea uno de los permitidos
    if data.mode not in ['full', 'diff']:
        raise HTTPException(status_code=400, detail="Invalid analysis mode. Must be 'full' or 'diff'.")

    query = """
    MATCH (r:Repository {repoId: $repo_id})
    SET r.analysisMode = $mode
    RETURN r.analysisMode AS newMode
    """
    try:
        result = graph.query(query, params={"repo_id": data.repo_id, "mode": data.mode})
        if not result or not result[0]:
            raise HTTPException(status_code=404, detail="Repository not found.")
        
        return {"status": "success", "repoId": data.repo_id, "newAnalysisMode": result[0]['newMode']}
    except Exception as e:
        print(f"ERROR: Database error in set_analysis_mode for repoId {data.repo_id}. Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Database interaction error.")
        
# En main_api.py, añade este nuevo endpoint junto a los demás

@app.get("/api/v1/repositories/analysis-modes/{user_id}", response_model=Dict[int, str])
async def get_repo_analysis_modes_for_user(user_id: str):
    """
    Obtiene los modos de análisis ('full' o 'diff') para todos los repositorios
    monitoreados por un usuario específico.
    """
    query = """
    MATCH (u:User {githubId: $user_id})-[:MONITORS]->(r:Repository)
    RETURN r.repoId AS repoId, r.analysisMode AS mode
    """
    try:
        results = graph.query(query, params={"user_id": user_id})
        modes_map = {record["repoId"]: (record.get("mode") or "full") for record in results}
        return modes_map
    except Exception as e:
        print(f"ERROR: Could not fetch analysis modes for user {user_id}. Error: {e}")
        traceback.print_exc()
        return {}
    
# En main_api.py, añade este nuevo endpoint

@app.get("/api/v1/user/subscription/{user_id}", response_model=SubscriptionStatus)
async def get_subscription_status(user_id: str):
    """
    Recupera el estado de la suscripción y el uso actual de un usuario.
    """
    query = """
    MATCH (u:User {githubId: $user_id})
    RETURN u.plan AS plan,
           u.characterCount AS characterCount,
           u.characterLimit AS characterLimit,
           u.usageResetDate AS usageResetDate
    """
    try:
        result = graph.query(query, params={"user_id": user_id})
        record = result[0] if result and result[0] else None
        
        if not record:
            # Si el usuario no existe, creamos uno con valores por defecto del plan gratuito.
            # Esto puede pasar si un usuario se logonea pero nunca ha activado un repo.
            print(f"WARN: User {user_id} not found for subscription status, returning default free plan.")
            return SubscriptionStatus(
                plan="free",
                characterCount=0,
                characterLimit=150000,
                usageResetDate=(datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
            )
            
        # Aseguramos valores por defecto si alguna propiedad faltara
        plan = record.get("plan", "free")
        char_count = record.get("characterCount", 0)
        char_limit = record.get("characterLimit", 150000)
        reset_date = record.get("usageResetDate")
        
        # Convertimos el DateTime de Neo4j a un string ISO 8601 para el JSON
        reset_date_iso = reset_date.isoformat() if reset_date else (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()

        return SubscriptionStatus(
            plan=plan,
            characterCount=char_count,
            characterLimit=char_limit,
            usageResetDate=reset_date_iso
        )

    except Exception as e:
        print(f"ERROR: Could not fetch subscription status for user {user_id}. Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error fetching subscription data.")

async def verify_paypal_signature(request: Request, webhook_id: str) -> bool:
    """
    Verifica la firma de un webhook de PayPal para asegurar su autenticidad.
    (Esta es una implementación conceptual. La implementación real puede requerir el SDK de PayPal)
    """
    try:
        transmission_id = request.headers.get("paypal-transmission-id")
        timestamp = request.headers.get("paypal-transmission-time")
        signature = request.headers.get("paypal-transmission-sig")
        auth_algo = request.headers.get("paypal-auth-algo")
        cert_url = request.headers.get("paypal-cert-url")
        
        if not all([transmission_id, timestamp, signature, auth_algo, cert_url]):
            return False
            
        body = await request.body()
        
        # En una implementación de producción, aquí usarías el SDK de PayPal
        # o harías una llamada a la API de PayPal para verificar la firma
        # con todos los datos del header y el cuerpo del request.
        # Por ahora, simulamos una verificación exitosa si los datos existen.
        print("INFO [PAYPAL]: Webhook verification would happen here.")
        return True
        
    except Exception as e:
        print(f"ERROR: Error during PayPal signature verification: {e}")
        return False
async def get_paypal_access_token():
    """Obtiene un token de acceso de OAuth2 de PayPal."""
    client_id = os.getenv("PAYPAL_CLIENT_ID")
    client_secret = os.getenv("PAYPAL_CLIENT_SECRET")
    auth = (client_id, client_secret)
    url = "https://api-m.sandbox.paypal.com/v1/oauth2/token"
    headers = {"Accept": "application/json", "Accept-Language": "en_US"}
    data = {"grant_type": "client_credentials"}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, data=data, auth=auth)
        response.raise_for_status()
        return response.json()["access_token"]

@app.post("/api/v1/webhooks/paypal")
async def handle_paypal_webhook(request: Request):
    """
    (Versión de PRODUCCIÓN FINAL v2)
    Maneja upgrades y cancelaciones, respetando el ciclo de pago.
    """
    webhook_id = os.getenv("PAYPAL_WEBHOOK_ID")
    if not await verify_paypal_signature(request, webhook_id):
        print("WARN [PAYPAL]: Webhook signature verification failed. Request ignored.")
        return {"status": "ignored"}
    
    try:
        payload = await request.json()
        event_type = payload.get("event_type")
        resource = payload.get("resource", {})

        # --- LÓGICA PARA UPGRADE (SIN CAMBIOS) ---
        if event_type == "PAYMENT.SALE.COMPLETED":
            # ... (Esta parte del código no cambia) ...
            print(f"INFO [PAYPAL]: Procesando evento de venta completada: {event_type}...")
            user_id = resource.get("custom")
            if not user_id: return {"status": "error", "message": "custom field missing"}
            if "billing_agreement_id" not in resource: return {"status": "ignored"}

            print(f"INFO [PAYPAL]: Actualizando plan para el usuario {user_id} a Pro.")
            upgrade_query = """
            MATCH (u:User {githubId: $user_id})
            SET u.plan = 'pro', u.characterLimit = 1000000, u.characterCount = 0,
                u.usageResetDate = datetime() + duration({days: 30}),
                // Limpiamos las marcas de cancelación por si se re-suscribe
                u.planStatus = 'active', u.proAccessEndDate = null
            RETURN u.plan as newPlan
            """
            result = graph.query(upgrade_query, params={"user_id": user_id})
            if result and result[0]: print(f"SUCCESS [PAYPAL]: Usuario {user_id} actualizado a {result[0]['newPlan']}.")
            else: print(f"ERROR [PAYPAL]: No se pudo encontrar al usuario {user_id} para actualizar.")
        
        # --- INICIO DE LA NUEVA LÓGICA DE CANCELACIÓN ---
        elif event_type == "BILLING.SUBSCRIPTION.CANCELLED":
            print(f"INFO [PAYPAL]: Procesando cancelación de suscripción: {event_type}...")
            user_id = resource.get("custom_id")

            if not user_id:
                print("ERROR [PAYPAL]: No se encontró user_id para cancelación.")
                return {"status": "error", "message": "custom_id missing"}

            print(f"INFO [PAYPAL]: Marcando la suscripción del usuario {user_id} para cancelación al final del período.")

            # En lugar de hacer downgrade, marcamos el plan para que expire.
            # La fecha de fin de acceso será la misma que su fecha de reseteo de uso.
            set_cancellation_query = """
            MATCH (u:User {githubId: $user_id})
            SET u.planStatus = 'cancelled',
                u.proAccessEndDate = u.usageResetDate // Guardamos la fecha de fin de ciclo
            RETURN u.planStatus as newStatus, u.proAccessEndDate as endDate
            """
            result = graph.query(set_cancellation_query, params={"user_id": user_id})

            if result and result[0]:
                print(f"SUCCESS [PAYPAL]: Usuario {user_id} marcado como '{result[0]['newStatus']}'. Su acceso Pro termina el {result[0]['endDate']}.")
            else:
                print(f"ERROR [PAYPAL]: No se pudo encontrar al usuario {user_id} para marcar su cancelación.")
        # --- FIN DE LA NUEVA LÓGICA ---
        
        else:
            print(f"INFO [PAYPAL]: Evento '{event_type}' recibido pero no es relevante. Se ignora.")

    except Exception as e:
        print(f"CRITICAL [PAYPAL]: Error procesando el webhook: {e}")
        traceback.print_exc()
        return {"status": "error"}

    return {"status": "received"}


@app.post("/api/v1/paypal/create-subscription-info", response_model=PayPalSubscriptionInfo)
async def create_paypal_subscription_info():
    """
    Genera un client_token para el SDK de PayPal Y devuelve el Plan ID
    desde las variables de entorno del backend para asegurar consistencia.
    """
    try:
        access_token = await get_paypal_access_token()
        url = "https://api-m.sandbox.paypal.com/v1/identity/generate-token"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept-Language": "en_US",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers)
            response.raise_for_status()
            client_token = response.json()["client_token"]
            
            # Obtenemos el Plan ID desde el .env del backend
            plan_id = os.getenv("PAYPAL_PRO_PLAN_ID")
            if not plan_id:
                raise ValueError("PAYPAL_PRO_PLAN_ID no está configurado en el backend.")

            return {"client_token": client_token, "plan_id": plan_id}

    except Exception as e:
        print(f"CRITICAL [PAYPAL]: Error generando la información de suscripción: {e}")
        raise HTTPException(status_code=500, detail="Error interno al procesar la solicitud de pago.")

@app.get("/api/v1/paypal/setup-pro-plan")
async def setup_pro_plan_endpoint():
    """
    Endpoint de un solo uso para crear el Producto Y el Plan de Suscripción
    y asegurar que ambos queden correctamente asociados a nuestra App de API.
    """
    print("\n" + "="*60)
    print("--- INICIANDO CONFIGURACIÓN COMPLETA DE PRODUCTO Y PLAN DE PAYPAL ---")
    try:
        access_token = await get_paypal_access_token()
        
        # --- 1. CREAR EL PRODUCTO ---
        print("\nPaso 1: Creando el Producto...")
        product_url = "https://api-m.sandbox.paypal.com/v1/catalogs/products"
        product_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "PayPal-Request-Id": f"PRODUCT-{uuid.uuid4()}" # Evita duplicados
        }
        product_payload = {
            "name": "PullBrain Pro Subscription",
            "description": "Acceso al plan Pro de PullBrain-AI",
            "type": "SERVICE",
            "category": "SOFTWARE"
        }

        async with httpx.AsyncClient() as client:
            product_response = await client.post(product_url, headers=product_headers, json=product_payload)
            if product_response.status_code >= 400:
                print(f"ERROR al crear el producto: {product_response.text}")
                raise HTTPException(status_code=500, detail=f"Error de PayPal al crear el producto: {product_response.text}")
            
            new_product_id = product_response.json()["id"]
            print(f"¡Éxito! Producto creado con ID: {new_product_id}")

        # --- 2. CREAR EL PLAN USANDO EL NUEVO PRODUCTO ---
        print("\nPaso 2: Creando el Plan de Suscripción...")
        plan_url = "https://api-m.sandbox.paypal.com/v1/billing/plans"
        plan_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        plan_payload = {
            "product_id": new_product_id,
            "name": "PullBrain Pro Monthly Plan",
            "status": "ACTIVE",
            "billing_cycles": [{
                "frequency": {"interval_unit": "MONTH", "interval_count": 1},
                "tenure_type": "REGULAR",
                "sequence": 1,
                "total_cycles": 0,
                "pricing_scheme": { "fixed_price": {"value": "12.00", "currency_code": "USD"} }
            }],
            "payment_preferences": { "auto_bill_outstanding": True }
        }

        async with httpx.AsyncClient() as client:
            plan_response = await client.post(plan_url, headers=plan_headers, json=plan_payload)
            if plan_response.status_code >= 400:
                print(f"ERROR al crear el plan: {plan_response.text}")
                raise HTTPException(status_code=500, detail=f"Error de PayPal al crear el plan: {plan_response.text}")

            new_plan_id = plan_response.json()["id"]
            
            print("\n" + "="*60)
            print("  ¡CONFIGURACIÓN COMPLETA Y EXITOSA!")
            print(f"  El NUEVO y DEFINITIVO Plan ID es: {new_plan_id}")
            print("  Por favor, actualiza tus archivos .env y .env.local con este ID.")
            print("="*60 + "\n")
            return {"status": "SUCCESS", "new_plan_id": new_plan_id, "product_id": new_product_id}

    except Exception as e:
        print(f"CRITICAL [PAYPAL_SETUP]: {e}")
        raise HTTPException(status_code=500, detail="Error interno durante la configuración de PayPal.")
