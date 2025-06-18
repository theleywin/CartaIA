from langchain_google_genai import ChatGoogleGenerativeAI
from schemas.bdi import Belief, Desire, Intention, BDIState
from typing import Dict, Union
import json
import re

def extract_json_block(text: str) -> str:
    """
    Extrae el bloque JSON de una respuesta tipo ```json ... ``` o lo devuelve tal cual si ya es JSON plano.
    """
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text.strip()

def normalize_desires(raw_data: Union[str, dict]) -> dict:
    """
    Transforma el JSON del LLM (string o dict) a un formato plano que puede usar el modelo `Desire`.
    """
    if isinstance(raw_data, str):
        parsed = json.loads(raw_data)
    else:
        parsed = raw_data

    pg = parsed.get("primary_goal", "")
    if isinstance(pg, dict):
        primary_goal = pg.get("description", "")
    else:
        primary_goal = pg

    sg_list = parsed.get("secondary_goals", [])
    secondary_goals = []
    for item in sg_list:
        if isinstance(item, dict):
            secondary_goals.append(item.get("description", str(item)))
        else:
            secondary_goals.append(str(item))

    sc_list = parsed.get("success_criteria", [])
    success_criteria = {}
    for c in sc_list:
        if isinstance(c, dict):
            key = c.get("metric") or c.get("criterion") or "undefined"
            value = c.get("target") or c.get("threshold")
            if isinstance(value, (int, float)):
                success_criteria[key] = float(value)

    return {
        "primary_goal": primary_goal,
        "secondary_goals": secondary_goals,
        "success_criteria": success_criteria,
    }
def normalize_success_criteria(raw_criteria) -> Dict[str, float]:
    """
    Normaliza un diccionario de criterios de éxito, extrayendo solo claves válidas con valores numéricos.
    """
    if not isinstance(raw_criteria, dict):
        return {}

    normalized = {}
    for key, value in raw_criteria.items():
        try:
            normalized[str(key)] = float(value)
        except (ValueError, TypeError):
            continue  # Ignorar si no se puede convertir a float
    return normalized

class BDIAgent:
    def __init__(self, llm):
        self.llm = llm
        self.state = BDIState(
            beliefs=Belief(
                student_knowledge={},
                learning_preferences=[]
            ),
            desires=Desire(
                primary_goal="",
                success_criteria={}
            ),
            intentions=Intention(
                action_plan=[]
            )
        )
    
    def update_beliefs(self, student_response: str, performance_data: dict):
        prompt = f"""
       Eres un sistema inteligente de tutoría.

        Actualiza las creencias sobre el estudiante basándote en:
        - Respuesta reciente: {student_response}
        - Evaluación del desempeño: {performance_data}
        - Estado previo de creencias: {self.state.beliefs.json()}

        Devuelve un JSON con exactamente esta estructura:
        {{
        "student_knowledge": {{
            "NOMBRE_DEL_TEMA": {{
            "comprension": float,
            "precision": float,
            "profundidad": float
            }}
        }},
        "misconceptions": [ "concepto erróneo 1", "concepto erróneo 2" ],
        "learning_preferences": [ "visual", "practico" ]
        }}
        Todos los valores de las dimensiones deben estar entre 0.0 y 1.0.
        NO expliques nada.
        Devuelve solo el JSON. Nada más
        """
        updated_beliefs = self.llm.invoke(prompt).content
        cleaned = extract_json_block(updated_beliefs)
        self.state.beliefs = Belief.parse_raw(cleaned)
    
    def generate_desires(self, learning_objective: str):
        prompt = f"""
        Como tutor de algoritmos, define objetivos pedagógicos para:
        - Objetivo principal: {learning_objective}
        - Creencias actuales: {self.state.beliefs.json()}

        Genera:
        1. 1 objetivo principal SMART
        2. 3 objetivos secundarios
        3. 3 criterios de éxito medibles (ej. {{'comprension': 0.8}})

        Formato JSON con: primary_goal, secondary_goals, success_criteria
        """
        response = self.llm.invoke(prompt).content
        cleaned = extract_json_block(response)

        try:
            normalized = normalize_desires(cleaned)
            self.state.desires = Desire.parse_obj(normalized)
        except Exception as e:
            raise ValueError(f"Error al procesar desires del LLM: {e}\nRespuesta LLM:\n{cleaned}")
    
    def plan_intentions(self):
        prompt = f"""
        Diseña un plan de enseñanza para:
        - Objetivo: {self.state.desires.primary_goal}
        - Nivel estudiante: {self.state.beliefs.student_knowledge}
        - Preferencias: {self.state.beliefs.learning_preferences}
        - Conceptos erróneos: {self.state.beliefs.misconceptions}
        
        El plan debe:
        - Tener máximo 5 pasos
        - Incluir estrategias para abordar conceptos erróneos
        - Considerar 1 estrategia alternativa (solo puede ser uno de: simplify_content, provide_more_examples, switch_learning_style, activate_prior_knowledge)
        
        Formato JSON: 
        {{
            "action_plan": ["paso1", "paso2", ...],
            "fallback_strategy": "nombre_estrategia"
        }}
        """
        new_intentions = self.llm.invoke(prompt).content
        cleaned = extract_json_block(new_intentions)
        new_intentions = json.loads(cleaned)
        self.state.intentions.action_plan = new_intentions.get("action_plan", [])
        self.state.intentions.fallback_strategy = new_intentions.get("fallback_strategy", "simplify_content")
        self.state.intentions.current_step = 0
    
    def execute_next_step(self) -> str:
        if self.state.intentions.current_step >= len(self.state.intentions.action_plan):
            return "end"
        
        action = self.state.intentions.action_plan[self.state.intentions.current_step]
        self.state.intentions.current_step += 1
        return action
    
    def evaluate_progress(self, assessment: dict) -> bool:
        """
        Retorna True si el promedio de las tres dimensiones es al menos 50%.
        """
        # Tomamos sólo las tres dimensiones clave
        dims = ["comprension", "precision", "profundidad"]
        valores = [assessment.get(dim, 0.0) for dim in dims]

        # Cálculo del promedio
        promedio = sum(valores) / len(valores)

        print(f"[Debug] Promedio BDI: {promedio:.2f}")
        return promedio >= 0.5  # aquí decides el umbral

    
    def handle_failure(self):
        strategies = [
            "simplify_content",
            "provide_more_examples",
            "switch_learning_style",
            "activate_prior_knowledge"
        ]
        
        current_strategy = self.state.intentions.fallback_strategy
        next_index = (strategies.index(current_strategy) + 1) % len(strategies)
        
        self.state.intentions.fallback_strategy = strategies[next_index]
        self.state.intentions.current_step = 0
        self.plan_intentions()