import re
from schemas.bdi import Belief, Desire, Intention, BDIState
from utils.optimizador_ag import optimizar_plan

def extract_json_block(text: str) -> str:
    """
    Extrae el bloque JSON de una respuesta tipo ```json ... ``` o lo devuelve tal cual si ya es JSON plano.
    """
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text.strip()

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
                success_criteria={
                    "comprension": 0.0,
                    "precision": 0.0,
                    "profundidad": 0.0
                }
            ),
            intentions=Intention(
                action_plan=[]
            )
        )
    
    async def update_beliefs(self, student_response: str, performance_data: dict):
        prompt = f"""
        Eres un sistema inteligente de tutoría.

        Actualiza las creencias sobre el estudiante basándote en:
        - Respuesta reciente: {student_response}
        - Evaluación del desempeño: {performance_data}
        - Estado previo de creencias: {self.state.beliefs.model_dump_json()}

        Genera
        - Un mapa de conocimiento del estudiante con temas y niveles de comprensión (0-1)
        - Preferencias de aprendizaje detectadas
        - Conceptos erróneos identificados
        Formato JSON con:
        {{
            "student_knowledge": dict[str, success_criteria],
            "learning_preferences": str[],
            "misconceptions": str[]
        }}
        donde `success_criteria` es un dict con claves "comprension", "precision", "profundidad" y valores entre 0.0 y 1.0.
        """
        updated_beliefs = await self.llm.ainvoke(prompt)
        updated_beliefs = extract_json_block(updated_beliefs.content)
        updated_beliefs = Belief.model_validate_json(updated_beliefs)
        self.state.beliefs = updated_beliefs
    
    async def generate_desires(self, learning_objective: str):
        prompt = f"""
        Como tutor de algoritmos, define objetivos pedagógicos para:
        - Objetivo principal: {learning_objective}
        - Creencias actuales: {self.state.beliefs.model_dump_json()}

        Genera:
        1. 1 objetivo principal SMART
        2. 3 objetivos secundarios
        3. 3 criterios de éxito medibles, con valores entre 0.0 y 1.0 para:
        - Comprensión (comprension)
        - Precisión (precision)
        - Profundidad de conocimiento (profundidad)

        Formato JSON con: primary_goal, secondary_goals, success_criteria
        """
        llm = self.llm.with_structured_output(Desire)
        updated_desires = await llm.ainvoke(prompt)
        self.state.desires = updated_desires

    
    async def plan_intentions(self):
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
        llm = self.llm.with_structured_output(Intention)
        new_intentions = await llm.ainvoke(prompt)
        self.state.intentions = new_intentions

        # Aquí optimizamos el action plan con la metaheurística
        estado_estudiante = self.state.beliefs
        plan_original = self.state.intentions.action_plan
        plan_optimizado = await optimizar_plan(plan_original, estado_estudiante)
        self.state.intentions.action_plan = plan_optimizado
            
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
        dims = ["comprension", "precision", "profundidad"]
        valores = [assessment.get(dim, 0.0) for dim in dims]
        promedio = sum(valores) / len(valores)

        # print(f"[Debug] Promedio BDI: {promedio:.2f}")
        return promedio >= 0.8  # aquí decides el umbral

    
    async def handle_failure(self):
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
        await self.plan_intentions()