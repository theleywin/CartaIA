from pydantic import BaseModel, Field
from typing import Dict, List
from enum import Enum

class TipoAyuda(str, Enum):
    TEORIA = "teoria"
    EJEMPLO = "ejemplo"
    PRACTICA = "practica"
    FINALIZAR = "finalizar"

class SuccessCriteria(BaseModel):
    comprension: float = Field(
        description="Nivel de comprensión esperado (0-1)"
    )
    precision: float = Field(
        description="Precisión esperada en la resolución de problemas (0-1)"
    )
    profundidad: float = Field(
        description="Profundidad de conocimiento esperada (0-1)"
    )

class Belief(BaseModel):
    student_knowledge: Dict[str, SuccessCriteria] = Field(..., 
        description="Mapa de temas con nivel de comprensión (0-1)")
    learning_preferences: List[str] = Field(["visual", "practico"], 
        description="Preferencias de aprendizaje detectadas")
    misconceptions: List[str] = Field([],
        description="Conceptos erróneos identificados")

class Desire(BaseModel):
    primary_goal: str = Field(..., 
        description="Objetivo pedagógico principal")
    secondary_goals: List[str] = Field([],
        description="Objetivos secundarios")
    success_criteria: SuccessCriteria = Field(
        description="Criterios de éxito cuantificables"
    )

class Intention(BaseModel):
    action_plan: List[str] = Field(...,
        description="Secuencia de acciones planeadas")
    current_step: int = Field(0,
        description="Paso actual en el plan")
    fallback_strategy: str = Field("simplify_content",
        description="Estrategia cuando el estudiante no progresa")

class BDIState(BaseModel):
    beliefs: Belief
    desires: Desire
    intentions: Intention