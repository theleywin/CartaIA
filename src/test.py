from typing import Dict, List
from pydantic import BaseModel, Field


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
    
#probar ejemplo con string raw
prueba = "{\"student_knowledge\": {\"grafos  asd\": {\"comprensión\": 0.8, \"precision\": 0.7, \"profundidad\": 0.6}, \"algoritmos\": {\"comprension\": 0.9, \"precision\": 0.85, \"profundidad\": 0.75}}, \"learning_preferences\": [\"visual\", \"practico\"], \"misconceptions\": [\"grafos dirigidos\", \"algoritmos de ordenamiento\"]}"
validate = Belief.model_validate_json(prueba)
print(validate)