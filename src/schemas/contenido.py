from pydantic import BaseModel
from typing import List, Dict

class ExplicacionTeorica(BaseModel):
    concepto: str
    definicion: str
    caracteristicas: List[str]
    complejidad: str
    casos_uso: list[str]
    analogia: str

class EjemploCodigo(BaseModel):
    problema: str
    lenguaje: str = "python"
    codigo: str
    explicacion: str
    variantes: List[str]

class ProblemaPractico(BaseModel):
    id: str
    dificultad: str
    enunciado: str
    solucion_referencia: str
    casos_prueba: List[dict]
    temas_relacionados: List[str]