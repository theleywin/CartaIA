from .bdi import BDIState, TipoAyuda
from pydantic import BaseModel
from typing import List, Dict, Optional
from schemas.contenido import ProblemaPractico
from schemas.contenido import ExplicacionTeorica

class EstadoEstudiante(BaseModel):
    nivel: str = "principiante"
    temas_vistos: List[str] = []
    errores_comunes: List[str] = []

class EstadoConversacion(BaseModel):
    tema: str
    historial: List[Dict[str, str]] = []
    docs_relevantes: List[str] = []
    tipo_ayuda_necesaria: Optional[TipoAyuda] = None
    material: Optional[ExplicacionTeorica] = None
    problema_actual: Optional[ProblemaPractico] = None
    solucion_estudiante: Optional[str] = None
    estado_estudiante: EstadoEstudiante
    bdi_state: Optional[BDIState] = None
    ultima_evaluacion: Optional[dict] = None
    pasos: int = 0