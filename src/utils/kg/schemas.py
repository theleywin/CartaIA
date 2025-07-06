from enum import Enum
from pydantic import BaseModel, Field

class DSATopicType(str, Enum):
    ALGORITHM = "algorithm"
    STRUCT = "structure"
    PROBLEM = "problem"
    THEORY = "theory"
    OTHER = "other"

class EdgeType(str, Enum):
    STRONG = "STRONG"
    WEAK = "WEAK"
    

class TopicMetadata(BaseModel):
    type: DSATopicType = Field(description="Tipo del tema (ALGORITHM, STRUCT, THEORY, PROBLEM, OTHER)")
    difficulty: float = Field(description="Dificultad estimada del tema, de 1.0 (fácil) a 5.0 (difícil)")
    estimated_time: float = Field(description="Tiempo estimado para aprender el tema, en horas")
