from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from agents import supervisor_agent, teoria_agent, ejemplo_agent, practica_agent, retrieval_agent
from utils.bdi_evaluator import evaluar_y_actualizar_bdi
from schemas.estado import EstadoConversacion, TipoAyuda
from agents.bdi_agent import BDIAgent

def crear_workflow_tutor(llm: ChatGoogleGenerativeAI, vector_store: FAISS):
    # Crear agentes
    supervisor_agent_instance = supervisor_agent.crear_supervisor(llm)
    teoria_agent_instance = teoria_agent.crear_agente_teoria(llm)
    ejemplo_agent_instance = ejemplo_agent.crear_agente_ejemplos(llm)
    practica_agent_instance = practica_agent.crear_agente_practica(llm)
    bdi_agent = BDIAgent(llm)
    
    retrieval_agent_instance = retrieval_agent.crear_agente_retrieval(vector_store, llm)
    
    # Definir el grafo
    graph = StateGraph(EstadoConversacion)
    
    async def iniciar_bdi_node(estado):
        return await iniciar_bdi(estado, bdi_agent)
    
    graph.add_node("iniciar_bdi", iniciar_bdi_node)
    graph.add_node("supervisor", supervisor_agent_instance)
    graph.add_node("retrieval", retrieval_agent_instance)
    graph.add_node("teoria", teoria_agent_instance)
    graph.add_node("ejemplo", ejemplo_agent_instance)
    graph.add_node("practica", practica_agent_instance)
    async def evaluar_respuesta_node(estado):
        return await evaluar_y_actualizar_bdi(estado, bdi_agent, llm)
    graph.add_node("evaluar_respuesta", evaluar_respuesta_node)
    
    # Establecer punto de entrada
    graph.set_entry_point("iniciar_bdi")
    
    # Definir transiciones
    def decidir_ruta(estado):
        decision = estado.tipo_ayuda_necesaria
        if decision == TipoAyuda.TEORIA:
            return "teoria"
        elif decision == TipoAyuda.EJEMPLO:
            return "ejemplo"
        elif decision == TipoAyuda.PRACTICA:
            return "practica"
        elif decision == TipoAyuda.FINALIZAR:
            return END
        else:
            raise ValueError(f"Ruta no válida desde supervisor: tipo_ayuda_necesaria={decision}")
        
    # Añadir bordes
    graph.add_edge("iniciar_bdi", "supervisor")
    graph.add_edge("supervisor", "retrieval")
    graph.add_conditional_edges(
        "retrieval",
        decidir_ruta,
        {
            TipoAyuda.TEORIA: "teoria",
            TipoAyuda.EJEMPLO: "ejemplo",
            TipoAyuda.PRACTICA: "practica",
            TipoAyuda.FINALIZAR: END
        }
    )
    graph.add_edge("teoria", "evaluar_respuesta")
    graph.add_edge("ejemplo", "evaluar_respuesta")
    graph.add_edge("practica", "evaluar_respuesta")
    
    # Evaluar y decidir si continuar
    def decidir_continuar(estado):
        # print("[Debug] Evaluación BDI:", estado.ultima_evaluacion)
        if estado.ultima_evaluacion and estado.bdi_state:
            progreso = bdi_agent.evaluate_progress(estado.ultima_evaluacion)
            # print(f"[INFO] Evaluación del progreso: {progreso}")
            if progreso:
                # print("[INFO] Progreso suficiente alcanzado. Finalizando...")
                return END
        # print("[INFO] Continuando con siguiente ciclo.")
        return "supervisor"
    
    graph.add_conditional_edges(
        "evaluar_respuesta",
        decidir_continuar,
        {
            "supervisor": "supervisor",
            END: END
        }
    )
    
    return graph.compile()

async def iniciar_bdi(estado: EstadoConversacion, bdi_agent: BDIAgent):
    # Inicializar el agente BDI
    await bdi_agent.generate_desires(estado.tema)
    await bdi_agent.plan_intentions()
    
    # Guardar el estado BDI en la conversación
    estado.bdi_state = bdi_agent.state
    
    return estado