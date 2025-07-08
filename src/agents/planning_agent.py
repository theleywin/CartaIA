from metaheuristic.ant_colony.aco import AntColony
from metaheuristic.learning_path import LearningPathProblem
from schemas.estado import EstadoConversacion, Planificacion
from langchain_core.language_models.chat_models import BaseChatModel
from utils.kg.build_graph import build_directed_graph

def crear_agente_planificacion(llm: BaseChatModel):
    nodes_file = "nodes.csv"
    strong_edges_file = "strong_edges.csv"
    weak_edges_file = "weak_edges.csv"
    graph = build_directed_graph(nodes_file, strong_edges_file, weak_edges_file)
    
    async def manejar_planificacion(estado: EstadoConversacion) -> EstadoConversacion:
        print("Planificando el estudio ...")
        target_topic = estado.tema
        if not graph.has_node(target_topic):
            return
        
        known_topics = {topic for topic in estado.estado_estudiante.temas_vistos if graph.has_node(topic)}
        weak_topics = {topic for topic in estado.estado_estudiante.errores_comunes if graph.has_node(topic)}

        if len(known_topics) == 0:
            return estado
        
        problem = LearningPathProblem(
            graph=graph,
            known_topics=known_topics,
            weak_topics=weak_topics,
            target_topic=target_topic
        )

        ant_colony = AntColony(
            problem=problem,
            num_ants=80,
            iterations=100
        )
        action_plan, _ = ant_colony.run()
        
        prompt = f"""
        Eres un tutor inteligente de estructuras de datos y algoritmos.
        Elabora el justificaciones en español de no mas de dos lineas para la siguiente lista de temas a aprender
        {action_plan}
        Necesito que tu respuesta justifique por que es importante cada paso dado que
        el estudiante tiene un conocimiento previo de los temas: {", ".join(known_topics)}
        y errores comunes en: {", ".join(weak_topics)}
        Ten en cuenta de que el estudiante desea poder aprender sobre el tema: {target_topic}, por tanto explícale por qué son útiles estos pasos.
        Cada justificación debe ser breve y concisa, no más de dos líneas.
        Devuelvelo en formato JSON con los siguientes campos, que cada elemento del plan sea una cadena de texto, evita comillas innecesarias:
        {{
            "plan": List[str]
        }}
        """
        structured_llm = llm.with_structured_output(Planificacion)
        response = await structured_llm.ainvoke(prompt)
        estado.planificacion = [f"{topic}: {action}" for topic, action in zip(action_plan, response.plan)]
        return estado

    return manejar_planificacion