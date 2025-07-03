import json

# 1. Función para cargar preguntas desde un archivo JSON
def cargar_json(archivo="questions.json"):
    try:
        with open(archivo, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: El archivo {archivo} no existe.")
        return []
    except json.JSONDecodeError:
        print(f"Error: El archivo {archivo} tiene formato JSON inválido.")
        return []
    

# 2. Función para añadir nuevas preguntas al JSON
def agregar_pregunta(pregunta, dificultad, archivo="questions.json"):
    # Cargar preguntas existentes
    preguntas = cargar_json(archivo)
    
    # Verificar si la pregunta ya existe
    for p in preguntas:
        if p.get('pregunta', '').strip().lower() == pregunta.strip().lower():
            print("⚠️ La pregunta ya existe en el archivo.")
            return False
    
    # Crear nueva pregunta
    nueva_pregunta = {
        "pregunta": pregunta,
        "dificultad": dificultad
    }
    
    # Agregar a la lista
    preguntas.append(nueva_pregunta)
    
    # Guardar en el archivo
    try:
        with open(archivo, 'w', encoding='utf-8') as f:
            json.dump(preguntas, f, indent=2, ensure_ascii=False)
        print("✅ Pregunta añadida exitosamente!")
        return True
    except Exception as e:
        print(f"Error al guardar: {e}")
        return False

# 3. Función para mostrar preguntas con su dificultad
def mostrar_preguntas(preguntas=None, archivo="questions.json"):
    if preguntas is None:
        preguntas = cargar_json(archivo)
    
    if not preguntas:
        print("No hay preguntas disponibles.")
        return
    
    print("\n" + "="*50)
    print(f"📚 LISTADO DE PREGUNTAS ({len(preguntas)} total)")
    print("="*50)
    
    for i, p in enumerate(preguntas, 1):
        # Mapear dificultad a emoji
        dificultad_emoji = {
            "Fácil": "⭐",
            "Media": "⭐⭐",
            "Difícil": "⭐⭐⭐"
        }.get(p.get('dificultad', 'Media'), "⭐")
        
        print(f"\n🔖 Pregunta #{i}:")
        print(f"   {p['pregunta']}")
        print(f"   Dificultad: {dificultad_emoji} ({p.get('dificultad', 'No especificada')})")

def ejecutar_evaluacion_y_guardar(preguntas):
    
    if not preguntas:
        print("No hay preguntas para evaluar.")
        return
    
    resultados = []
    
    # Recorrer todas las preguntas
    for idx, pregunta_data in enumerate(preguntas, 1):
        pregunta_texto = pregunta_data["pregunta"]
        dificultad = pregunta_data.get("dificultad", "Media")
        
        # Obtener respuesta del sistema tutor
        #respuesta = obtener_respuesta_tutor(pregunta_texto, dificultad)
        print(f"   🤖 Respuesta del tutor: respuesta")
        
        # Guardar resultado
        resultados.append({
            "pregunta_id": idx,
            "pregunta": pregunta_texto,
            "dificultad": dificultad,
            "respuesta_tutor": "lol"
        })
    
    # Guardar resultados en archivo JSON
    try:
        with open("results.json", 'w', encoding='utf-8') as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Resultados guardados en results.json")
        print(f"   Total de respuestas: {len(resultados)}")
        return resultados
    except Exception as e:
        print(f"❌ Error al guardar resultados: {e}")
        return None