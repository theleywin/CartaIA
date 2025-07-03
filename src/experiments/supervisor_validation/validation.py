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
    