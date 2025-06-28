from schemas.bdi import TipoAyuda

def print_output(estado_final):
    show_final_result(estado_final)
    show_final_evaluation(estado_final)
    save_bdi_state(estado_final)
    
        
def show_final_result(estado_final):
    print("\n=== RESULTADO DEL TUTOR ===")
    problema = estado_final["problema_actual"]
    tipo = estado_final["tipo_ayuda_necesaria"]
    if tipo == TipoAyuda.TEORIA and estado_final["material"]:
        material = estado_final["material"]
        print("🔍 Explicación teórica:")
        print(material.concepto + "\n" + material.definicion)
    elif tipo == TipoAyuda.EJEMPLO and estado_final["material_ejemplo"]:
        ejemplo = estado_final["material_ejemplo"]
        print("💡 Ejemplo de código:")
        print(ejemplo.problema + "\n" + ejemplo.codigo + "\n" + ejemplo.explicacion)
    elif tipo == TipoAyuda.PRACTICA and problema.enunciado:
        print("📝 Problema para resolver:")
        print(problema.enunciado)
    elif tipo == TipoAyuda.FINALIZAR:
        print("✅ Progreso suficiente alcanzado. Sesión finalizada.")
    else:
        print("⚠️ Tipo de ayuda no reconocido o información incompleta.")
        print(f"Tipo de ayuda recibido: {tipo}")
        
def show_final_evaluation(estado_final):
    if estado_final["ultima_evaluacion"] is not None:
        # print(f"[Debug] ultima evaluacion {estado_final['ultima_evaluacion']}")
        print("\n📊 Evaluación final del estudiante:")
        for criterio, valor in estado_final["ultima_evaluacion"].items():
            print(f" - {criterio}: {valor:.2f}")
    else:
        print("\nℹ️ No hay evaluación final disponible.")
        
def save_bdi_state(estado_final):
    if estado_final["bdi_state"] is not None:
        try:
            with open("estado_bdi.json", "w") as f:
                f.write(estado_final["bdi_state"].model_dump_json())
            print("\n💾 Estado BDI guardado en 'estado_bdi.json'")
        except Exception as e:
            print(f"\n❌ Error al guardar estado BDI: {e}")
    else:
        print("\n⚠️ No hay estado BDI para guardar.")