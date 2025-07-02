import pdfplumber
import os

def extract_index_from_pdf(pdf_path: str, max_pages: int = 20) -> str:
    """
    Extrae el texto del índice de las primeras páginas de un PDF.

    Args:
        pdf_path (str): Ruta al archivo PDF.
        max_pages (int): Número de páginas a leer desde el inicio.

    Returns:
        str: Texto extraído de las primeras páginas.
    """
    index_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i in range(min(len(pdf.pages), max_pages)):
            page = pdf.pages[i]
            text = page.extract_text()
            if text:
                index_text += f"--- Página {i + 1} ---\n{text}\n\n"
    return index_text

def save_index_to_txt(text: str, output_path: str):
    """
    Guarda el texto extraído a un archivo .txt.

    Args:
        text (str): Texto a guardar.
        output_path (str): Ruta del archivo de salida.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Índice guardado en: {output_path}")

if __name__ == "__main__":
    # Cambia la ruta si es necesario
    pdf_path = "./data/algoritmos/pdf/IntroToAlgo.pdf"
    output_path = "./src/utils/kg/index.txt"

    print(f"📖 Extrayendo índice desde: {pdf_path}")
    raw_index = extract_index_from_pdf(pdf_path, max_pages=13)
    save_index_to_txt(raw_index, output_path)