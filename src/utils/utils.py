import regex as re

def extract_app_id(url):
    # Buscar URLs con id=
    match = re.search(r'id=([^&]+)', url)
    
    if match:
        return match.group(1)
    
    # Si es una URL de búsqueda, devolver None o lanzar advertencia
    if 'search?q=' in url:
        search_term = re.search(r'q=([^&]+)', url)
        if search_term:
            print(f"⚠️  URL de búsqueda detectada (sin app_id): '{search_term.group(1)}'")
            print(f"   Debes buscar manualmente la app y usar la URL de detalle")
            return None
    
    # URL no válida
    print(f"❌ Error: URL no válida - {url}")
    return None