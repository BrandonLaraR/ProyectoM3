import pandas as pd
import numpy as np
import re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Semilla global
np.random.seed(42)

# --- Cargar datos ---
try:
    df_users = pd.read_csv('usuarios_con_cluster_optimizado.csv', encoding='latin-1')
    df_products = pd.read_csv('productos_con_cluster_optimizado.csv', encoding='latin-1')
    df_transactions = pd.read_csv('transactions_alebrije.csv', encoding='latin-1')
    df_rules = pd.read_csv('association_rules.csv', encoding='latin-1')
    print("Archivos cargados exitosamente.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# --- Agregar campos faltantes desde la base de datos ---
try:
    import sqlite3  # Cambia por tu motor si usas PostgreSQL o MySQL
    conn = sqlite3.connect('ruta_a_tu_base_de_datos.db')  # ← Ajusta esto

    df_extra = pd.read_sql_query(
        'SELECT id AS producto_id, precio, imagen_url FROM productos',
        conn
    )
    df_products = df_products.merge(df_extra, on='producto_id', how='left')
    conn.close()
except Exception as e:
    print(f"Error al cargar imagen_url y precio desde la base: {e}")



# --- Convertir frozenset a listas ---
def parse_frozenset(x):
    if pd.isna(x):
        return []
    match = re.match(r"frozenset\(\{(.*)\}\)", x)
    if match:
        items = [item.strip().strip("'") for item in match.group(1).split(',') if item.strip()]
        return items
    try:
        return eval(x) if isinstance(x, str) else x
    except:
        return []

df_rules['antecedents'] = df_rules['antecedents'].apply(parse_frozenset)
df_rules['consequents'] = df_rules['consequents'].apply(parse_frozenset)

# --- Convertir tipos NumPy a tipos nativos de Python ---
def convert_to_native_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    if isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    if pd.isna(obj):
        return None
    return obj

# --- Función para mapear ítems genéricos a productos específicos ---
def map_item_to_products(item, df_products):
    matches = df_products[df_products['tipo'].str.contains(item, case=False, na=False) |
                         df_products['descripcion_producto'].str.contains(item, case=False, na=False)]
    if not matches.empty:
        return matches[['producto_id', 'descripcion_producto', 'Cluster', 'tipo_id', 'categoria_id', 'tipo']].to_dict('records')
    return []

# --- Función para obtener recomendaciones basadas en clústeres ---
def recommend_by_cluster(purchased_items, df_products, purchased_product_ids, top_n=10):
    product_clusters = set()
    purchased_types = set(purchased_items)
    for item in purchased_items:
        matched_products = map_item_to_products(item, df_products)
        for product in matched_products:
            product_clusters.add(convert_to_native_types(product['Cluster']))

    if not product_clusters:
        return [], "No se encontraron clústeres para los ítems comprados"

    cluster_products = df_products[df_products['Cluster'].isin(product_clusters)]
    recommendations = []
    for _, product in cluster_products.iterrows():
        if product['producto_id'] not in purchased_product_ids:
            priority = 1.0 if product['tipo'].upper() in purchased_types else 0.5
            recommendations.append({
    'item': product['tipo'],
    'producto_id': convert_to_native_types(product['producto_id']),
    'descripcion_producto': product['descripcion_producto'],
    'confidence': 0.0,
    'lift': 0.0,
    'product_cluster': convert_to_native_types(product['Cluster']),
    'priority': priority,
    'tipo_id': convert_to_native_types(product.get('tipo_id')),
    'categoria_id': convert_to_native_types(product.get('categoria_id')),
    'precio': convert_to_native_types(product.get('precio')),
    'imagen_url': product.get('imagen_url') or ''
})


    recommendations = sorted(recommendations, key=lambda x: (x['priority'], x['product_cluster'], x['descripcion_producto']), reverse=True)[:top_n]
    return recommendations, f"Recomendaciones basadas en clústeres: {product_clusters}, Priorizando tipos: {purchased_types}"

# --- Función para recomendaciones personalizadas por usuario ---
def recommend_products(user_id, df_users, df_transactions, df_rules, df_products, top_n=10, min_confidence=0.1):
    user_cluster = df_users[df_users['usuario_id'] == user_id]['Cluster'].iloc[0] if 'usuario_id' in df_users.columns and user_id in df_users['usuario_id'].values else None
    if user_cluster is None:
        return [], None, f"Usuario {user_id} no encontrado en usuarios_con_cluster_optimizado.csv"

    user_transactions = df_transactions[df_transactions['usuario_id'] == user_id]['transaction'].apply(
        lambda x: [item.strip() for item in x.split(',') if isinstance(x, str)]
    )

    if user_transactions.empty:
        return [], convert_to_native_types(user_cluster), f"No se encontraron transacciones para usuario {user_id}"

    purchased_items = set(user_transactions.explode().dropna())
    if not purchased_items:
        return [], convert_to_native_types(user_cluster), f"No se encontraron ítems comprados para usuario {user_id}"

    purchased_product_ids = set()
    mapping_debug = []
    for item in purchased_items:
        matched_products = map_item_to_products(item, df_products)
        mapping_debug.append(f"Ítem: {item}, Productos encontrados: {len(matched_products)}")
        for product in matched_products:
            purchased_product_ids.add(convert_to_native_types(product['producto_id']))

    relevant_rules = df_rules[df_rules['confidence'] >= min_confidence]
    recommendations = []
    debug_info = mapping_debug

    for _, rule in relevant_rules.iterrows():
        antecedents = set(rule['antecedents'])
        consequents = set(rule['consequents'])
        if antecedents.issubset(purchased_items):
            for item in consequents:
                matched_products = map_item_to_products(item, df_products)
                if not matched_products:
                    debug_info.append(f"No se encontraron productos para el ítem {item} en la regla {antecedents} → {consequents}")
                    continue
                for product in matched_products:
                    if product['producto_id'] not in purchased_product_ids:
                        recommendations.append({
    'item': item,
    'producto_id': convert_to_native_types(product['producto_id']),
    'descripcion_producto': product['descripcion_producto'],
    'confidence': rule['confidence'],
    'lift': rule['lift'],
    'product_cluster': convert_to_native_types(product['Cluster']),
    'priority': 1.0,
    'tipo_id': convert_to_native_types(product.get('tipo_id')),
    'categoria_id': convert_to_native_types(product.get('categoria_id')),
    'precio': convert_to_native_types(product.get('precio')),
    'imagen_url': product.get('imagen_url') or ''
})

        else:
            debug_info.append(f"Regla descartada {antecedents} → {consequents} porque los antecedentes no coinciden con {purchased_items}")

    if len(recommendations) < top_n:
        cluster_recommendations, cluster_debug = recommend_by_cluster(purchased_items, df_products, purchased_product_ids, top_n=top_n - len(recommendations))
        recommendations.extend(cluster_recommendations)
        debug_info.append(cluster_debug)

    if not recommendations:
        return [], convert_to_native_types(user_cluster), f"No se encontraron recomendaciones para ítems comprados: {purchased_items}. Detalles: {debug_info}"

    recommendations = sorted(recommendations, key=lambda x: (x['priority'], x['confidence'], x['product_cluster'], x['descripcion_producto']), reverse=True)[:top_n]
    return recommendations, convert_to_native_types(user_cluster), f"Recomendaciones generadas para ítems comprados: {purchased_items}. Detalles: {debug_info}"

# --- Función para recomendaciones relacionadas por producto ---
def recommend_related_products(product_id, df_products, top_n=10):
    product_info = df_products[df_products['producto_id'] == product_id]
    if product_info.empty:
        return [], f"Producto {product_id} no encontrado en productos_con_cluster_optimizado.csv"

    product_cluster = convert_to_native_types(product_info['Cluster'].iloc[0])
    product_type = product_info['tipo'].iloc[0].upper()

    cluster_products = df_products[df_products['Cluster'] == product_cluster]
    recommendations = []
    for _, product in cluster_products.iterrows():
        if product['producto_id'] != product_id:
            priority = 1.0 if product['tipo'].upper() == product_type else 0.5
            recommendations.append({
    'item': product['tipo'],
    'producto_id': convert_to_native_types(product['producto_id']),
    'descripcion_producto': product['descripcion_producto'],
    'confidence': 0.0,
    'lift': 0.0,
    'product_cluster': convert_to_native_types(product['Cluster']),
    'priority': priority,
    'tipo_id': convert_to_native_types(product.get('tipo_id')),
    'categoria_id': convert_to_native_types(product.get('categoria_id')),
    'precio': convert_to_native_types(product.get('precio')),
    'imagen_url': product.get('imagen_url') or ''
})


    recommendations = sorted(recommendations, key=lambda x: (x['priority'], x['descripcion_producto']), reverse=True)[:top_n]
    return recommendations, f"Productos relacionados para producto {product_id} (Clúster {product_cluster}, Tipo {product_type})"

# --- Configurar Flask ---
app = Flask(__name__)
CORS(app)

# --- Ruta para la página principal ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Ruta para recomendaciones personalizadas ---
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id', type=int)
    top_n = request.args.get('top_n', default=10, type=int)
    min_confidence = request.args.get('min_confidence', default=0.1, type=float)
    if not user_id:
        return jsonify({'error': 'Se requiere user_id'}), 400
    if top_n < 1 or top_n > 10:
        return jsonify({'error': 'top_n debe estar entre 1 y 10'}), 400
    if min_confidence < 0 or min_confidence > 1:
        return jsonify({'error': 'min_confidence debe estar entre 0 y 1'}), 400

    recommendations, user_cluster, debug_message = recommend_products(user_id, df_users, df_transactions, df_rules, df_products, top_n, min_confidence)
    response = {
        'user_id': user_id,
        'user_cluster': user_cluster,
        'recommendations': recommendations,
        'debug_message': debug_message
    }
    return jsonify(response)

# --- Ruta para recomendaciones relacionadas ---
@app.route('/related_products', methods=['GET'])
def related_products():
    product_id = request.args.get('product_id', type=int)
    top_n = request.args.get('top_n', default=10, type=int)
    if not product_id:
        return jsonify({'error': 'Se requiere product_id'}), 400
    if top_n < 1 or top_n > 10:
        return jsonify({'error': 'top_n debe estar entre 1 y 10'}), 400

    recommendations, debug_message = recommend_related_products(product_id, df_products, top_n)
    response = {
        'product_id': product_id,
        'recommendations': recommendations,
        'debug_message': debug_message
    }
    return jsonify(response)

# --- Iniciar la aplicación ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)