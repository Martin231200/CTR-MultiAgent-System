import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import operator
import os

# 1. Definición del Estado Compartido
class AgentState(TypedDict):
    messages: Annotated[Sequence[str], operator.add]
    file_path: str
    mse: float

# --- HERRAMIENTAS (TOOLS) ---

@tool
def encode_categorical_data(file_path: str) -> str:
    """Convierte ad_type en números."""
    df = pd.read_csv(file_path)
    le = LabelEncoder()
    if 'ad_type' in df.columns:
        df['ad_type_encoded'] = le.fit_transform(df['ad_type'])
        df.to_csv(file_path, index=False)
        return "Categorías codificadas."
    return "Error: No se encontró ad_type."

@tool
def standardize_data(file_path: str) -> str:
    """Escala time_spent para que el modelo no se confunda."""
    df = pd.read_csv(file_path)
    scaler = StandardScaler()
    if 'time_spent' in df.columns:
        df['time_spent_scaled'] = scaler.fit_transform(df[['time_spent']])
        df.to_csv(file_path, index=False)
        return "Datos estandarizados."
    return "Error: No se encontró time_spent."

@tool
def train_compare_models(file_path: str):
    """Compara Regresión Lineal y Random Forest. Elige el mejor."""
    df = pd.read_csv(file_path)
    X = df[['ad_type_encoded', 'time_spent_scaled']]
    y = df['ctr']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenamiento
    model_lr = LinearRegression().fit(X_train, y_train)
    mse_lr = mean_squared_error(y_test, model_lr.predict(X_test))

    model_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    mse_rf = mean_squared_error(y_test, model_rf.predict(X_test))

    # Selección
    best_model = model_lr if mse_lr < mse_rf else model_rf
    df['predicted_ctr'] = best_model.predict(X)
    df.to_csv(file_path, index=False)
    
    mejor_mse = min(mse_lr, mse_rf)
    winner = "Lineal" if mse_lr < mse_rf else "Random Forest"
    return f"Ganador: {winner} con MSE: {mejor_mse:.4f}", mejor_mse

@tool
def plot_predictions(file_path: str) -> str:
    """Genera la gráfica de resultados."""
    df = pd.read_csv(file_path)
    plt.figure(figsize=(8, 6))
    plt.scatter(df['ctr'], df['predicted_ctr'], color='blue', alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--', lw=2)
    plt.title("Actual vs. Predicted CTR")
    plt.xlabel("Actual")
    plt.ylabel("Predicción")
    plt.savefig("ctr_scatter_plot.png")
    plt.close()
    return "Gráfica guardada como ctr_scatter_plot.png."

# --- AGENTES (NODOS) ---

def eda_agent(state: AgentState):
    print("--- TRABAJANDO EDA ---")
    res1 = encode_categorical_data.invoke({"file_path": state["file_path"]})
    res2 = standardize_data.invoke({"file_path": state["file_path"]})
    return {"messages": [f"EDA: {res1} {res2}"]}

def stats_agent(state: AgentState):
    print("--- TRABAJANDO ESTADÍSTICA ---")
    info, mse_val = train_compare_models.invoke({"file_path": state["file_path"]})
    return {"messages": [info], "mse": mse_val}

def viz_agent(state: AgentState):
    print("--- TRABAJANDO VISUALIZACIÓN ---")
    res = plot_predictions.invoke({"file_path": state["file_path"]})
    return {"messages": [res]}

# --- CONFIGURACIÓN DEL GRAFO ---

def should_visualize(state: AgentState):
    # Si el error es menor a 0.5, graficamos. Si no, terminamos.
    return "Visualization_Expert" if state["mse"] < 0.5 else END

workflow = StateGraph(AgentState)
workflow.add_node("EDA_Expert", eda_agent)
workflow.add_node("Statistician", stats_agent)
workflow.add_node("Visualization_Expert", viz_agent)

workflow.set_entry_point("EDA_Expert")
workflow.add_edge("EDA_Expert", "Statistician")
workflow.add_conditional_edges("Statistician", should_visualize)
workflow.add_edge("Visualization_Expert", END)

app = workflow.compile()

# --- INTERFAZ WEB (DEPLOY) ---

def launch_app(file_obj):
    temp_path = "dataset_actual.csv"
    # Leemos el archivo que subas en la web
    df = pd.read_csv(file_obj.name)
    df.to_csv(temp_path, index=False)
    
    # Corremos el sistema multi-agente
    final_state = app.invoke({"file_path": temp_path, "messages": [], "mse": 0.0})
    
    logs = "\n".join(final_state["messages"])
    image = "ctr_scatter_plot.png" if os.path.exists("ctr_scatter_plot.png") else None
    return logs, image

demo = gr.Interface(
    fn=launch_app,
    inputs=gr.File(label="Sube tu archivo .csv"),
    outputs=[gr.Textbox(label="Logs del Sistema"), gr.Image(label="Gráfica de CTR")],
    title="Predictor CTR Multi-Agente"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)