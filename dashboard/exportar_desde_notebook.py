"""Exporta resultados y figuras del notebook actual a archivos para Streamlit.

Uso al FINAL del notebook, tras entrenar y visualizar todos los clasificadores:

    from exportar_desde_notebook import exportar_dashboard
    exportar_dashboard(globals())

No reentrena los seis modelos evaluados. Únicamente puede reajustar los dos
KNN auxiliares de visualización para usar el k seleccionado en GridSearchCV.
"""
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
except ImportError as exc:  # pragma: no cover - depende del entorno
    raise RuntimeError(
        "Este script requiere matplotlib instalado para generar las visualizaciones."
    ) from exc

from sklearn.tree import plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _nombre_archivo(nombre):
    slug = re.sub(r"[^a-z0-9]+", "_", nombre.lower().strip()).strip("_")
    return slug


def _guardar_frontera(modelo, datos, etiquetas, titulo, xlabel, ylabel, ruta):
    """Dibuja regiones con una malla acotada para no generar millones de puntos."""
    datos = np.asarray(datos, dtype=float)
    etiquetas = np.asarray(etiquetas, dtype=int)
    if datos.ndim != 2 or datos.shape[1] != 2:
        raise ValueError("Las regiones de decisión requieren exactamente 2 columnas.")
    margen_x = max(np.ptp(datos[:, 0]) * 0.04, 0.3)
    margen_y = max(np.ptp(datos[:, 1]) * 0.04, 0.3)
    xmin, xmax = datos[:, 0].min() - margen_x, datos[:, 0].max() + margen_x
    ymin, ymax = datos[:, 1].min() - margen_y, datos[:, 1].max() + margen_y
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 135), np.linspace(ymin, ymax, 135))
    region = modelo.predict(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8.8, 5.9))
    ax.contourf(xx, yy, region, levels=[-0.5, 0.5, 1.5],
                cmap=ListedColormap(["#e6edf4", "#f9e3e7"]), alpha=0.95)
    for estado, etiqueta, color in [(0, "Permanece (0)", "#296FAD"), (1, "Canceló (1)", "#D2172B")]:
        seleccion = etiquetas == estado
        ax.scatter(datos[seleccion, 0], datos[seleccion, 1], c=color,
                   s=9, alpha=0.35, linewidths=0, label=etiqueta)
    ax.set_title(titulo, fontsize=14, weight="semibold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.08)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def exportar_dashboard(ns, carpeta="."):
    """Genera CSV e imágenes fieles a los modelos entrenados en el notebook.

    ns: el resultado de globals() desde una celda del notebook.
    carpeta: por defecto, la carpeta desde la que se ejecuta el notebook.
    """
    base = Path(carpeta).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    imagenes = base / "visualizaciones"
    imagenes.mkdir(exist_ok=True)

    requeridas = ["X", "X_test", "y_test", "X_test_scaled", "modelo_nb", "modelo_dt",
                  "modelo_rf", "modelo_mlp", "modelo_svm", "busqueda_knn"]
    faltan = [v for v in requeridas if v not in ns]
    if faltan:
        raise ValueError("Faltan variables del notebook: " + ", ".join(faltan) +
                         ". Ejecuta primero las celdas de entrenamiento.")

    # Usar SIEMPRE el KNN elegido por GridSearchCV, no una variable mejor_k
    # que pudo ser sobrescrita en una celda antigua.
    busqueda = ns["busqueda_knn"]
    mejor_k = busqueda.best_params_["kneighborsclassifier__n_neighbors"]
    modelo_knn = busqueda.best_estimator_
    test_raw = ns["X_test"]
    test_scaled = ns["X_test_scaled"]
    y_true = np.asarray(ns["y_test"], dtype=int)
    if not set(np.unique(y_true)).issubset({0, 1}) or len(set(y_true)) != 2:
        raise ValueError("La variable churned debe contener clases 0 y 1 en test.")

    modelos = {
        "Naive Bayes": (ns["modelo_nb"], test_raw),
        f"KNN (k={mejor_k})": (modelo_knn, test_raw),
        "Decision Tree": (ns["modelo_dt"], test_raw),
        "Random Forest": (ns["modelo_rf"], test_raw),
        "MLP": (ns["modelo_mlp"], test_scaled),
        "SVM": (ns["modelo_svm"], test_scaled),
    }

    registros = []
    resultados = []
    for nombre, (modelo, X_prueba) in modelos.items():
        pred = np.asarray(modelo.predict(X_prueba), dtype=int)
        if hasattr(modelo, "predict_proba"):
            scores = np.asarray(modelo.predict_proba(X_prueba)[:, 1], dtype=float)
        else:
            scores = np.asarray(modelo.decision_function(X_prueba), dtype=float)
        if len(pred) != len(y_true) or len(scores) != len(y_true):
            raise ValueError(f"Longitud de predicciones incorrecta en {nombre}.")
        resultados.append({
            "Modelo": nombre,
            "Accuracy": accuracy_score(y_true, pred),
            "Precision": precision_score(y_true, pred, zero_division=0),
            "Recall": recall_score(y_true, pred, zero_division=0),
            "F1-Score": f1_score(y_true, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_true, scores),
            "Average Precision": average_precision_score(y_true, scores),
        })
        registros.append(pd.DataFrame({
            "Modelo": nombre,
            "y_real": y_true,
            "y_pred": pred,
            "score": scores,
        }))

    met = pd.DataFrame(resultados).sort_values("F1-Score", ascending=False).reset_index(drop=True)
    met.to_csv(base / "resultados_modelos.csv", index=False)
    pd.concat(registros, ignore_index=True).to_csv(base / "evaluacion_modelos.csv", index=False)

    # Comparación de importancias de los dos modelos de árbol.
    importancias = []
    columnas = list(ns["X"].columns)
    for nombre, clave in [("Decision Tree", "modelo_dt"), ("Random Forest", "modelo_rf")]:
        modelo = ns[clave]
        if len(modelo.feature_importances_) != len(columnas):
            raise ValueError(f"Las columnas no coinciden con el modelo {nombre}.")
        for col, importancia in zip(columnas, modelo.feature_importances_):
            importancias.append({"Modelo": nombre, "Variable": col, "Importancia": float(importancia)})
    pd.DataFrame(importancias).to_csv(base / "importancia_variables.csv", index=False)

    # Gráfico legible de los primeros niveles del árbol original.
    fig, ax = plt.subplots(figsize=(21, 10))
    plot_tree(ns["modelo_dt"], feature_names=columnas, class_names=["Permanece", "Canceló"],
              filled=True, rounded=True, max_depth=3, fontsize=8, ax=ax)
    ax.set_title("Árbol de decisión — primeros niveles", fontsize=16)
    fig.savefig(imagenes / "arbol_decision.png", dpi=145, bbox_inches="tight")
    plt.close(fig)

    manifest = []
    meta = {"mejor_k_gridsearch": int(mejor_k), "nota": "Regiones auxiliares 2D; no son evaluación del modelo completo."}
    y_2d = np.asarray(ns.get("Y_plot", []), dtype=int)

    for tipo, modelos_clave, data_clave, xlabel, ylabel in [
        ("Dos variables", "modelos_2d", "X_plot", "Horas de reproducción", "Días desde el último login"),
        ("PCA", "modelos_pca", "X_pca", "Componente principal 1", "Componente principal 2"),
    ]:
        if modelos_clave not in ns or data_clave not in ns or len(y_2d) == 0:
            print(f"Aviso: faltan {modelos_clave}/{data_clave}; se omiten las regiones de {tipo}.")
            continue
        X_2d = np.asarray(ns[data_clave], dtype=float)
        if X_2d.shape != (len(y_2d), 2):
            raise ValueError(f"Forma inconsistente en {data_clave}; debería ser (n_clientes, 2).")
        visuales = ns[modelos_clave]
        for nombre_antiguo, visual in visuales.items():
            nombre = nombre_antiguo
            if nombre_antiguo.startswith("KNN"):
                nombre = f"KNN (k={mejor_k})"
                # Únicamente se reentrena el KNN ilustrativo en dos dimensiones.
                # NO se afecta al KNN ya validado con GridSearchCV.
                if tipo == "Dos variables":
                    visual = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=int(mejor_k)))
                else:
                    visual = KNeighborsClassifier(n_neighbors=int(mejor_k))
                visual.fit(X_2d, y_2d)
            archivo = f"{ 'region_2d' if tipo == 'Dos variables' else 'pca' }_{_nombre_archivo(nombre)}.png"
            _guardar_frontera(visual, X_2d, y_2d, f"{tipo} — {nombre}",
                             xlabel, ylabel, imagenes / archivo)
            manifest.append({"Tipo": tipo, "Modelo": nombre, "Archivo": archivo})

    pd.DataFrame(manifest, columns=["Tipo", "Modelo", "Archivo"]).to_csv(
        base / "manifest_visualizaciones.csv", index=False)
    if "pca" in ns and hasattr(ns["pca"], "explained_variance_ratio_"):
        meta["varianza_pca_dos_componentes"] = float(np.sum(ns["pca"].explained_variance_ratio_))
    with (base / "resumen_visualizaciones.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n✅ Exportación completada en:", base)
    print("   - resultados_modelos.csv")
    print("   - evaluacion_modelos.csv")
    print("   - importancia_variables.csv")
    print("   - manifest_visualizaciones.csv")
    print("   - visualizaciones/ (árbol + regiones + PCA)")
    print(f"   - KNN seleccionado por validación cruzada: k={mejor_k}")
    print("\nColoca app.py, requirements.txt y netflix_customer_churn.csv en esta carpeta.")
    return met
