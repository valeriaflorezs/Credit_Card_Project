ESTADO DE LA CORRIDA 5 x 5 (se actualiza durante la corrida)

Donde se corre cada modelo:
- PC (CPU), results/experiments_5x5__local.csv: Naive Bayes, regresion logistica, arbol de decision, Random Forest y XGBoost (este ultimo corre despues de RF).
- Colab T4 (GPU, cuML), results/experiments_5x5__colab_<modelo>.csv: SVM y KNN. KNN termino sus 16 combinaciones; SVM va por 15 de 16.
- XGBoost se intento primero en Colab, pero la T4 (2 vCPU) no lo sacaba en tiempo razonable y se movio a la PC.
- Respaldo: si Colab no completa KNN, logs/wait_then_knn_local.py lo corre en la PC al terminar XGBoost.
Remote Control: bloqueado por la organizacion (no se pudo activar).
Colab: sesiones manejadas con logs/colab/sync_loop.sh (latido, descarga de CSV, recreacion si se pierde).

SI SE ACABAN LOS TOKENS (como continuar):
1. Local: python -u -m src.run_local_5x5 y despues python -u -m src.run_local_5x5 xgboost (reanuda solo).
2. Colab: logs/colab/setup_session.sh svm|knn crea la T4, instala RAPIDS y lanza la corrida; o notebooks/12_colab_5x5.ipynb.
3. Al terminar: python -m src.merge_session_results y ver EXPERIMENT_STATUS.md (pasos pendientes).

PAUSA (2026-11-01, noche): las 112 combinaciones 5 x 5 estan completas y unidas. Los resultados de 3 x 3 se movieron a results/legacy_3x3/ para que ningun script reanude con datos viejos.
Hecho: merge (results/experiments_master_5x5.csv y results/svm_gpu_results_colab_5x5.csv), posthoc_metrics 96/96.
Pendiente, en este orden (todo reanuda solo, desde la raiz con el entorno creditcard):
1. python -u -m src.svm_comparability   (4/16 hechas, ~5 min por combinacion)
2. python -u -m src.anytime_study       (reanuda; omite lo que ya esta en results/anytime)
3. python -m src.master_table           (necesita posthoc y svm_comparability terminados para el PR-AUC de SVM)
4. python -u -m src.interpretability_multi   (necesita master_table)
5. Reejecutar notebooks 05, 06, 08, 09; actualizar textos con cifras 3 x 3 (introduccion.md, portada.md, README.md, notebooks 00/01/02/07/10); reconstruir el libro.
Nota: KNN en GPU difiere del reajuste en CPU hasta ~0.007 de F1 (posthoc f1_check); documentarlo en 08.
Lanzar procesos largos con PowerShell Start-Process con -WorkingDirectory (bash con nohup se muere al cerrar la llamada).
