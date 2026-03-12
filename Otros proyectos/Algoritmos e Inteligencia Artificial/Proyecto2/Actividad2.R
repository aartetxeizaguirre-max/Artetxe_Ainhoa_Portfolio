
############################################################
# Actividad 2: Aprendizaje Supervisado en Datos Biológicos #
# Estudiante: Ainhoa Artetxe Izaguirre                     #
# Fecha: 29/12/2025                                        #
############################################################

### =========================================================
#	1. Preparación del entorno de trabajo.
### =========================================================
install.packages("caret")
install.packages("glmnet")
install.packages("tidyverse")
install.packages("pROC")
library("caret")
library("glmnet")
library("tidyverse")
library("pROC")



### =========================================================
### 2. Cargar el conjunto de datos
### =========================================================
setwd("C:/Users/lenovo/Desktop/BIOINFORMATICA/Algoritmos e Inteligencia Artificial/Actividad2")
# Cargar el nuevo dataset
df <- read.csv("data.csv")

# Limpieza de nombres (importante: tu archivo tiene un tabulador en 'symmetry2')
colnames(df) <- trimws(colnames(df))



### =========================================================
### 3. Selección de variables (LASSO)
### =========================================================
# En este archivo, los predictores van de la columna 3 a la 32
genes <- names(df[3:32])

# x: matriz de datos, y: clase (Maligno/Benigno)
x <- as.matrix(df[, genes])
y <- factor(df$Diagnosis)

# Ejecutamos LASSO con validación cruzada
set.seed(1995)
lasso_model <- cv.glmnet(x, y, family = "binomial", alpha = 1)

# Extraer coeficientes y filtrar los que NO son cero
# Usamos [,1] para referirnos a la columna de coeficientes de forma segura
coefs <- as.matrix(coef(lasso_model, s = "lambda.min"))
coefs_df <- as.data.frame(coefs)

nombres_seleccionados <- rownames(coefs_df)[
  coefs_df[,1] != 0 & rownames(coefs_df) != "(Intercept)"
]

# Mensaje de control para que veas si ha funcionado
print(paste("Variables seleccionadas por LASSO:", length(nombres_seleccionados)))



### =========================================================
### 4. Creación del dataset reducido
### =========================================================
# Creamos el dataframe final con la respuesta y las variables elegidas
data_final <- df %>%
  dplyr::select(Diagnosis, all_of(nombres_seleccionados))

# Ponemos el ID como nombre de fila
rownames(data_final) <- df$ID

# Convertir la respuesta a factor (necesario para modelos de clasificación)
data_final$Diagnosis <- as.factor(data_final$Diagnosis)



### =========================================================
### 5. División en entrenamiento y test 
### =========================================================
set.seed(1995)
trainIndex <- createDataPartition(data_final$Diagnosis, p = 0.8, list = FALSE)

# Creamos los conjuntos (el drop = FALSE evita que se convierta en vector si hay pocas variables)
trainData <- data_final[trainIndex, , drop = FALSE]
testData  <- data_final[-trainIndex, , drop = FALSE]

# --- COMPROBACIÓN FINAL ---
print("Dimensiones de Entrenamiento:")
print(dim(trainData))
print("Dimensiones de Test:")
print(dim(testData))



### =========================================================
### 6. ENTRENAMIENTO DE MODELOS DE CLASIFICACIÓN
### =========================================================
# Configuramos el método de validación (Cross-Validation de 10 carpetas)
# Esto sirve para que el modelo sea más robusto
control <- trainControl(method = "cv", number = 10)

# --- MODELO 1: Regresión Logística ---
set.seed(1995)
fit_glm <- train(Diagnosis ~ ., data = trainData, 
                 method = "glm", family = "binomial", 
                 trControl = control)

# --- MODELO 2: Random Forest ---
set.seed(1995)
fit_rf <- train(Diagnosis ~ ., data = trainData, 
                method = "rf", 
                trControl = control)

# --- MODELO 3: K-Nearest Neighbors (KNN) ---
set.seed(1995)
fit_knn <- train(Diagnosis ~ ., data = trainData, 
                 method = "knn", 
                 trControl = control,
                 preProcess = c("center", "scale")) # KNN requiere escalar datos



### =========================================================
### 7. Evaluación: ¿Cuál es el mejor modelo?
### =========================================================
# Hacemos predicciones sobre el conjunto de TEST (los datos que el modelo no ha visto)
pred_glm <- predict(fit_glm, testData)
pred_rf  <- predict(fit_rf, testData)
pred_knn <- predict(fit_knn, testData)

# Creamos las matrices de confusión para ver el porcentaje de acierto (Accuracy)
cm_glm <- confusionMatrix(pred_glm, testData$Diagnosis)
cm_rf  <- confusionMatrix(pred_rf, testData$Diagnosis)
cm_knn <- confusionMatrix(pred_knn, testData$Diagnosis)

# Mostramos los resultados de Accuracy
print(paste("Precisión Logística:", round(cm_glm$overall['Accuracy'], 4)))
print(paste("Precisión Random Forest:", round(cm_rf$overall['Accuracy'], 4)))
print(paste("Precisión KNN:", round(cm_knn$overall['Accuracy'], 4)))

# Ver el detalle del mejor modelo (KNN)
print(cm_knn)
print(cm_glm)
print(cm_rf)

# Ver estadísticas enfocadas en detectar Malignos (M)
confusionMatrix(pred_knn, testData$Diagnosis, positive = "M")

# 1. Calcular importancia
importancia <- varImp(fit_rf, scale = FALSE)

# 2. Graficar las 10 variables más importantes
plot(importancia, top = 10, main = "Variables Clave para el Diagnóstico")

# Calculamos probabilidades para los 3 modelos
prob_glm <- predict(fit_glm, testData, type = "prob")
prob_rf  <- predict(fit_rf, testData, type = "prob")
prob_knn <- predict(fit_knn, testData, type = "prob")

# Creamos los objetos ROC (enfocados en la clase M)
roc_glm <- roc(testData$Diagnosis, prob_glm$M)
roc_rf  <- roc(testData$Diagnosis, prob_rf$M)
roc_knn <- roc(testData$Diagnosis, prob_knn$M)

# Dibujamos la comparación
plot(roc_knn, col = "blue", main = "Comparativa de Modelos (Curva ROC)")
lines(roc_rf, col = "red")
lines(roc_glm, col = "green")
legend("bottomright", legend = c("KNN", "Random Forest", "Logística"), 
       col = c("blue", "red", "green"), lwd = 2)

# Tabla resumen de AUC (Área bajo la curva)
resumen_auc <- data.frame(
  Modelo = c("Logística", "Random Forest", "KNN"),
  AUC = c(auc(roc_glm), auc(roc_rf), auc(roc_knn)),
  Accuracy = c(cm_glm$overall['Accuracy'], cm_rf$overall['Accuracy'], cm_knn$overall['Accuracy'])
)
print(resumen_auc)



### =========================================================
### 8. CONCLUSIONES
### =========================================================
# 1. OPTIMIZACIÓN Y SELECCIÓN DE VARIABLES:
# La implementación de la regularización LASSO (cv.glmnet) permitió una reducción de 
# dimensionalidad efectiva, filtrando las características más relevantes para el 
# diagnóstico y eliminando el ruido. Esto simplifica el modelo sin perder poder predictivo.

# 2. ROBUSTEZ METODOLÓGICA:
# Se garantizó la fiabilidad de los resultados mediante:
#   - División estratificada (80/20): Mantiene la proporción de clases en entrenamiento y test.
#   - Validación Cruzada (10-fold CV): Asegura que el modelo sea estable y generalizable.
#   - Preprocesamiento: El escalado y centrado fue clave para el alto rendimiento de KNN.

# 3. EVALUACIÓN DE DESEMPEÑO:
# Los tres modelos evaluados presentan un rendimiento sobresaliente (AUC > 0.98):
#   - KNN: Es el modelo superior con un Accuracy del 99.1% y un AUC de 0.999.
#   - Logística: Muestra una gran capacidad de generalización con un Accuracy del 96.4%.
#   - Random Forest: Aunque tiene un AUC excelente (0.995), su precisión es ligeramente 
#     menor (94.6%), sugiriendo que el umbral de clasificación podría optimizarse.

# 4. CONCLUSIÓN FINAL:
# El flujo de trabajo desarrollado es altamente eficaz para la clasificación de 
# diagnósticos. El modelo KNN, bajo estas condiciones de preprocesamiento y selección 
# de variables, se recomienda como la herramienta más precisa para este conjunto de datos.

