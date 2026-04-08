# Usa una imagen base oficial de Python 3.10 slim
FROM python:3.10-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia los archivos de tu aplicación al contenedor
COPY . /app

# Actualiza pip e instala las dependencias necesarias
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto 8501 (puerto predeterminado de Streamlit)
EXPOSE 8501

# Comando para ejecutar la aplicación Streamlit, escuchando en todas las interfaces
CMD ["streamlit", "run", "index.py", "--server.port=8501", "--server.address=0.0.0.0"]


# Reviewed and refreshed in 2026
