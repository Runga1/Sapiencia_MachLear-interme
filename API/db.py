"""Sapiencia ML Intermedio - module reviewed and updated in 2026."""

import os
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Solo intenta conectar si hay una base de datos definida
DATABASE_URL = os.getenv("DATABASE_URL")

# Si no hay conexión definida (como en Streamlit Cloud), usar un modo seguro sin conexión real
USE_DATABASE = DATABASE_URL is not None

if USE_DATABASE:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
else:
    engine = None
    SessionLocal = None

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    account = Column(String, index=True)
    probability = Column(Float)
    deadline = Column(String)

def init_db():
    if USE_DATABASE and engine is not None:
        Base.metadata.create_all(bind=engine)
    else:
        print("⚠️ Base de datos no inicializada: DATABASE_URL no configurada")

def get_session():
    if USE_DATABASE and SessionLocal is not None:
        return SessionLocal()
    else:
        print("⚠️ No hay conexión activa a la base de datos")
        return None



